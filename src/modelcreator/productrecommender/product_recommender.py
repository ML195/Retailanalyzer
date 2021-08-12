import re
import warnings
from typing import Union
from pathlib import Path
from joblib import Parallel, delayed
from multiprocessing import cpu_count as mp_cpu_count
from logging import Logger

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from path_definitions import PATHS
from modelcreator.exception import CorruptRecommenderError, ModelNotFoundError, NoTrainedModelError
from modelcreator.exception import ModelNotFoundError
from modelcreator import model_creator_utils


class ProductRecommender():
    """ Wrapper for an product recommender.

    The ProductRecommender class provides the following functionality:

        - Creates a kNN-inspired item-based collaborative filtering recommender (evaluated using hit-rate)
        - Loading of pre-existing recommenders via the name attribute
        - Recommending the top n products for a given product () or a given customer ().

    Attributes:
        name (str): Passed name used to identify the recommender .
        _load_model (bool): Passed bool that defines if a pre-existing recommender with the same name should be loaded.
        _logger (Logger): Passed logger to track execution steps or errors. 
        _model_subdirectory (str): Subdirectory name consisting of the class-name.
        _model_path (Path): The path to the directory where the model is stored.
        _evaluation_dir_path (Path): The path to the directory where the model evaluations are stored.
        _evaluation_file_path (Path): The path to the model evaluation file.
        _save_file_format (str): File format used to store the model.
        _item_neighborhood_matrix_name (DataFrame): Name of the item neighborhood matrix (identifier for loading the matrix).
        _customer_item_recommendation_matrix_name (DataFrame): Name of the customer-item recommendation matrix (identifier for loading the matrix).
    """

    def __init__(self, name: str, load_existing_model: bool = False, logger: Logger=None, random_state: int = None):
        """Initializes ProductRecommender.

        Args:
            name (str):
                Name of the product recommender.
            
            load_existing_model (bool):
                Defines if a pre-existing model with the same name should be loaded (True = yes, False = no).

            logger (Logger):
                A logger to track execution steps or errors. 

            random_state (int):
                Random state for reproducible results.
                
        Raises:
            ValueError: If the name does not follow the specifications (only letters, numbers and underscore allowed).
        """

        # check input types
        fun_params = locals()
        fun_params.pop('self', None)
        type_signature = [str, bool, (Logger, type(None)), (int, type(None))]
        model_creator_utils.check_type(type_signature, **fun_params)

        if not re.match(r'^\w+$', name):
            raise ValueError('Please specify a valid name (can contain letters, numbers and underscore.')

        # Set passed attributes
        self.name = name
        self._load_model = load_existing_model
        self._logger = logger
        self._random_state = random_state

        # Set internal attributes
        self._model_subdirectory = self.__class__.__name__
        self._model_path = PATHS['MODELS_DIR'] / self._model_subdirectory / self.name
        self._evaluation_dir_path = PATHS['EVALUATIONS_DIR'] / self._model_subdirectory / str(self.name)
        self._evaluation_file_path = self._evaluation_dir_path / str(self.name + '_evaluation.txt')
        self._item_neighborhood_matrix_name = 'item_neighborhood_matrix'
        self._customer_item_recommendation_matrix_name = 'customer_item_recommendation_matrix'
        self._save_file_format = '.feather'
        
        # Set matrices initially to None
        self._n_item_neighborhood_matrix = None
        self._customer_item_recommendation_matrix = None

        if self._load_model:
            self._load_existing_recommender()
        else:
            # If no pre-existing model should be loaded but a model under the name exists a warning is printed
            if self._model_path.exists():
                warnings.warn("A Model with the name {0} already exists, model is overwritten if .initialize_forecaster() is called".format(self.name), stacklevel=2)



    ####################################################################################################
    # Public functions                                                                                 #
    ####################################################################################################

    def initialize_recommender(self, customer_item_matrix: pd.DataFrame, number_of_recommendations: int):
        """Initializes ProductRecommender instance.

        This function includes the following steps:
        
            - Create a subfolder for the ProductRecommender in the models and evaluation folder if it does not exist already
            - Create an evaluation file for the recommender
            - Initialize a kNNIBCF recommender

        Args:
            customer_item_matrix (DataFrame):
                Customer-item matrix as pd.Dataframe with the shape (customers, items), where customerID is the index and the items' Stock codes are the column names.

            number_of_recommendations (int):
                The number of item recommendations to generate by the recommender for each customer .
        """

        # check input types
        fun_params = locals()
        fun_params.pop('self', None)
        type_signature = [pd.DataFrame, int]
        model_creator_utils.check_type(type_signature, **fun_params)

        if number_of_recommendations > customer_item_matrix.shape[1] or number_of_recommendations <= 0:
            raise ValueError(f'The number of recommendations must be larger than zero and cannot exceed the number of available products. Minamal value for number_of_recommendations is 1. Maximal value for number_of_recommendations is {customer_item_matrix.shape[1]}.')

        if self._load_model:
           print('Recommender '+self.name+' already initialized.')
        else:
        
            # Create the subfolder in models if it does not exist
            self._model_path.mkdir(parents=True, exist_ok=True)

            # Create the subfolders in reports if it does not exist
            self._evaluation_dir_path.mkdir(parents=True, exist_ok=True)

            # Create an evaluation file for the recommender
            with open(self._evaluation_file_path, 'w') as evaluation:
                evaluation.write(model_creator_utils.get_title_line('General Model Info'))

            model_creator_utils.write_to_report_file(f'\tModel-Name: {self.name}', self._evaluation_file_path)

            if self._logger: self._logger.info(f'Initializing kNN-IBCF Recommender {self.name}')
            self._initialize_kNNIBCFRecommender(customer_item_matrix, number_of_recommendations)

            # set model as loaded
            self._load_model = True


    def get_top_recommended_products_for_product(self, stock_code: Union[str, list], ignore_if_not_exists=False) -> Union[pd.Series, pd.DataFrame]:
        """Returns the top products to the given stock_code.

        Args:
            stock_code (str or list):
                The stock code id to which to find the top recommended items. If ``stock_code`` is given as list it should include multiple stock codes for example [stock_code_1, stock_code_1, ...] of type str.

            ignore_if_not_exists (bool):
                Determines what to to when a given ``stock_code`` does not exist. Defaul is False, which raises an error. True will return None. In the case of ``stock_code`` being a list all entries for the not existent stock codes are set to np.nan.

        Returns:
            If passed ``stock_code`` is str the items recommended for the item with the given stock code as pd.Series where the index represents the rank and the values the stock codes of the items. If the passed ``stock_code`` is a list, the recommended items for each product in the list as pd.DataFrame where the index represents the stock codes of the passed products and the columns the recommandation rank with the stock codes of recommended items as values.


        Raises: 
            TypeError: If type is not string.
            ValueError: If ignore_if_not_exists = False and the given stock_code does not exist.
            NoTrainedModelError: If the function is called but no recommender is ready to make recommendations.
        """

        # check if stock_code is string
        if isinstance(stock_code, str):
            if self._load_model:
                # check if the stock_code exists
                if stock_code in self._n_item_neighborhood_matrix.index:
                    return self._n_item_neighborhood_matrix.loc[stock_code]
                
                elif ignore_if_not_exists:
                    return None

                else:
                    raise ValueError('There are no recommendations for the given stock_code, as the item seems to not exist.')
            else:
                raise NoTrainedModelError('There is no trained recommender to get recommendations from, please call initialize_recommender() first or set load_existing_model to True.')
        
        elif isinstance(stock_code, list):
            if not all(isinstance(s, str) for s in stock_code):
                raise TypeError('One or more stock_code elements in the given list have the wrong type (must be of type str).')

            if self._load_model:
            
                # for each customer id get the top n recommendations and store them into a data frame
                item_recs = []
                for item in stock_code:
                    if item in self._n_item_neighborhood_matrix.index:
                        item_recs.append(self._n_item_neighborhood_matrix.loc[item].values)

                    elif ignore_if_not_exists:
                        item_recs.append([np.nan for i in range(self._n_item_neighborhood_matrix.shape[1])])

                    else:
                        raise ValueError('There are no recommendations for the given stock_code, as the item seems to not exist.')
            
                columns = self._n_item_neighborhood_matrix.columns
            
                item_recommendations = pd.DataFrame(data=item_recs, index=stock_code, columns=columns)
                return item_recommendations

            else:
                raise NoTrainedModelError('There is no trained recommender to get recommendations from, please call initialize_recommender() first or set load_existing_model to True.')

        else:
            raise TypeError(f'stock_code has the wrong type: {type(stock_code).__name__} given but expected str.')


    def get_top_recommended_products_for_customer(self, customer_id: Union[float, list], ignore_if_not_exists=False) -> Union[pd.Series, pd.DataFrame]:
        """Returns the top recommended products to the given customer id.

        Args:
            customer_id (float or list):
                The customer id to which to find the top recommended items. If ``customer_id`` is given as list it should include multiple customer IDs for example [id_1, id_2, ...] of type float.

            ignore_if_not_exists (bool):
                 Determines what to to when a given ``customer_id`` does not exist. Defaul is False, which raises an error. True will return None. In the case of ``customer_id`` being a list all entries for the not existent customers are set to np.nan.
                 
        Returns:
            If passed ``customer_id`` is float the items recommended for the customer with the given ``customer_id`` as pd.Series where the index represents the rank and the values the stock codes of the items. If the passed ``customer_id`` is a list, the recommended items for each customer ID in the list as pd.DataFrame where the index represents the customer_id and the columns the recommandation rank with stock codes as values.

        Raises: 
            TypeError: When type of customer_id is neither float nor list-
            ValueError: When ignore_if_not_exists = False and the given customer_id does not exist.
            NoTrainedModelError: If the function is called but no recommender is ready to make recommendations.
        """
        
        if isinstance(customer_id, float):
            if self._load_model:
                # if customer_id is simply a float return the top_n recommendations as series
                if customer_id in self._customer_item_recommendation_matrix.index:
                    return self._customer_item_recommendation_matrix.loc[customer_id]

                elif ignore_if_not_exists:
                    return None
                
                else:
                    raise ValueError('There are no recommendations for the given customer_id, as the customer seems to not exist.')
            else:
                raise NoTrainedModelError('There is no trained recommender to get recommendations from, please call initialize_recommender() first or set load_existing_model to True.')
        
        elif isinstance(customer_id, list):
            # check if all elements in the list are float
            if not all(isinstance(n, float) for n in customer_id):
                raise TypeError('One or more customer_id elements in the given list have the wrong type (must be of type float).')
            
            if self._load_model:
            
                # for each customer id get the top n recommendations and store them into a data frame
                customer_recs = []
                for customer in customer_id:
                    if customer in self._customer_item_recommendation_matrix.index:
                        customer_recs.append(self._customer_item_recommendation_matrix.loc[customer].values)

                    elif ignore_if_not_exists:
                        nan_row = np.empty(self._customer_item_recommendation_matrix.shape[1])
                        nan_row.fill(np.nan)
                        customer_recs.append(nan_row)

                    else:
                        raise ValueError('There are no recommendations for the given customer_id, as the customer seems to not exist.')
            
                columns = self._customer_item_recommendation_matrix.columns
                
                customer_recommendations = pd.DataFrame(data=customer_recs, index=customer_id, columns=columns)
                return customer_recommendations
            else:
                raise NoTrainedModelError('There is no trained recommender to get recommendations from, please call initialize_recommender() first or set load_existing_model to True.')

        else:
            raise TypeError(f'customer_id has the wrong type: {type(customer_id).__name__} given but expected float or list.')

    

    ####################################################################################################
    # Private functions                                                                                #
    ####################################################################################################

    def _load_existing_recommender(self):
        """Loads an existing Recomender.

        Raises: 
            ModelNotFoundError: If there is no pre-existing recommender under the specified path ``_model_path``.
            CorruptRecommenderError: If the some of the required recommender files (matrices) are missing or corrupted.
        """

        if self._logger: self._logger.info(f'({self.name}) Trying to load a pre-existing recommender.')

        if self._model_path.exists():
            # if the model directory for the given name exists create path pointing to the item neighborhood matrix and the customer-item recommendation matrix
            item_neighborhood_matrix_path = self._model_path / str(self._item_neighborhood_matrix_name + self._save_file_format)
            customer_item_recommendation_matrix_path = self._model_path / str(self._customer_item_recommendation_matrix_name + self._save_file_format)

            # if both matrices were stored as files and exist, set the corresponding attributes
            if item_neighborhood_matrix_path.is_file() and customer_item_recommendation_matrix_path.is_file():
                self._n_item_neighborhood_matrix = self._load_matrix(item_neighborhood_matrix_path)
                self._customer_item_recommendation_matrix = self._load_matrix(customer_item_recommendation_matrix_path)
                if self._logger: self._logger.info(f'({self.name}) Recommender successfully loaded.')
            
            # if one or both of the matrices do not exist or are not a file, raise an error
            else:
                if self._logger: self._logger.info(f'({self.name}) Model loading failed: CorruptRecommenderError.')
                self._load_model = False
                raise CorruptRecommenderError()
        
        # if no directory under the given name exists raise an error
        else:
            if self._logger: self._logger.info(f'({self.name}) Model loading failed: ModelNotFoundError.')
            self._load_model = False
            error_message = 'There is no pre-existing recommender with the name '+self.name+', specify an existing recommender or build one with .initialize_recommender().'
            raise ModelNotFoundError(error_message)


    def _initialize_kNNIBCFRecommender(self, customer_item_matrix: pd.DataFrame, number_of_recommendations: int):
        """Initializes an kNNIBCF recommender.

        Wrapper function to encapsulate the steps needed to create an kNNIBCF recommender

        This function includes the following steps:
        
            - Create 5 random splits of train and test data on which the recommender is trained and evaluated
                - the mean hit rate is then used as evaluation metric for the recommender
            - Train the recommender on the whole data and store it to disk

        Args:
            customer_item_matrix (DataFrame):
                Customer-item matrix as pd.Dataframe with the shape (customers, items), where customerID is the index and the items' Stock codes are the column names.

            number_of_recommendations (int):
                The number of item recommendations to generate by the recommender for each customer.
        """

        # lines to write into report file
        report_file_lines = []

        # number of random train-test splits to generate
        n_splits = 5

        # Hit rates across train-test splits
        hit_rates = []
        reciprocal_hit_rates = []

        # How much items to "remove" from the training data and "add" them to the test data      
        test_items_per_customer = 10

        if self._logger: self._logger.info(f'({self.name}) Starting training and testing the recommender on {n_splits} random splits with {test_items_per_customer} items for each customer in the test set.')

        if self._random_state:
            # Define numpy random generator with a seed
            rng = np.random.default_rng(self._random_state)

        else:
            # Define numpy random generator without a seed
            rng = np.random.default_rng()

        # Train and test the recommender with five random splits
        for n in range(n_splits):
            # Split data into train and test set
            train_matrix, test_matrix = self._random_train_test_split(customer_item_matrix, test_items_per_customer, rng)
            
            # compute recommendations based on train set
            train_item_item_similarity_matrix = self._construct_item_item_similarity_matrix(train_matrix)
            train_n_item_neighborhood_matrix = self._construct_n_item_neighborhood_matrix(train_item_item_similarity_matrix, number_of_recommendations)
            
            n_jobs = int(mp_cpu_count()/2)
            train_customer_item_recommendation_matrix = self._construct_customer_item_recommendation_matrix(train_matrix, train_item_item_similarity_matrix, train_n_item_neighborhood_matrix, number_of_recommendations, n_jobs)

            # Evaluate the recommender on test data using hit-rate
            hit_rate = self._calulate_hit_rate(test_matrix, train_customer_item_recommendation_matrix)
            reciprocal_hit_rate = self._calulate_hit_rate(test_matrix, train_customer_item_recommendation_matrix, True)

            hit_rates.append(hit_rate)
            reciprocal_hit_rates.append(reciprocal_hit_rate)

        if self._logger: self._logger.info(f'({self.name}) Testing finished.')
        
        mean_hit_rate = np.mean(hit_rates)
        mean_reciprocal_hit_rate = np.mean(reciprocal_hit_rates)

        if self._logger: self._logger.info(f'({self.name}) Training recommender on whole dataset.')


        # Training the recommender on the whole dataset
        item_item_similarity_matrix = self._construct_item_item_similarity_matrix(customer_item_matrix)
        self._n_item_neighborhood_matrix = n_item_neighborhood_matrix = self._construct_n_item_neighborhood_matrix(item_item_similarity_matrix, number_of_recommendations)
        
        n_jobs = int(mp_cpu_count()/2)

        self._customer_item_recommendation_matrix = self._construct_customer_item_recommendation_matrix(customer_item_matrix, item_item_similarity_matrix, self._n_item_neighborhood_matrix, number_of_recommendations, n_jobs)

        # Write lines to report file
        report_file_lines.append('\tModel-Type: kNN-Inspired Item-Based Collaborative Filtering')
        report_file_lines.append(f'\tHyperparameters: \n\t\t number_of_recommendations: {number_of_recommendations}\n\t\t similarity_function: cosine\n')
        report_file_lines.append(model_creator_utils.get_title_line(f'Mean Hit-Rate ({n_splits} Random Splits) ({test_items_per_customer} Items per Customer in Test Set)'))
        report_file_lines.append(f'\tHit-Rate: {mean_hit_rate:.3%}')
        report_file_lines.append(f'\tReciprocal Hit-Rate: {mean_reciprocal_hit_rate:.3%}')
        model_creator_utils.write_to_report_file(report_file_lines, self._evaluation_file_path)

        if self._logger: self._logger.info(f'({self.name}) Training finished, storing the recommender on disk.')

        # Storing the trained recommender (consisting of itme neighborhood and customer-item recommendation matrix) on disk
        item_neighborhood_save_path = str(self._model_path / str(self._item_neighborhood_matrix_name + self._save_file_format))
        self._save_matrix(self._n_item_neighborhood_matrix, item_neighborhood_save_path)

        customer_item_save_path = str(self._model_path /  str(self._customer_item_recommendation_matrix_name + self._save_file_format))
        self._save_matrix(self._customer_item_recommendation_matrix, customer_item_save_path)


    def _random_train_test_split(self, customer_item_matrix: pd.DataFrame, items_to_remove: int, random_generator) -> tuple:
        """ Splits the data customer item matrix into training and testing matrix.

        The split is done in such a way that ``items_to_remove`` many items are "removed" (set to 0) from each customer (that has at least 2 * ``items_to_remove`` purchases) and "put" (set to 1) in the testing set.
        
        Args:
            customer_item_matrix (DataFrame):
                Customer-item matrix as pd.Dataframe with the shape (customers, items), where customerID is the index and the items' Stock codes are the column names.

            items_to_remove (int):
                The number of items to "remove" from each customer in the training set to put them into the testing set.

            random_generator (Generator):
                Numpy random generator. 

        Returns:
            Tuple of the form (train_matrix, test_matrix).
        """

        # Get unique customers
        unique_customers = customer_item_matrix.index.values

        # Determine the number of customers in the test set (only customers that have at least 2 * items_to_remove items)
        number_of_purchases = customer_item_matrix.sum(axis=1)
        test_customers_indices = number_of_purchases[number_of_purchases >= (2*items_to_remove)].index

        # preset the test and train matrix
        test_matrix = pd.DataFrame(0, index=test_customers_indices, columns=customer_item_matrix.columns)
        train_matrix = customer_item_matrix.copy()
        
        for customer in unique_customers:
            customer_items = customer_item_matrix.loc[customer]
            #print(customer_items)

            # if the customer is a candidate for the test set
            if customer_items.sum() >= (2*items_to_remove):

                # get the all purchased items
                purchased_items = customer_items.loc[customer_items > 0].index.values

                # select items_to_remove many items
                selected_items = random_generator.choice(a=purchased_items, size=items_to_remove, replace=False)

                # "remove" them in the training set by setting them to zero
                train_matrix.loc[customer, selected_items] = 0.0

                # "add" them in the testing set by setting them to one
                test_matrix.loc[customer, selected_items] = 1.0
        
        return  train_matrix, test_matrix
    

    def _calulate_hit_rate(self, test_matrix: pd.DataFrame, customer_item_recommendation_matrix: pd.DataFrame, reciprocal: bool=False) -> float:
        """ Calculates the per customer averaged hit-rate achieved by the recommender.

        Args:
            test_matrix (DataFrame):
                Test customer-item matrix as pd.Dataframe with the shape (customers, items) containing the "removed" items, where customerID is the index and the items' Stock codes are the column names.

            customer_item_recommendation_matrix (DataFrame):
                Matrix containing the top n item recommendations for customers as pd.DataFrame with shape (n_customers, n_items), where the customerID is the index.

            reciprocal (bool):
                Determines if the standard or reciprocal (including the rank of recommended item that achieved a hit) hit-rate should be calculated.

        Returns:
            Tuple of the form (train_matrix, test_matrix).
        """

        unique_customers = test_matrix.index.values
        customer_hit_rates = []

        for customer in unique_customers:
            # hit rate for customer
            customer_hit_rate = 0

            # get purchased items for customer
            customer_items = test_matrix.loc[customer]
            purchased_items = customer_items.loc[customer_items > 0].index.values
            
            # get recommended items for customer
            recommended_items = customer_item_recommendation_matrix.loc[customer].values

            for rank, recommended_item in enumerate(recommended_items):
                # if the recommended item is in the purchased items of the test set register a "hit"
                if recommended_item in purchased_items:
                    if reciprocal:
                        customer_hit_rate = customer_hit_rate + (1/(rank+1))
                    else:
                        customer_hit_rate += 1
            
            # normalize hit rate by the number of maximum achievable hits, namely the number of items included in the test set
            customer_hit_rates.append(customer_hit_rate/purchased_items.shape[0])
        
        total_hit_rate = sum(customer_hit_rates) / len(customer_hit_rates)
        
        return total_hit_rate
        

    def _construct_item_item_similarity_matrix(self, customer_item_matrix: pd.DataFrame) -> pd.DataFrame:
        """Creates the item similarity matrix.

        Constructs an item similarity matrix based on the item similarities computed with cosinus similarity. The matrix is of shape (items, items), where the index and the column names are the items' Stock codes.

        Args:
            customer_item_matrix (DataFrame):
                Customer-item matrix as pd.Dataframe with the shape (customers, items), where customerID is the index and the items' Stock codes are the column names.

        Returns:
            The item similarity matrix as pd.DataFrame with shape (items, items).
        """

        # Remove CustomerID as it is not needed in this matrix
        prepared_customer_item_matrix = customer_item_matrix.reset_index().drop('CustomerID', axis=1)
        
        # get stock codes of products
        columns = prepared_customer_item_matrix.columns

        # calculate cosine-similarity between products
        item_item_similarity = cosine_similarity(prepared_customer_item_matrix.T)
       
        # Set self-similarities to zero
        np.fill_diagonal(item_item_similarity, 0)

        item_item_similarity_matrix = pd.DataFrame(data=item_item_similarity, index=columns, columns=columns)

        return item_item_similarity_matrix
        
    def _construct_n_item_neighborhood_matrix(self, item_item_similarity_matrix: pd.DataFrame, n: int) -> pd.DataFrame:
        """Creates the item neighborhood matrix.

        Constructs an item neighborhood matrix based on the item similarity matrix. The matrix is of shape (items, n), where n is the number of most similar neighbors.  

        Args:
            item_item_similarity_matrix (DataFrame):
                Matrix containing the similarities between items as pd.DataFrame with shape (items, items), where the index and the column names are the items' Stock codes.

            n (int):
                Number of most similar neighbors to include in the matrix.

        Returns:
            The item-neighborhood matrix as pd.DataFrame with shape (items, n), where the index represents the item Stock codes.
        """

        top_n_neighbor_matrix = item_item_similarity_matrix.apply(lambda x: pd.Series(x.sort_values(ascending=False).iloc[:n].index), axis=1)
        top_n_neighbor_matrix.columns = ['top_'+str(x) for x in range(1,n+1)]
   
        return top_n_neighbor_matrix


    def _construct_customer_item_recommendation_matrix(self, customer_item_matrix: pd.DataFrame, item_item_similarity_matrix: pd.DataFrame, n_item_neighborhood_matrix: pd.DataFrame, n: int, n_jobs: int) -> pd.DataFrame:
        """Gets the top n recommended items for a single customer.

        This function is executed in parallel by ``_construct_customer_item_recommendation_matrix()``

        Args:
            customer_item_matrix (DataFrame):
                Customer-item matrix as pd.Dataframe with the shape (customers, items), where customerID is the index and the items' Stock codes are the column names.

            item_item_similarity_matrix (DataFrame):
                Matrix containing the similarities between items as pd.DataFrame with shape (items, items), where the index and the column names are the items' Stock codes.

            n_item_neighborhood_matrix (DataFrame):
                Matrix containing the top n neighbor items for each item as pd.DataFrame with shape (items, n), where the index is represented by the Stock codes.

            n (int):
                Number of top recommended items to return. 
            
            n_jobs (int):
                With how much jobs to run the calculation.

        Returns:
            The customer-item recommendation matrix as pd.DataFrame with shape (customers, top n recommendations) with customerID as index.
        """

        unique_customers = customer_item_matrix.index.values

        # calculate the top n recommendations for each customer in parallel
        result = Parallel(n_jobs=n_jobs)(delayed(self._get_top_n_item_recommendations_to_customer)(n, customer, customer_item_matrix, item_item_similarity_matrix, n_item_neighborhood_matrix) for customer in unique_customers)
        zipped_results = list(map(list, zip(*result)))

        customers = zipped_results[0]
        recommendations = zipped_results[1]
        columns = ['top_'+str(i) for i in range(1, n+1)]

        # create a customer item recommendation matrix 
        customer_item_recommendation_matrix = pd.DataFrame(data = recommendations, index=customers, columns=columns)
        
        return customer_item_recommendation_matrix


    def _get_top_n_item_recommendations_to_customer(self, n: int, customer_id: float, customer_item_matrix: pd.DataFrame, item_item_similarity_matrix: pd.DataFrame, n_item_neighborhood_matrix: pd.DataFrame)-> tuple:
        """Gets the top n recommended items for a single customer.

        This function is executed in parallel by ``_construct_customer_item_recommendation_matrix()``

        Args:
            n (int):
                Number of top recommended items to return. 

            customer_id (float):
                The customer id to which to find the top recommended items.

            customer_item_matrix (DataFrame):
                Customer-item matrix as pd.Dataframe with the shape (customers, items), where customerID is the index and the items' Stock codes are the column names.

            item_item_similarity_matrix (DataFrame):
                Matrix containing the similarities between items as pd.DataFrame with shape (items, items), where the index and the column names are the items' Stock codes.

            n_item_neighborhood_matrix (DataFrame):
                Matrix containing the top n neighbor items for each item as pd.DataFrame with shape (items, n), where the index is represented by the Stock codes.
        
        Returns:
            A tuple containing the ``customer_id`` as foat as well as the ``top_n_recommended_items`` as list containing the item codes (first item is the top recommended one, second item the second top and so on).
        """

        # Get the row to the customer_id
        customer_purchases = customer_item_matrix.loc[customer_id]

        # Filter out the items the customer has purchased
        customer_purchases_indices = customer_purchases.loc[customer_purchases > 0].index.values

        top_n_similar_to_purchases = n_item_neighborhood_matrix.loc[customer_purchases_indices]
        
        # Remove duplicates
        unique_top_n_similar_to_purchases = np.unique(top_n_similar_to_purchases.values)

        # Select the corresponding similarity values 
        similarity_neighbohood = item_item_similarity_matrix.loc[unique_top_n_similar_to_purchases, unique_top_n_similar_to_purchases]

        similarity_customer_purchases = customer_purchases.loc[unique_top_n_similar_to_purchases]

        # Recommendation score based on item-based collaborative filtering
        recommendation_score = similarity_neighbohood.dot(similarity_customer_purchases).div(similarity_neighbohood.sum(axis=1))       

        # Drop already purchased items
        recommendation_score.drop([x for x in customer_purchases_indices if x in recommendation_score.index], inplace=True)

        # Sort by recommendation score and get the top n
        top_n_recommended_items = recommendation_score.sort_values(ascending=False).head(n).index.tolist()
        
        return customer_id, top_n_recommended_items


    def _save_matrix(self, matrix: pd.DataFrame, path: Path):
        """Save the given matrix to the given path as .feather.

        Args:
            matrix (DataFrame):
                The matrix to store as pd.DataFrame

            path (Path):
                The path as Path object to which to store the matrix.
        """

        matrix.reset_index().to_feather(path)


    def _load_matrix(self, path: Path) -> pd.DataFrame:
        """Load a matrix stored as .feather at the given path. 

        Args:
            path (Path):
                The path as Path object from which to load the matrix file.
        
        Returns:
            The matrix as pd.DataFrame.
        """

        matrix = pd.read_feather(path)
        index_col = matrix.columns[0]
        return matrix.set_index(index_col)

