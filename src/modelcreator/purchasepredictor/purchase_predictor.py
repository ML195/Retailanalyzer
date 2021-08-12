import re
import warnings
import pickle
from typing import Union
from pathlib import Path
from operator import itemgetter
from multiprocessing import cpu_count as mp_cpu_count
from logging import Logger

import numpy as np
import pandas as pd

from sklearn.utils import all_estimators
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

from path_definitions import PATHS
from modelcreator.exception import ModelNotFoundError, IncompatibleDataError, NoTrainedModelError
from modelcreator import model_creator_utils


class PurchasePredictor():
    """ Wrapper for a purchase predictor.

    The PurchasePredictor class provides the following functionality:

        - Get the best model in terms of a threshold optimized generalized F-score (fbeta) for given estimators and hyperparameters through grid search with cross-validation
            - The threshold optimized generalized F-score is basically a generalized F-score calculated for a range of thresholds (from 0 to 1 with 0.01 steps), where the maximum score for a specific threshold is then taken as final score.
        - Loading of pre-existing models with via the name attribute
        - Prediction if a customer with a given customer_id will purchase in the next quarter (label = 1) or not (label = 0) using ``predict_if_customer_purchases_next_quarter()``.
        - Get all customers (as customer_id) that are predicted to purchase in the next quarter using ``get_all_customers_purchasing_next_quarter()``.
        - Get all customers (as customer_id) that are predicted to not purchase in the next quarter using ``get_all_customers_not_purchasing_next_quarter()``.

    Attributes:
        name (str): Passed name used to identify the purchase predictor.
        _data (DataFrame): Passed data as pd.DataFrame containing training (column target == 'train') and prediction (column target == 'prediction') set.
        _standardize_data (bool): Passed bool that defines if standardization (x - mu / sigma) should be applied.
        _load_model (bool): Passed bool that defines if a pre-existing model with the same name should be loaded.
        _logger (Logger): Passed logger to track execution steps or errors. 
        _random_state (int): Passed random state as int for reproducible results.
        _model_subdirectory (str): Subdirectory name consisting of the class-name.
        _save_file_format (str): File format used to store the model.
        _model_path (Path): The path to the stored model.
        _evaluation_dir_path (Path): The path to the directory where the model evaluations are stored.
        _evaluation_file_path (Path): The path to the model evaluation file.
        _optimal_threshold (float): The threshold resulting in the best result according to GridSearchCV.
        _final_model (Estimator): A trained sklearn Estimator.
        _final_scaler (Transformer): Sklearn StandardScaler used for the standardization of the training data.
        _prediction_data (): Subset of the data that is used for prediction only (target == 'prediction').
    """

    def __init__(self, name: str, data: pd.DataFrame, apply_standardization: bool, load_existing_model: bool = False, logger: Logger = None, random_state: int = None):
        """Initializes PurchasePredictor.
        
        Args:
            name (str):
                Name of the purchase predictor.

            data (DataFrame):
                The data as pd.DataFrame containing training (column target == 'train') and prediction (column target == 'prediction') set.

            apply_standardization (bool):
                Defines if standardization with a sklearn StandardScaler should be applied to the data.

            load_existing_model (bool):
                Defines if a pre-existing model with the same name should be loaded (True = yes, False = no). Default is False.

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
        type_signature = [str, pd.DataFrame, bool, bool, (Logger, type(None)), (int, type(None))]
        model_creator_utils.check_type(type_signature, **fun_params)

        if not re.match(r'^\w+$', name):
            raise ValueError('Please specify a valid name (can contain letters, numbers and underscore)')

        # Set passed attributes
        self.name = name + ''
        self._data = data
        self._standardize_data = apply_standardization
        self._load_model = load_existing_model
        self._logger = logger
        self._random_state = random_state

        # Set internal attributes
        self._model_subdirectory = self.__class__.__name__
        self._save_file_format = '.pickle'
        self._model_path = PATHS['MODELS_DIR'] / self._model_subdirectory / str(self.name + self._save_file_format)
        self._evaluation_dir_path = PATHS['EVALUATIONS_DIR'] / self._model_subdirectory / str(self.name)
        self._evaluation_file_path = self._evaluation_dir_path / str(self.name+'_evaluation.txt')
        self._optimal_threshold = 0.5
        
        self._final_model = None
        self._final_scaler = None
        self._prediction_data = None

        # If a pre-existing model should be loaded
        if self._load_model:
            self._load_existing_purchase_predictor()
        else:
            # If no pre-existing model should be loaded but a model under the name exists a warning is printed
            if self._model_path.exists():
                warnings.warn("A Model with the name {0} already exists, model is overwritten if .initialize_forecaster() is called".format(self.name), stacklevel=2)
    


    ####################################################################################################
    # Public functions                                                                                 #
    ####################################################################################################

    def initialize_purchase_predictor(self, test_size: Union[int, float], settings: list = None, **kwargs):
        """Initializes PurchasePredictor instance

        This function includes the following steps:

            - Set the beta value for the generalized F-score (fbeta score)
            - Create a subfolder for the PurchasePredictor in the models and evaluation folder if it does not exist already
            - Create an evaluation file for the recommender
            - Prepare data for subsequent steps
                - Split in training and prediction dataset
                - Perform stratified train-test split
            - Perform grid search and find the best model with the best threshold for prediction
                - Evaluate the model on the hold-out test set
            - Set the ``_final_model`` attribute with a model trained on the whole training set
        
        Args:
            test_size (int or float):
                How many samples of the training data should be used as a hold-out test set. If float (should be between 0.0 and 1.0), the value represents the proportion of the dataset used as test samples. If int, the value represents the absolute number of test samples.

            settings (list):
                The settings for the pruchase predictor as a list with the form: [{'CLF_NAME': classifier_name_1, 'HYPERPARAMS': {'parameter_1': [value_1, value_2, ...], 'parameter_2': [...], ...}}, {'CLF_NAME': classifier_name_2, 'HYPERPARAMS': {...}}, {...}, ...].

            **kwargs:
                Additional parameters. Takes beta.

        Raises:
            ValueError: If settings are not specified.
            TypeError: If a beta is given and it is not of type int or float.
        """

        # check input types
        fun_params = locals()
        fun_params.pop('self', None)

        # model_creator_utils.check_type does not support checking kwargs
        fun_params.pop('kwargs', None)

        # also remove the settings parameter as checking the type of settings involves more logic
        fun_params.pop('settings', None)

        type_signature = [(int, float)]
        model_creator_utils.check_type(type_signature, **fun_params)

        # check if settings are valid
        self._check_settings_validity(settings)

        if self._load_model:
            print('Purchase predictor already initialized, you can make predictions with .predict_if_customer_purchases_next_quarter(), .get_all_customers_purchasing_next_quarter() or with .get_all_customers_not_purchasing_next_quarter()')

        else:
            if self._logger: self._logger.info(f'Initializing PurchasePredictor {self.name}')

            if 'beta' in kwargs:
                beta = kwargs.get("beta")
                if not isinstance(beta, (int, float)):
                    raise TypeError(f'Beta must be of type \'int\' or \'float\', but got {type(beta).__name__}.')
            else:
                beta = 2

            # Create the subfolder in models if it does not exist
            models_dir_path = PATHS['MODELS_DIR'] / self._model_subdirectory
            models_dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create the subfolders in reports if it does not exist
            self._evaluation_dir_path.mkdir(parents=True, exist_ok=True)

            with open(self._evaluation_file_path, 'w') as evaluation:
                 evaluation.write(model_creator_utils.get_title_line('General Model Info'))

            model_creator_utils.write_to_report_file('\tModel-Name: '+str(self.name), self._evaluation_file_path)

            X, y = self._get_training_data()
            self._prediction_data = self._get_prediction_data()

            # Generate hold-out test set
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, stratify=y, random_state=self._random_state)

            if settings is not None:
                search_space = self._transform_settings_to_search_space(settings)

                # Set the _optimal_threshold attribute with the value resulting from grid search
                best_unfitted_classifier, self._optimal_threshold, _ = self._perform_grid_search(search_space, X_train, y_train, X_test, y_test, beta)
            
                # Fit the final scaler to the whole data
                if self._standardize_data:
                    self._final_scaler = StandardScaler().fit(X)

                # Set the _final_model with the best classifier fitted on the whole data
                self._set_final_model(best_unfitted_classifier, X, y)
            
            else:
                if self._logger: self._logger.info(f'({self.name}) Settings are None: ValueError')
                raise ValueError('Settings are None, please specify parameters to train a model.')
                
            # set model as loaded
            self._load_model = True


    def predict_if_customer_purchases_next_quarter(self, customer_id: Union[float, list], ignore_if_not_exists=False) -> Union[int, pd.Series]:
        """ Returns the label (1 = purchases next quarter, 0 = does not purchase next quarter) for a given ``customer_id``.

        Args:
            customer_id (float or list):
                A customer's ID for which to get the label. If ``customer_id`` is given as list it should include multiple customer IDs for example [id_1, id_2, ...] of type float.
            
            ignore_if_not_exists (bool):
                 Determines what to do when a given ``customer_id`` does not exist. Defaul is False, which raises an error. True will return None. If ``customer_id``is a list, ``ignore_if_not_exists=True`` will return np.nan for the entry.
           
        Returns:
            If ``customer_id`` was is given as float, the predicted label (1 = purchases next quarter, 0 = does not purchase next quarter) to the given ``customer_id``. If the passed ``customer_id`` is a list the label for each customer ID in the list as pd.Series where the index represents the customers' ID and the values the corresponding labels.
        
        Raises:
            ValueError: If the customer under the given ID does not exist.
            TypeError: If the given ``customer_id`` is not of type float.
            NoTrainedModelError: If there is no trained model to make predictions with.
        """

        if isinstance(customer_id, float):

            
            if self._load_model:
                y_labels = self._predict_labels()

                if customer_id in y_labels.index:
                    return int(y_labels.loc[customer_id])
                
                elif ignore_if_not_exists:
                    return None
                
                else:
                    raise ValueError('There is no prediction for the given customer_id, as the customer seems to not exist.')
            else:
                raise NoTrainedModelError('There is no trained model to make predictions with, please call initialize_purchase_predictor() first or set load_existing_model to True.')
        
        elif isinstance(customer_id, list):
            # check if all elements in the list are float
            if all(isinstance(n, float) for n in customer_id):
                pass

            else:
                raise TypeError('One or more customer_id elements in the given list have the wrong type (must be of type float).')

            if self._load_model:
                y_labels = self._predict_labels()

                # for each customer id get the label and store it into a series
                customers = []
                for customer in customer_id:
                    if customer in y_labels.index:
                        customers.append(int(y_labels.loc[customer]))

                    elif ignore_if_not_exists:
                        customers.append(np.nan)
            
                    else:
                        raise ValueError('There are no recommendations for the given customer_id, as the customer seems to not exist.')
                
                customer_predictions = pd.Series(data=customers, index=customer_id)
                return customer_predictions
            else:
                raise NoTrainedModelError('There is no trained model to make predictions with, please call initialize_purchase_predictor() first or set load_existing_model to True.')

        else:
            raise TypeError(f'customer_id has the wrong type: {type(customer_id).__name__} given but expected float.')
        

    def get_all_customers_purchasing_next_quarter(self) -> list:
        """Returns all customers that are predicted to purchase in the next quarter.

        Returns:
            A list with the customer IDs of customers that were predicted to purchase in the next quarter.
        
        Raises:
            NoTrainedModelError: If there is no trained model to make predictions with.
        """

        #Check if predictions can be made
        if self._load_model:
            y_labels = self._predict_labels()
            return y_labels[y_labels == 1].index.tolist()
        else:
             raise NoTrainedModelError('There is no trained model to make predictions with, please call initialize_purchase_predictor() first or set load_existing_model to True.')


    def get_all_customers_not_purchasing_next_quarter(self) -> list:
        """Returns all customers that are predicted to not purchase in the next quarter.
        
        Returns:
           A list with the customer IDs of customers that were predicted to not purchase in the next quarter.
        
        Raises:
            NoTrainedModelError: If there is no trained model to make predictions with.
        """

        #Check if predictions can be made
        if self._load_model:
            y_labels = self._predict_labels()
            return y_labels[y_labels == 0].index.tolist()

        else:
             raise NoTrainedModelError('There is no trained model to make predictions with, please call initialize_purchase_predictor() first or set load_existing_model to True.')



    ####################################################################################################
    # Private functions                                                                                #
    #################################################################################################### 

    def _predict_labels(self) -> pd.Series:
        """Predicts the labels for the prediction data.

        Convenience function that predicts the labels for the prediction dataset with the ``_final_model`` using the ``_optimal_threshold``.

        Returns:
            The predicted labels as pd.Series, where the index consists of customer IDs and the values are the labels. 
        """

        # get the prediction dataset
        data = self._get_prediction_data()
        data_as_array = data.to_numpy()

        if self._standardize_data:
            data_as_array = self._final_scaler.transform(data_as_array)

        # predict with final model using the optimal threshold
        y_pred = self._final_model.predict_proba(data_as_array)[:,1]
        threshold_predictions = [1 if y > self._optimal_threshold else 0 for y in y_pred]

        # create series out of predictions
        y_labels = pd.Series(data = threshold_predictions, index = data.index)
        return y_labels


    def _load_existing_purchase_predictor(self):
        """Loads an existing purchase predictor.

        Raises: 
            ModelNotFoundError: If there is no pre-existing purchase predictor under the specified path ``_model_path``.
            IncompatibleDataError: If the loaded model was trained on standardized data but apply_standardization is set to False or the loaded model was trained on original data but apply_standardization is set to True.
        """
        if self._logger: self._logger.info(f'({self.name}) Trying to load a pre-existing model.')

        if self._model_path.exists():
            # Fit the StandardScaler for the whole data as initialize_purchase_predictor() was not called
            X, _ = self._get_training_data()
            self._final_scaler = StandardScaler().fit(X)

            self._final_model, trained_on_standardized_data, self._optimal_threshold = pickle.load(open(self._model_path, "rb" ))

            # Error is raised if apply_standardization was set to false on this instance but loaded model was trained on standardized data or apply_standardization was set to true on this instance but loaded model was trained on original data
            if not self._standardize_data and trained_on_standardized_data or self._standardize_data and not trained_on_standardized_data:
                if self._logger: self._logger.info(f'({self.name}) Model loading failed: IncompatibleDataError.')
                msg = 'apply_standardization was set to {0}, but the model specified to load was {1}trained on standardized data, please set apply_standardization to {2} to get interpretable results.'.format(self._standardize_data, 'not ' if self._standardize_data else '', not self._standardize_data)
                raise IncompatibleDataError(msg)

            if self._logger: self._logger.info(f'({self.name}) Model successfully loaded.')

        else:
            if self._logger: self._logger.info(f'({self.name}) Model loading failed: ModelNotFoundError.')
            self._load_model = False
            error_message = 'There is no pre-existing purchase predictor with the name '+self.name+', specify an existing purchase predictor or build one with .initialize_purchase_predictor().'
            raise ModelNotFoundError(error_message)


    def _threshold_optimized_fbeta_score(self, y_true: np.ndarray, y_pred: np.ndarray, beta: float, return_threshold: bool=False) -> float:
        """Calculates the threshold optimized generalized F-score (fbeta).

        The threshold optimized generalized F-score is basically a generalized F-score calculated for a range of thresholds (from 0 to 1 with 0.01 steps), where the maximum score for a specific threshold is then taken as final score.

        Args:
            y_true (array):
                The true labels as np.ndarray with shape (n_samples,).

            y_pred (array):
                The predicted labels as np.ndarray with shape (n_samples,).

            beta (float):
                The beta value for the generalized F-score.

            return_threshold (bool):
                Determines if the maximal score or the corresponding threshold at which the maximal score was achieved should be returned. Default is False (returns maximal score).
           
        Returns:
            If ``return_threshold = False`` the maximal generalized F-score (fbeta) is returned. If ``return_threshold = True`` the corresponding threshold at which the maximal generalized F-score was achieved is returned. 
        """

        thresholds = np.arange(0,1.01,0.01)
        threshold_fbeta_scores = []

        # calculate fbeta score for all thresholds
        for threshold in thresholds:
            threshold_predictions = [1 if y > threshold else 0 for y in y_pred]
            fb_score = fbeta_score(y_true, threshold_predictions, beta=beta)
            threshold_fbeta_scores.append((fb_score, threshold))

        # return the maximal fbeta score
        if not return_threshold:
            max_score = max(threshold_fbeta_scores, key=itemgetter(0))[0] 
            return max_score

        # return the threshold corresponding to the maximal fbeta score
        elif return_threshold:
            threshold_to_max_score = max(threshold_fbeta_scores, key=itemgetter(0))[1]
            return threshold_to_max_score


    def _compute_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, return_metric: bool) -> int:
        """Calculates a confusion matrix.

        This function is used as scorer for the grid search to calculate confusion matrix elements (true negatives, false positives, false negatives, true positives). 

        Args:
            y_true (array):
                The true labels as np.ndarray with shape (n_samples,).

            y_pred (array):
                The predicted labels as np.ndarray with shape (n_samples,).

            return_metric (str):
                The metric to return, either 'TN' for rue negatives, 'FP' for false positives, 'FN' for false negatives or 'TP' for true positives.
           
        Returns:
            The count of either true negatives (``return_metric='TN'``), false positives (``return_metric='FP'``), false negatives (``return_metric='FN'``) or true positives (``return_metric='TP'``) as integer.
        """

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        if return_metric == 'TN':
            return tn

        elif return_metric == 'FP':
            return fp

        elif return_metric == 'FN':
            return fn

        elif return_metric == 'TP':
            return tp

        
    # perform grid search on one estimator or a pipeline
    def _perform_grid_search(self, search_space: list, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, beta, return_fitted_model = False) -> tuple:
        """Performs a grid-search with cross validation using the given hyperparameters.

        Based on the given search space, a grid-search with CV (StratifiedKFold with 5 splits) is executed and the best resulting model in terms of a threshold optimized generalized F-score (fbeta) is selected. The threshold optimized generalized F-score is basically a generalized F-score calculated for a range of thresholds (from 0 to 1 with 0.01 steps), where the maximum score for a specific threshold is then taken as final score. This function also takes care of evaluating the best model and computing the confusion matrices, which are stored on disk as plots.

        Args:
            search_space (list):
                A list representing a valid search space with estimator and hyperparameter. For example: [{'clf': [Classifier_1()], 'clf__parameter_1': [value_1, value_2, ...], 'clf__parameter_2': [...], ...}, {'clf': [Classifier_2()], 'clf__parameter_1': [...], ...}, {...}, ...].

            X_train (array):
                The features of the training data as np.ndarray with shape (n_samples, n_features).

            y_train (array):
                The label of the test data as np.ndarray with shape (n_samples, 1).

            X_test (array):
                The features of the test data as np.ndarray with shape (n_samples, n_features).

            y_test (array):
                The label of the test data as np.ndarray with shape (n_samples, 1).

            beta (float):
                The beta value for the generalized F-score.

            return_fittd_model (bool):
                Determines if a fitted model should be returned. Default is False (returns unfitted Estimator object with parameter configuration).
           
        Returns:
           A tuple with three elements, where the first element is an unfitted or fitted estimator (depending on ``return_fittd_model``), the second element is the optimal prediction threshold and the third element are the best parameters (without prefix) as dict of the form {'hyperparameter_1': value, 'hyperparameter_2': value, ...}.
        """

        # lines to write into report file
        report_file_lines = []

        n_jobs = int(mp_cpu_count()/2)

        n_splits = 5
        cv = StratifiedKFold(n_splits=n_splits)

        # use the threshold optimized fbeta score for finding the best model via grid search
        threshold_optimized_fbeta_score = make_scorer(self._threshold_optimized_fbeta_score, greater_is_better=True, needs_proba=True, beta=beta)
        
        # calculate normal fbeta score (standard threshold of 0.5)
        normal_fbeta_score = make_scorer(fbeta_score, greater_is_better=True, beta=beta)
        
        # count-metrics for confusion matrix
        true_negatives_score = make_scorer(self._compute_confusion_matrix, return_metric='TN')
        false_positives_score = make_scorer(self._compute_confusion_matrix, return_metric='FP')
        false_negatives_score = make_scorer(self._compute_confusion_matrix, return_metric='FN')
        true_positive_score = make_scorer(self._compute_confusion_matrix, return_metric='TP')
        
        # to get the best threshold per split also calculate the optimal threshold as an additional score
        optimal_threshold =  make_scorer(self._threshold_optimized_fbeta_score, needs_proba=True, beta=beta, return_threshold=True)

        scoring = {
                    'threshold_optimized_fbeta_score': threshold_optimized_fbeta_score, 
                    'fbeta_score': normal_fbeta_score,
                    'optimal_threshold': optimal_threshold, 
                    'balanced_accuracy': 'balanced_accuracy',
                    'true_negatives': true_negatives_score,
                    'false_positives': false_positives_score,
                    'false_negatives': false_negatives_score,
                    'true_positives' : true_positive_score
                  }

        refit_score = 'threshold_optimized_fbeta_score'

        # Setup Pipeline
        if self._standardize_data:
            pipeline = Pipeline([
                ('standardize', StandardScaler()),
                ('clf', DummyClassifier())
            ])
        else:
            pipeline = Pipeline([
                ('clf', DummyClassifier())
            ])

        # Set up GridSearchCV
        grid_search = GridSearchCV(
                            estimator=pipeline, 
                            param_grid=search_space, 
                            scoring=scoring, 
                            n_jobs=n_jobs, 
                            refit=refit_score, 
                            cv=cv, 
                            verbose=0)

        if self._logger: 
            self._logger.info(f'({self.name}) Starting Hyperparameter tuning with grid search using {n_jobs} Jobs.') 
            n_combinations = len(ParameterGrid(search_space)) 
            self._logger.info(f'({self.name}) Fitting {n_splits} folds for each of {n_combinations} candidates, totalling {n_splits*n_combinations} fits.')
        
        result = grid_search.fit(X_train, y_train)

        if self._logger: 
            self._logger.info(f'({self.name}) Hyperparameter tuning finished.')
            self._logger.info(f'({self.name}) Evaluating best model.')


        # Get the best refit classifier
        best_fitted_classifier = result.best_estimator_.steps[-1][1]

        # Get the best mean test fbeta score 
        cv_mean_test_score = result.best_score_

        # Get index of best classifier
        best_index = result.best_index_

        # Get the mean optimal threshold for best classifier
        optimal_threshold = result.cv_results_['mean_test_optimal_threshold'][best_index]

        # Get the mean test balanced accuracy for best classifier
        cv_mean_test_balanced_accuracy = result.cv_results_['mean_test_balanced_accuracy'][best_index]

        cv_mean_test_fbeta_score = result.cv_results_['mean_test_fbeta_score'][best_index]

        cv_test_confusion_matrix = self._obtain_summed_confusion_matrix(n_splits, result.cv_results_, best_index)

        # Get the fitted scaler if the pipeline with standardization was used
        fitted_scaler = None
        if self._standardize_data:
            fitted_scaler = result.best_estimator_.steps[0][1]
        
    	# Evaluate the classifier's training performance:
        train_fb_score, train_ba_score = self._evaluate_classifer(best_fitted_classifier, X_train, y_train, fitted_scaler, optimal_threshold, beta)
        
        # Evaluate the classifier on the hold-out test set
        test_fb_score, test_ba_score, test_conf_mat = self._evaluate_classifer(best_fitted_classifier, X_test, y_test, fitted_scaler, optimal_threshold, beta, calculate_confusion_matrix=True)
        
        # Plot confusion matrices
        save_to = self._evaluation_dir_path / str(self.name+'_summed_cv_confusion_matrix.png')
        self._plot_confusion_matrix(cv_test_confusion_matrix, str(self.name)+' Summed CV Confusion Matrix', save_to)

        save_to = self._evaluation_dir_path / str(self.name+'_test_confusion_matrix.png')
        self._plot_confusion_matrix(test_conf_mat, str(self.name)+' Hold-Out Set Confusion Matrix', save_to)

        if self._logger: self._logger.info(f'({self.name}) Evaluation finished, evaluation file and plots are now accessible.')

        # Write lines to report file
        report_file_lines.append(f'\tModel-Type: {best_fitted_classifier.__class__.__name__}')
        report_file_lines.append(f'\tModel-Hyperparameters:\n{self._format_parameter_output(best_fitted_classifier.get_params())}')
        report_file_lines.append(model_creator_utils.get_title_line(f'Best Mean CV Test-Scores ({n_splits} Splits)'))
        report_file_lines.append(f'\tF{beta}-Score at different thresholds across splits (mean={optimal_threshold:.2f}): {cv_mean_test_score:.4f}')
        report_file_lines.append(f'\tF{beta}-Score at threshold = 0.5: {cv_mean_test_fbeta_score:.4f}')
        report_file_lines.append(f'\tBalanced-Accuracy-Score: {cv_mean_test_balanced_accuracy:.4f}\n')
        report_file_lines.append(model_creator_utils.get_title_line(f'Training-Scores (Refit Model) (Threshold = {optimal_threshold:.2f})'))
        report_file_lines.append(f'\tF{beta}-Score: {train_fb_score:.4f}')
        report_file_lines.append(f'\tBalanced-Accuracy-Score: {train_ba_score:.4f}\n')
        report_file_lines.append(model_creator_utils.get_title_line(f'Hold-Out-Set Test-Scores (Refit Model) (Threshold = {optimal_threshold:.2f})'))
        report_file_lines.append(f'\tF{beta}-Score: {test_fb_score:.4f}')
        report_file_lines.append(f'\tBalanced-Accuracy-Score: {test_ba_score:.4f}\n')
        model_creator_utils.write_to_report_file(report_file_lines, self._evaluation_file_path)
        
        #Get the parameters of the best classifier
        best_params_result = result.best_params_

        # Get the best unfit clasifier and the best of the passed parameters to return
        best_unfitted_classifier = best_params_result.pop('clf')
        prefix = 'clf__'
        best_params = {}
        for key in best_params_result.keys():
            new_key = key[len(prefix):]
            best_params[new_key] = result.best_params_[key]

        if return_fitted_model:
            return best_fitted_classifier, optimal_threshold, best_params

        else:
            return best_unfitted_classifier, optimal_threshold, best_params
    

    def _obtain_summed_confusion_matrix(self, n_splits: int, cv_results: dict, best_index: int) -> np.ndarray:
        """ Calculates a summed confusion matrix out of GridSearchCV results.

        The true negatives, false positives, false negatives and true positives are calculated when doin grid search with cross validation. This function sums these metrics (of the best estimator) over all splits.
        
        Args:
            n_splits (int):
                Number of splits performed in CV.

            cv_results (dict):
                Cross validation results from sklearn's grid search ``cv_results_``.

            best_index (int):
                Index of the best estimator in the grid search cv results. 
           
        
        Returns:
            The summed confusion matrix as np.ndarray with shape (n_classes, n_classes).
           
        """

        metrics = ['true_negatives', 'false_positives', 'false_negatives', 'true_positives']
        summed_metrics = []

        # over all splits
        for n in range(n_splits):
            for i, metric in enumerate(metrics):
                # get the metric for the best model (best_index) from the cv_results
                metric_value = cv_results['split{0}_test_{1}'.format(n, metric)][best_index]
                if n == 0:
                    summed_metrics.append(metric_value)
                else:
                    summed_metrics[i] = summed_metrics[i] + metric_value

        return np.array(summed_metrics).reshape(2,2)


    def _evaluate_classifer(self, classifier: object,  X_test: np.ndarray, y_test: np.ndarray, scaler: StandardScaler, optimal_threshold: float, beta: float, calculate_confusion_matrix:bool = False) -> tuple:
        """ Evaluates a given estimator.

        The estimator is evaluated using the generalized F-Score with the given beta value, the balanced accuracy and a confusion matrix (optional).

        Args:
            classifier (Estimator):
                An sklearn estimator object.

            X_test (array):
                Features of the test data as np.ndarray with the shape (n_samples, n_features).

            y_test (array):
                Labels of the test data as np.ndarray with the shape (n_samples, ).

            scaler (Transformer):
                A sklearn StandardScaler object.

            optimal_threshold (float):
                The threshold from which to classify the label as positive (prediction > optimal_threshold = 1 and prediction <= optimal_threshold = 0) as float.
        
            beta (float):
                The beta value for the generalized F-score.

            calculate_confusion_matrix (bool):
                Determines if the confusion matrix should be calculated and returned as additional "score". Default is False.
        
        Returns:
            If ``calculate_confusion_matrix=False`` a tuple with two elements, including the fbeta score (first element) and the balanced accuracy score (second element). If ``calculate_confusion_matrix=True`` a tuple with three elements, including the fbeta score (first element), the balanced accuracy score (second element) and the confusion matrix as np.ndarray with shape (n_classes, n_classes).
            
        """

        # If the data was scaled in the pipeline the scaler will be not none othersie (none) don't scale the data
        if scaler is not None:
            X_test = scaler.transform(X_test)

        # get probabilities for positive class
        y_pred = classifier.predict_proba(X_test)[:,1]

        # predict based on optimal_threshold
        threshold_predictions = [1 if y > optimal_threshold else 0 for y in y_pred]

        # calculate scores
        fb_score = fbeta_score(y_test, threshold_predictions, beta=beta)
        balanced_accurcacy = balanced_accuracy_score(y_test, threshold_predictions)

        if calculate_confusion_matrix:
            conf_mat = confusion_matrix(y_test, threshold_predictions)
            return fb_score, balanced_accurcacy, conf_mat

        return fb_score, balanced_accurcacy


    def _get_training_data(self) -> tuple:
        """Splits the dataset by the column target == 'train' to obtain training data

        As the dataset includes training data (features calculated on every quarter except the latest one) and prediction data (features calculated on every quarter of the whole dataset), it needs to be seperated. 

        Returns:
            The training data as tuple containing the features (first element) as np.ndarray of shape (n_samples, n_features) and the corresponding labels (second element) as np.ndarray of shape (n_samples,).
        """

        training_data = self._data.loc[self._data.target == 'train'].drop('target', axis=1)
        y = training_data.y_label.to_numpy()
        X = training_data.drop('y_label', axis=1).to_numpy()

        return X, y


    def _get_prediction_data(self) -> pd.DataFrame:
        """Splits the dataset by the column target == 'predict' to obtain prediction data.

        As the dataset includes training data (features calculated on every quarter except the latest one) and prediction data (features calculated on every quarter of the whole dataset), it needs to be seperated. 

        Returns:
            The data to do the predictions on (i.e. predict if a customer purchases next quarter) as pd.DataFrame of the shape (n_customers, n_features), where the index are customer IDs and the columns represent the features.
        """

        prediction_data = self._data.loc[self._data.target == 'predict'].drop('target', axis=1)
        prediction_data = prediction_data.drop('y_label', axis=1)
        return prediction_data


    def _set_final_model(self, classifier, X, y):
        """ Set the final pruchase predictor model.

        Fits an sklearn estimator (with the best parameter combination resulting from grid search) on the given data and stores the model on disk. The estimator can then be used for predictions.

        Args:
            classifier (Estimator):
                An sklearn estimator object.
        
            X (array):
                Features on which to fit the classifer on as np.ndarray with the shape (n_samples, n_features).

            y (array):
                Target labels as np.ndarray with the shape (n_samples,).           
        """

        # standardize data if specified
        if self._standardize_data:
            X = self._final_scaler.transform(X)

        # set the final model to the fit classifer
        self._final_model = classifier.fit(X, y)

        if self._logger: self._logger.info(f'({self.name}) Storing final model to disk.')

        # Store the model on disk as .pickle
        pickle.dump((self._final_model, self._standardize_data, self._optimal_threshold), open(self._model_path, "wb"))


    def _transform_settings_to_search_space(self, settings: list) -> list:
        """ Transforms the passed purchase predictor settings to a valid gird-search space.
        
        Args:
            settings (list):
                The settings for the pruchase predictor as a list with the form: [{'CLF_NAME': classifier_name_1, 'HYPERPARAMS': {'parameter_1': [value_1, value_2, ...], 'parameter_2': [...], ...}}, {'CLF_NAME': classifier_name_2, 'HYPERPARAMS': {...}}, {...}, ...].
        
        Returns:
            A list representing a valid search space. For example: [{'clf': [Classifier_1()], 'clf__parameter_1': [value_1, value_2, ...], 'clf__parameter_2': [...], ...}, {'clf': [Classifier_2()], 'clf__parameter_1': [...], ...}, {...}, ...].
        """

        # search space to return
        search_space = []

        # prefix for the pipeline to identify if the entry is an classifier or a hyperparameter
        prefix = 'clf__'
        for classifier in settings:
            clf_name = classifier['CLF_NAME']
             # get the classifier object to the specified name
            clf = self._get_classifier_to_name(clf_name)

            # if hyperparameters are specified
            if 'HYPERPARAMS' in classifier.keys():
                clf_hyperparamter = classifier['HYPERPARAMS']

                # if hyperparameters are given as list instead of a dict loop through the list and create a search space entry for each element and append it to the list
                if isinstance(clf_hyperparamter, list):
                    for parameters in clf_hyperparamter:
                        search_space_entry = self._get_search_space_entry(parameters, clf, prefix)
                        search_space.append(search_space_entry)
                
                # otherwise just add the entry to the search space
                else:
                    search_space_entry = self._get_search_space_entry(clf_hyperparamter, clf, prefix)
                    search_space.append(search_space_entry)
            
            # otherwise just append a classifier with its default setting predefined by sklearn
            else:
                search_space.append({prefix[:-2]: [clf]})

        return search_space


    def _get_search_space_entry(self, parameter: dict, clf: object, prefix: str) -> dict:
        """ Get an list element for the grid search space.

        Convenience function for ``_transform_settings_to_search_space()`` that encapsulates the logic of creating a single search space entry.

        Args:
            parameter (dict):
                A dict defining the hyperparameters of the form {'hyperparamer1': value, 'hyperparameter2': value, ...}.

            clf (Estimator):
                An sklearn estimator object.
        	
            prefix (str):
                Prefix to use for the pipeline to identify classifier and hyperparameters.
           
        Returns:
           A dict representing the search space entry of the form {'clf': [Estimator()], 'hyperparameter_1': [value_1, value_2], 'hyperparameter_2': [...], ... }.
        """

        # Set classifier in search space entry
        search_space_entry = {prefix[:-2]: [clf]}

        # for each hyperparamter add it to the search space entry under the prefixed key.
        for parameter_name in parameter.keys():
            search_space_entry[prefix + parameter_name] = parameter[parameter_name]

        # add a ranodm state if it is specified (for reproducable results)
        random_state_key = prefix + 'random_state'
        if random_state_key not in search_space_entry:
            search_space_entry[random_state_key] = [self._random_state]

        return search_space_entry

            
    def _get_classifier_to_name(self, classifier_name: str) -> object:
        """ Get an sklearn classifier to the given name.

        Args:
            classifier_name (str):
                The name of the classifier as defined in sklearn.

        Returns:
            An sklearn classifier.

        Raises: 
            NameError: If there is no existing classifier to the given name. 
        """

        classifiers = all_estimators('classifier')
        clf_class = [clf[1] for clf in classifiers if clf[0] == classifier_name]

        if clf_class:
             return clf_class[0]()
            
        else:
            msg = f'The passed classifier name \'{classifier_name}\' has no corresponding classifier, please make sure that the passed name corresponds to an actual sklearn classifier.'
            raise NameError(msg)
    

    def _check_settings_validity(self, settings: list):
        """ Checks if the given settings are valid.

        Args:
            settings (list):
                The settings for the pruchase predictor as a list with the form: [{'CLF_NAME': classifier_name_1, 'HYPERPARAMS': {'parameter_1': [value_1, value_2, ...], 'parameter_2': [...], ...}}, {'CLF_NAME': classifier_name_2, 'HYPERPARAMS': {...}}, {...}, ...].

        Raises:
            TypeError: If the setting or the setting elements have the wrong type.
            ValueError: If settings or hyperparameters are an empty list or if a given valid classifier has no ``predict_proba()`` function.
            KeyError: If needed identifiers like CLF_NAME are not present 
            NameError: If invalid parameter names (not part of a sklearn classifier) are present.
        """

        if isinstance(settings, list):
            # if list is empty
            if not settings:
                raise ValueError('The given settings are an empty list, please make sure to add a dictionary with a key \'CLF_NAME\' and a corresponding classfier name as value. You can specify hyperparameters for the classifier with the key \'HYPERPARAMS\'.')
            
            # if not all entries in the list are of type dict raise an error
            if not all(isinstance(s, dict) for s in settings):
                raise TypeError(f'Elements in settings are expected to be of type \'dict\'.')

            for setting in settings:
                # if there is no CLF_NAME key in the dict of the setting entry raise an error
                if 'CLF_NAME' not in setting.keys():
                    raise KeyError(f'Every entry in settings is required to have a \'CLF_NAME\' key, please make sure that this key exists in every entry in settings.')
                
                # get the classifier and its corresponding parameters
                classifier = self._get_classifier_to_name(setting['CLF_NAME'])

                # check if the classifier also has a predict_proba() function
                if not(hasattr(classifier,'predict_proba') and callable(getattr(classifier,'predict_proba'))):
                    raise ValueError('')
                
                clf_params_keys = classifier.get_params().keys()

                # check if hyperparameters are given as list or as dict
                if 'HYPERPARAMS' in setting.keys():
                    hyperparams = setting['HYPERPARAMS']

                    # if given as list, all elements in the list must be of type dict
                    if isinstance(hyperparams, list):
                        # if hyperparameter list is empty
                        if not hyperparams:
                            raise ValueError('The given hyperparameters are an empty list, please make sure to add hyperparameters as \'dict\' where a key represents the parameter name and the value is the parameter value/values wrapped in a list.')

                        if not all(isinstance(s, dict) for s in settings):
                            raise TypeError(f'Elements in the settings hyperparameters are expected to be of type \'dict\'.')
                        
                        # loop through the dicts in HYPERPARAMS
                        for hyperparams_entry in hyperparams:
                            # for each dict check if the keys are valid paramters of the corresponding classifier
                            for hyperparams_entry_key in hyperparams_entry.keys():
                                # check if the value to the key is a list otherwise raise an error:
                                hyperparams_entry_value = hyperparams_entry[hyperparams_entry_key]
                                
                                if not isinstance(hyperparams_entry_value, list):
                                    raise TypeError(f'The hyperparameter {hyperparams_entry_key} in the {classifier.__class__.__name__} settings must be of type \'list\', got type \'{type(hyperparams_entry_value).__name__}\', make sure that every specified hyperparameter is wrapped in a list.')

                                # if the parameter value list is empty
                                if not hyperparams_entry_value:
                                    raise ValueError(f'Valuelist for hyperparameter {hyperparams_entry_key} is empty. Please specify values for the hyperparameter {hyperparams_entry_key} or remove it from HYPERPARAMS.')

                                # if the key is not in the parameters specified by sklearn raise an error
                                if not hyperparams_entry_key in clf_params_keys:
                                    raise NameError(f'The specified hyperparameter {hyperparams_entry_key} is not a supported paramter of {classifier.__class__.__name__}, make sure to only use supported parameters (see the sklearn documentation of {classifier.__class__.__name__} for a list of valid parameters).')
                    
                    # if given as dict just check if the keys are valid paramters of the corresponding classifier
                    elif isinstance(hyperparams, dict):
                        for hyperparam_key in hyperparams.keys():
                            # check if the value to the key is a list otherwise raise an error:
                            hyperparams_value = hyperparams[hyperparam_key]

                            if not isinstance(hyperparams_value, list):
                                raise TypeError(f'The hyperparameter {hyperparam_key} in the {classifier.__class__.__name__} settings must be of type \'list\', got type \'{type(hyperparams_value).__name__}\', make sure that every specified hyperparameter is wrapped in a list.')
                            
                            # if the key is not in the parameters specified by sklearn raise an error
                            if not hyperparam_key in clf_params_keys:
                                    raise NameError(f'The specified hyperparameter {hyperparam_key} is not a supported paramter of {classifier.__class__.__name__}, make sure to only use supported parameters (see the sklearn documentation of {classifier.__class__.__name__} for a list of valid parameters).')

                    else:
                        raise TypeError(f'Hyperparameters in settings must be either of type \'dict\' or \'list\', got type \'{type(hyperparams).__name__}\'')

        else:
            raise TypeError(f'Settings must be of type \'list\', passed settings are of type \'{type(settings).__name__}\'')


    def _format_parameter_output(self, parameters: dict) -> str:
        """ Formats parameters for printing.

        Helper function to format model parameters for writing them into the evaluation file.

        Args:
            parameters (dict):
                A dict defining the hyperparameters of the form {'hyperparamer1': value, 'hyperparameter2': value, ...}.
        
        Returns:
            A formatted string
        """
        
        output = ''
        for key, value in parameters.items():
            output = output + '\t\t' + str(key) + ': ' + str(value) + '\n'
        
        return output

    def _plot_confusion_matrix(self, confusion_matrix: np.ndarray, title: str, save_path: Path):
        """ Plots a confusion matrix and stores the plot on disk.

        Args:
            confusion_matrix (array):
                The confusion matrix as np.ndarray with the shape (n_classes, n_classes).

            title (str):
                Title of the plot.

            save_path (Path):
                The path to which to store the plot to as Path object.
        """

        # Prepare labels
        labels = ['No Purchase', 'Purchase']
        metrics = ['TN', 'FP', 'FN', 'TP']
        values = [value for value in confusion_matrix.flatten()]
        values_as_percentage = [f'{value:.2%}' for value in values / np.sum(values)]

        # Calculate true negative and true positive rate
        tnr = values[0] / (values[0] + values[1])
        tpr = values[3]  / (values[3] + values[2])

        metric_rates = [tnr, 1-tnr, 1-tpr, tpr]

        annotation = [f'{value:.0f} ({metric})\n{percentage}\n{metric}R = {rate:.2f}' for value, metric, percentage, rate in zip(values, metrics, values_as_percentage, metric_rates)]

        annotation = np.array(annotation).reshape(2, 2)

        # Plot confusion matrix
        plt.figure(figsize=(8, 8))
        ax = sns.heatmap(confusion_matrix, annot=annotation, fmt='s', cmap='Blues', cbar=False, annot_kws={'fontsize': 15.5, 'linespacing': 1.75})
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.xaxis.tick_top()
        ax.set_xticklabels(labels, minor=False, fontsize=15)
        ax.set_yticklabels(labels, minor=False, va='center', fontsize=15)
        
        divider = make_axes_locatable(ax)
        
        #Add custom x-axis
        custom_x_axis = divider.append_axes('top', size='8%', pad=0.47)
        custom_x_axis.get_xaxis().set_visible(False)
        custom_x_axis.get_yaxis().set_visible(False)

        custom_x_axis_text = AnchoredText('Predicted', loc='center', frameon=False, prop=dict(backgroundcolor='white', size=15.5, color='black'))
        custom_x_axis.add_artist(custom_x_axis_text)
        
        #Add custom y-axis
        custom_y_axis = divider.append_axes('left', size='8%', pad=0.47)
        custom_y_axis.get_xaxis().set_visible(False)
        custom_y_axis.get_yaxis().set_visible(False)
        
        custom_y_axis_text = AnchoredText('Actual', loc='center', frameon=False, prop=dict(backgroundcolor='white', size=15.5, color='black', rotation=90))
        custom_y_axis.add_artist(custom_y_axis_text)
        
        ax.set_title(title, pad=80, fontsize=20)
        
        if not save_path.suffix :
            save_path = str(save_path)+'.png'
        
        plt.savefig(save_path, bbox_inches='tight')

       