import re
import warnings
import pickle
from typing import Union
from interface import implements
from joblib import Parallel, delayed
from multiprocessing import cpu_count as mp_cpu_count
from logging import Logger

import numpy as np
import pandas as pd
from datetime import timedelta

from pmdarima import ARIMA
from pmdarima.preprocessing import BoxCoxEndogTransformer
from pmdarima.pipeline import Pipeline

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from path_definitions import PATHS
from modelcreator.salesforecaster import IForecaster, forecaster_utils
from modelcreator.exception import ModelNotFoundError,  IncompatibleDataError, NoTrainedModelError
from modelcreator import model_creator_utils

class ARIMAForecaster(implements(IForecaster)):
    """Wrapper for an ARIMA forecaster.

    The ARIMAForecaster class provides the following functionality:

        - Get the best model in terms of AIC for given hyperparameters through grid search with cross-validation
        - RSME across CV-splits is also calculated
        - Loading of pre-existing models via the name attribute
        - Making forecasts for n periods into the future

    Attributes:
        name (str): Passed name used to identify the forecaster.
        _time_series (DataFrame): Passed time series on which the forecaster is trained and tested on.
        _transform_data (bool): Passed bool that defines if a Box-Cox transformation should be applied.
        _load_model (bool): Passed bool that defines if a pre-existing model with the same name should be loaded.
        _logger (Logger): Passed logger to track execution steps or errors. 
        _model_subdirectory (str): Subdirectory name consisting of the class-name.
        _save_file_format (str): File format used to store the model.
        _model_path (Path): The path to the stored model.
        _evaluation_dir_path (Path): The path to the directory where the model evaluations are stored.
        _evaluation_file_path (Path): The path to the model evaluation file.
        _final_model (ARIMA): A trained pmdarima ARIMA model.
        _final_transformer (Transformer): Data transformer used for transforming the data.
        _box_cox_lambda2 (float): Value to add to the time series values to make it non-negative/non-zero.
    """

    def __init__(self, name: str, time_series: pd.DataFrame, apply_box_cox: bool = False, load_existing_model: bool = False, logger: Logger=None):
        """Initializes ARIMAForecaster.

        Args:
            name (str):
                Name of the ARIMA forecaster.

            time_series (DataFrame):
                Time series on which the forecaster is trained and tested on.

            apply_box_cox (bool):
                Defines if a Box-Cox transformation should be applied (True = yes, False = no). Default is False.

            load_existing_model (bool):
                Defines if a pre-existing model with the same name should be loaded (True = yes, False = no). Default is False.
            
            logger (Logger):
                A logger to track execution steps or errors. 
                
        Raises:
            ValueError: If the name does not follow the specifications (only letters, numbers and underscore allowed).
        """

        # check input types
        fun_params = locals()
        fun_params.pop('self', None)
        type_signature = [str, pd.DataFrame, bool, bool, (Logger, type(None))]
        model_creator_utils.check_type(type_signature, **fun_params)

        if not re.match(r'^\w+$', name):
            raise ValueError('Please specify a valid name (can contain letters, numbers and underscore)')

        # Set passed attributes
        self.name = name
        self._time_series = time_series
        self._transform_data = apply_box_cox
        self._load_model = load_existing_model
        self._logger = logger

        # Set internal attributes
        self._model_subdirectory = self.__class__.__name__
        self._save_file_format = '.pickle'
        self._model_path = PATHS['MODELS_DIR'] / self._model_subdirectory / str(self.name + self._save_file_format)
        self._evaluation_dir_path = PATHS['EVALUATIONS_DIR'] / self._model_subdirectory / str(self.name)
        self._evaluation_file_path = self._evaluation_dir_path / str(self.name + '_evaluation.txt')
        self._result_dir_path = PATHS['RESULTS_DIR'] / self._model_subdirectory / str(self.name)
        
        # Set model initially to None
        self._final_model = None
        self._final_transformer = None
        
        # Define transformer
        self._box_cox_lambda2 = 1e-6

        # If a pre-existing model should be loaded
        if self._load_model:
            self._load_existing_ARIMA_forecaster()
        else:
            # If no pre-existing model should be loaded but a model under the name exists a warning is printed
            if self._model_path.exists():
                warnings.warn("A Model with the name {0} already exists, model is overwritten if .initialize_forecaster() is called".format(self.name), stacklevel=2)



    ####################################################################################################
    # Public functions                                                                                 #
    ####################################################################################################
 
    def initialize_forecaster(self, test_size: Union[int, float], hyperparameters: dict = None, **kwargs):
        """Initializes ARIMAForecaster instance.

        This function includes the following steps:

            - Fit the transformer if _transform_data is True
            - Create a subfolder for the ARIMAForecaster in the models and evaluation folder if it does not exist already
            - Create an evaluation file for the the forecaster
            - Prepare data for subsequent steps (train-test split)
            - Perform grid search and find best model 
                - Evaluate the model on the hold-out test set
            - Set the ``_final_model`` attribute with a model trained on the whole time series
        
        Args:
            test_size (int or float):
                How many samples of the given time series should be used as a hold-out test set. If float (should be between 0.0 and 1.0), the value represents the proportion of the dataset used as test samples. If int the value represents the absolute number of test samples.

            hyperparameters (dict):
                Hyperparameters used for model building as dict of the form {'parameter_1': [value_1, value_2, ...], 'parameter_2': [...], ...}.

            **kwargs: 
                Additional parameters.
        Raises:
            ValueError: If hyperparmeters are not specified.
        """

        # check input types
        fun_params = locals()
        fun_params.pop('self', None)

        # model_creator_utils.check_type does not support checking kwargs
        fun_params.pop('kwargs', None)

        # also remove the hyperparameters parameter as checking the type of hyperparameters involves more logic
        fun_params.pop('hyperparameters', None)

        type_signature = [(int, float)]
        model_creator_utils.check_type(type_signature, **fun_params)

        # check if settings are valid
        self._check_hyperparameters_validity(hyperparameters)

        # When an existing model was loaded, initialize_forecaster will do nothing
        if self._load_model:
            print('Forecaster already initialized, you can make forecasts with .make_forecast()')
        else:
            if self._logger: self._logger.info(f'Initializing ARIMAForecaster {self.name}')
            # Create the subfolder in models if it does not exist
            dir_path = PATHS['MODELS_DIR'] / self._model_subdirectory
            dir_path.mkdir(parents=True, exist_ok=True)

            # Create the subfolders in reports if it does not exist
            self._evaluation_dir_path.mkdir(parents=True, exist_ok=True)
            self._result_dir_path.mkdir(parents=True, exist_ok=True)

            # Create an evaluation file for the forecaster
            with open(self._evaluation_file_path, 'w') as evaluation:
                evaluation.write(model_creator_utils.get_title_line('General Model Info'))

            model_creator_utils.write_to_report_file(f'\tModel-Name: {self.name}', self._evaluation_file_path)

            # Get train test split
            data = self._time_series.values
            train, test = forecaster_utils.time_series_train_test_split(data, test_size)

            # If hyperparameters are given a grid search is performed
            if hyperparameters is not None:
                n_jobs = int(mp_cpu_count()/2)
                if self._logger: self._logger.info(f'({self.name}) Starting Hyperparameter tuning with grid search using {n_jobs} Jobs.')
                _, best_parameters = self._perform_grid_search(hyperparameters, train, test, n_jobs)
                
                if self._transform_data:
                    # Before building and fitting the model to the whole data fit a BoxCoxEndogTransformer to the whole time series
                    self._final_transformer = BoxCoxEndogTransformer(lmbda2=self._box_cox_lambda2).fit(data)

                self._set_final_model(best_parameters, data)
                
            else:
                if self._logger: self._logger.info(f'({self.name}) Hyperparameters are None: ValueError')
                raise ValueError('Hyperparameters are None, please specify parameters to train a model.')

            # Set model as loaded
            self._load_model = True


    def make_forecast(self, n_periods: int) -> pd.Series:
        """Makes a forecast for ``n_periods``.

        Args:
            n_periods (int):
                Defines the number of future periods to forecast.

        Returns:
            A time series of type pd.Series, where the index consists of the future n timesteps and the values quantify the future observations (aggregated sales)  corresponding to the predicted timesteps.

        Raises: 
            NoTrainedModelError: If there is no trained model to make forecasts with.
            ValueError: If n_periods is greater than 52 (a year).
        """

        # check input types
        fun_params = locals()
        fun_params.pop('self', None)
        type_signature = [int]
        model_creator_utils.check_type(type_signature, **fun_params)

        forecaster_utils.check_n_periods(n_periods)

        if self._load_model:
            # Get the last week in the data
            last_week = self._time_series.tail(1).index[0]
            
            # Predict for n_periods
            predictions = self._final_model.predict(n_periods=n_periods)

            if self._transform_data:
                predictions, _ = self._final_transformer.inverse_transform(predictions)

            # Build series with the predictions and the corresponding future dates
            prediction_dates = [last_week + timedelta(days=(n*7)) for n in range(1,n_periods+1)]
            prediction_series = pd.Series(data=predictions, index=prediction_dates)

            save_path = self._result_dir_path / str(self.name+f'_forecast_{n_periods}_periods.png')
            title = self.name+f' Forecast for {n_periods} Periods'
            forecaster_utils.plot_forecast(self._time_series, prediction_series, title, save_path)

            return prediction_series
        
        else:
            raise NoTrainedModelError('There is no trained model to make forecasts with, please call initialize_forecaster() first or set load_existing_model to True.')

    

    ####################################################################################################
    # Private functions                                                                                #
    ####################################################################################################

    def _load_existing_ARIMA_forecaster(self):
        """Loads an existing ARIMA forecaster.

        Raises: 
            ModelNotFoundError: If there is no pre-existing ARIMA forecaster under the specified path ``_model_path``.
            IncompatibleDataError: If the loaded model was trained on transformed data but apply_box_cox is set to False or the loaded model was trained on original data but apply_box_cox is set to True.
        """
        if self._logger: self._logger.info(f'({self.name}) Trying to load a pre-existing model.')
        if self._model_path.exists():
            # Create subfolder in results if it not exists already
            self._result_dir_path.mkdir(parents=True, exist_ok=True)

            # If the model exists fit the transformer...
            if self._transform_data:
                self._final_transformer = BoxCoxEndogTransformer(lmbda2=self._box_cox_lambda2).fit(self._time_series.values)

            # ... and load the model
            self._final_model, trained_on_transformed_data = pickle.load(open(self._model_path, "rb" ))

            # Error is raised if apply_box_cox was set to false on this instance but loaded model was trained on transformed data or apply_box_cox was set to true on this instance but loaded model was trained on original data
            if not self._transform_data and trained_on_transformed_data or self._transform_data and not trained_on_transformed_data:
                if self._logger: self._logger.info(f'({self.name}) Model loading failed: IncompatibleDataError.')
                self._load_model = False
                msg = 'apply_box_cox was set to {0}, but the model specified to load was {1}trained on transformed data, please set apply_box_cox to {2} to get interpretable results.'.format(self._transform_data, 'not ' if self._transform_data else '', not self._transform_data)
                raise IncompatibleDataError(msg)

            if self._logger: self._logger.info(f'({self.name}) Model successfully loaded.')
        
        # Otherwise raise a ModelNotFoundError
        else:
            if self._logger: self._logger.info(f'({self.name}) Model loading failed: ModelNotFoundError.')
            self._load_model = False
            error_message = 'There is no pre-existing forecaster with the name '+self.name+', specify an existing forecaster or build one with .initialize_forecaster().'
            raise ModelNotFoundError(error_message)
        
    def _build_model(self, order: tuple, seasonal_order: tuple) -> ARIMA:
        """Builds an ARIMA model and returns it.

        Args:
            order (tuple):
                Defines the order of the ARIMA model and has the form (p, d, q).

            seasonal_order (tuple):
                Defines the seasonal order of the ARIMA model and has the form (P, D, Q, m).

        Returns:
            An ARIMA model from pmdarima built according to the given orders.
        """

        model = ARIMA(order=order, seasonal_order=seasonal_order, suppress_warnings =True)
        return model


    def _perform_grid_search(self, hyperparameters: dict, training_data: np.ndarray, test_data: np.ndarray, n_jobs: int) -> tuple:
        """Performs a grid-search with cross validation using the given hyperparameters.

        Based on the given parameter grid, a grid-search with CV (TimeSeriesSplit with 5 splits, predicting 8 time steps at each) is executed and the best resulting model in terms of AIC and its paramters are returned. This function also takes care of evaluating the best model and plotting the predictions as time series (plots are stored on disk).

        Args:
            hyperparameters (dict):
                Hyperparameters used for model building as dict of the form {'parameter_1': [value_1, value_2, ...], 'parameter_2': [...], ...}.

            training_data (array):
                Training data in the form of a numpy array with shape (n_samples, 1), representing the time series part.
            
            test_data (array):
                Test data in the form of a numpy array with shape (n_samples, 1), representing the time series part.

            n_jobs (int):
                Number of jobs to run in parallel. More specifically how many grid-search steps are executed in parallel. 

        Returns:
            A tuple containing the best model and its parameters of form (best_model, best_params), where ``best_model`` is is a pmdarima ARIMA model and ``best_params`` is a tuple containing the orders of the model of the form ((p, d, q), (P, D, Q, m)) is returned.
        """
        # lines to write into report file
        report_file_lines = []

        # Remove inappropriate parameter combinations
        parameter_combinations = self._remove_inappropriate_combinations(list(ParameterGrid(hyperparameters)))
       
        # number of combinations
        n_combinations = len(parameter_combinations)

        # numer of cv splits 
        n_splits = 6
        test_size = 8

        # Cross-validation generator for time series data
        cv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

        if self._logger: self._logger.info(f'({self.name}) Fitting {n_splits} folds for each of {n_combinations} candidates, totalling {n_splits*n_combinations} fits.')

        # Execute parallel grid-search steps 
        result = Parallel(n_jobs=n_jobs)(delayed(self._execute_grid_search_step)(parameter_combination, training_data, cv) for parameter_combination in parameter_combinations)

        if self._logger: self._logger.info(f'({self.name}) Hyperparameter tuning finished.')

        # Get the best result out of the grid-search results
        best_params, mean_cv_test_rmse, best_aic_score = self._get_best_result(result)

        if self._logger: self._logger.info(f'({self.name}) Evaluating best model.')
        
        # Build the model based on the best parameters
        best_model = self._build_model(*best_params)

        if self._transform_data:
            best_model = Pipeline([
                ("boxcox", BoxCoxEndogTransformer(lmbda2=self._box_cox_lambda2)),
                ("model", best_model)
            ])

        # Refit the model with best parameters
        best_model.fit(training_data)

        # Evaluate model on hold-out test set
        n_periods = test_data.shape[0]
        predictions = best_model.predict(n_periods)
        test_score = mean_squared_error(test_data ,predictions, squared=False)


        save_path = self._evaluation_dir_path / str(self.name+'_test_data_prediction.png')
        title = self.name+f' Hold-Out Set Prediction (RMSE = {test_score:.2f})'
        forecaster_utils.plot_predictions(time_series = self._time_series, predicted_data=predictions, title=title, save_path=save_path, mode='test')

        # Write lines to report file
        report_file_lines.append(f'\tModel-Type: {best_model.__class__.__name__}')
        report_file_lines.append(f'\tModel-Hyperparameters:\n\t\tOrder: {best_params[0]}')
        report_file_lines.append(f'\t\tSeasonal Order: {best_params[1]}\n')
        report_file_lines.append(model_creator_utils.get_title_line(f'Mean CV Test-Scores ({n_splits} Splits) ({test_size} Test Periods each)'))
        report_file_lines.append(f'\tAIC: {best_aic_score:.4f}')
        report_file_lines.append(f'\tRMSE: {mean_cv_test_rmse:.4f}\n')
        report_file_lines.append(model_creator_utils.get_title_line(f'Test-Score (Refit Model) ({n_periods} Test Periods each)'))
        report_file_lines.append(f'\tRMSE: {test_score:.4f}\n')
        model_creator_utils.write_to_report_file(report_file_lines, self._evaluation_file_path)

        # To return model as pmdarima ARIMA model
        if self._transform_data:
            best_model = best_model.named_steps['model']

        if self._logger: self._logger.info(f'({self.name}) Evaluation finished, evaluation file and plots are now accessible.')

        return best_model, best_params


    def _execute_grid_search_step(self, parameters: dict, training_data: np.ndarray, cv) -> tuple:
        """Single parameter combination grid-search step.

        Executes a single grid-search step based on a single parameter combination. This includes:

             - The transformation of the parameters into ARIMA orders
             - Splitting the data based on the given cross-validation generator
             - Building the model with the transformed parameter combination
             - Fitting and testing the model on each cv-split
             - Calulting the error metric for each cv-split
             - Returning the transformed paramters and the mean cv error

        The function is called for every parameter combination in the parameter grid by the ``_perform_grid_search()`` function.
        
        Args:
            parameters (dict):
                Parameter combination in the form of a dict. For example {'p': 1, 'd': 1, 'q': 1, 'P': 1, 'D': 0, 'Q': 2, m=52}.

            training_data (array): 
                Training data in the form of a numpy array with shape (n_samples,), representing the values of the time series.

            cv (cross-validation generator): 
                A cross-validation generator that features a ``get_n_splits()`` and ``split()`` method.

        Returns:
            A tuple in the form of (orders, rmse, aic), where orders is again a tuple containing two tuples representing order and seasonal order for an ARIMA model, while rmse is a float value quantifying the mean cross-validation rmse and aic is a float value quantifying the mean cross-validation AIC. An example would be: (((2, 0, 1), (0, 0, 0, 0)), 69041.5458, 705.1763).
        """

        rmse_results = []
        aic_results = []
        orders = self._parameters_to_orders(parameters)

        for train_index, test_index in cv.split(training_data):
            n_periods = len(test_index)
            
            # Training data for this fold
            train_fold = training_data[train_index]
            
            # Test data for this fold
            test_fold = training_data[test_index]

            # Build the model
            model = self._build_model(*orders)

            if self._transform_data:
                model = Pipeline([
                    ("boxcox", BoxCoxEndogTransformer(lmbda2=self._box_cox_lambda2)),
                    ("model", model)
                ])

            model.fit(train_fold)

            # predict n_periods future timesteps
            predictions = model.predict(n_periods)

            rmse = mean_squared_error(test_fold, predictions, squared=False)
            rmse_results.append(rmse)

            if self._transform_data:
                aic_results.append(model.named_steps['model'].aic())
            else:
                aic_results.append(model.aic())

        return orders, np.mean(rmse_results), np.mean(aic_results)


    def _get_best_result(self, grid_search_result: list) -> tuple:
        """Get best grid-search result
        
        Get the best grid-search result in form of (best order, best mean cv-error).

        Args:
            grid_search_result (list): 
                The result of the grid-search in form of a list containing tuples. For example a grid-search with two parameter-combinations and thus two results: [(((1, 0, 1), (0, 0, 0, 0)), 68348.42647101967, 698.3847), (((2, 0, 1), (0, 0, 0, 0)), 69041.5458, 705.1763)].

        Returns:
            A tuple including in the form (best orders, mean cv rmse, best mean AIC), where best orders is again a tuple including the order and seasonal order for an ARIMA model, while the mean cv rmse is a float value quantifying the mean cross-validation rmse corresponding to the best mean AIC, which is also a float value. An example would be: (((1, 0, 1), (0, 0, 0, 0)), 68348.4264, 705.1763).
        """
        # Zip the given list grid_search_result elementwise and apply the list function to the resulting tuples
        zipped_results = list(map(list, zip(*grid_search_result)))

        parameter_list = zipped_results[0]
        mean_rmse_list = zipped_results[1]
        mean_aic_list = zipped_results[2]
        
        # Get the index of the best result (lowest aic)
        min_idx = mean_aic_list.index(min(mean_aic_list))

        return parameter_list[min_idx], mean_rmse_list[min_idx], mean_aic_list[min_idx]


    def _set_final_model(self, parameters: tuple, data: np.ndarray) -> ARIMA:
        """Set a final forecasting model

        Builds a final forecasting model with the given parameters and trains it on the given time series. The model is then stored on disk as tuple where the first element is the model and the second element is a bool defining if the data the model was built on was transformed (true) or not (false). 

        Args:
            parameters (tuple):
                A tuple representing the order for an ARIMA model in the form of ((p, d, q), (P, D, Q, m)).

            data (array):
                A time series in the form of a np.ndarray with shape(n_samples, 1)
        """
        # Build the model with the given parameters and fit it on the given data
        model = self._build_model(*parameters)

        if self._transform_data:
            data, _ = self._final_transformer.transform(data)

        if self._logger: self._logger.info(f'({self.name}) Storing final model to disk.')

        self._final_model = model.fit(data)

        # Store the model on disk as .pickle
        pickle.dump((self._final_model, self._transform_data), open(self._model_path, "wb"))


    def _check_hyperparameters_validity(self, hyperparameters: dict):
        """Checks if the given hyperparameters are valid.

        Args:
            hyperparameters (dict):
                A dict of the form {'parameter_1': [value_1, value_2, ...], 'parameter_2': [...], ...}.

        Raises:
            TypeError: If the hyperparameters have the wrong type.
            ValueError: If hyperparameters are an empty dict or the a parameter has no attached value list.
            KeyError: If necassary keys are absent.
            NameError: If invalid parameter names are present
        """
        required_keys = ['p', 'd', 'q', 'P', 'D', 'Q', 'm']

        if isinstance(hyperparameters, dict):
            if not hyperparameters:
                raise ValueError(f'Hyperparameters is an empty dict. Please specify supported hyperparameters. Supported parameters are: {", ".join(val_key for val_key in required_keys)}')

            if not all(k in hyperparameters for k in required_keys):
                raise KeyError(f'The specified hyperparameters miss required keys. Make sure to include {", ".join(val_key for val_key in required_keys)}')

            for key in hyperparameters.keys():
                if key in required_keys:
                    parameter = hyperparameters[key]
                    if isinstance(parameter, list):
                        if not parameter:
                            raise ValueError(f'Valuelist for hyperparameter {key} is empty. Please specify values for the hyperparameter {key} or remove it from HYPERPARAMS.')

                        if not all(isinstance(p, int) for p in parameter):
                            raise TypeError(f'The hyperparameter {key} contains one or more wrong types, all entries in the value list are expected to be of type \'int\'.')
                            
                    else:
                        raise TypeError(f'The hyperparameter {key} must be of type \'list\', got type \'{type(parameter).__name__}\', make sure that every specified hyperparameter is wrapped in a list.')

                else:
                    raise NameError(f'The specified hyperparameter {key} is not a supported paramter of ARIMAForecaster, make sure to only use supported parameters. Supported parameters are: {", ".join(val_key for val_key in required_keys)}')
        else:
            raise TypeError(f'Hyperparameters must be of type \'dict\', passed hyperparameters are of type \'{type(hyperparameters).__name__}\'')


    def _parameters_to_orders(self, parameter_combination: dict) -> tuple:
        """Transform parameters to ARIMA orders.

        A single parameter-combination in form of a dict is transformed into ARIMA orders of form order (p,d,q) and seasonal order (P, D, Q, m).
        
        Args:
            parameter_combination (dict): 
                Single parameter combination in form of a dict. For example: {'p': 1, 'd': 1, 'q': 1, 'P': 1, 'D': 0, 'Q': 2, m=52}.

        Returns:
            A tuple representing the orders for an ARIMA model in the form of ((p,d,q), (P, D, Q, m)). For Example ((2,1,0) (1,0,2,52)).
        
        """
        order = (parameter_combination['p'], 
                 parameter_combination['d'], 
                 parameter_combination['q'])
        
        seasonal_order = (parameter_combination['P'], 
                          parameter_combination['D'], 
                          parameter_combination['Q'],  
                          parameter_combination['m'])

        return order, seasonal_order


    def _remove_inappropriate_combinations(self, paramter_combinations: list) -> dict:
        """Removes inappropriate parameter-grid combinations.

        Removes ARIMA settings in form of a parameter grid that would make no sense, for example the setting (p,d,q) (1,0,2,0), which would have no seasonality (m=0) but non-zero seasonal terms ((Q, P, D) != 0).

        Args:
            parameter_combinations (list):
                A list that containes paramter combinations in form of a dict, for example: [{'p': 1, 'd': 1, 'q': 1, 'P': 1, 'D': 0, 'Q': 2, m=52}, {...}, ...].
        
        Returns:
            A dict in the same format as the input dict that only contains appropriate settings.
        """

        appropriate_combinations = [combination 
                                    for combination in paramter_combinations 
                                    if (combination['m'] == 0 
                                    and combination['P'] == 0 
                                    and combination['D'] == 0 
                                    and combination['Q'] == 0) 
                                    or combination['m'] > 0]

        return appropriate_combinations