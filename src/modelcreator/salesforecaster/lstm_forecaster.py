import re
import warnings
from typing import Union
from interface import implements
from multiprocessing import cpu_count as mp_cpu_count
from logging import Logger

import numpy as np
import pandas as pd
from datetime import timedelta

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor

from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from path_definitions import PATHS
from modelcreator import model_creator_utils
from modelcreator.salesforecaster import IForecaster, forecaster_utils
from modelcreator.exception import ModelNotFoundError, NoTrainedModelError


class LSTMForecaster(implements(IForecaster)):
    """Wrapper for an LSTM forecaster.

    The LSTMForecaster class provides the following functionality:

        - Get the best model in terms of RMSE for given hyperparameters through grid search with cross-validation
        - Loading of pre-existing models with via the name attribute
        - Making forecasts for n periods into the future

    Attributes:
        name (str): Passed name used to identify the forecaster.
        _time_series (DataFrame): Passed time series on which the forecaster is trained and tested on.
        _load_model (bool): Passed bool that defines if a pre-existing model with the same name should be loaded.
        _logger (Logger): Passed logger to track execution steps or errors.
        _random_state (int): Passed random state as int for reproducible results.
        _model_subdirectory (str): Subdirectory name consisting of the class-name.
        _model_path (Path): The path to the stored model.
        _evaluation_dir_path (Path): The path to the directory where the model evaluations are stored.
        _evaluation_file_path (Path): The path to the model evaluation file.
        _scaler_range (tuple): Determines the range of the MinMaxScaler.
        _final_model (Sequential): A trained keras sequential model.
        _final_scaler (Transformer): MinMaxScaler used for scaling the data.
        _input_shape (tuple): Defines the input_shape of the form (n_timesteps, n_features) for the LSTM model.
    """
    
    def __init__(self, name: str, time_series: pd.DataFrame, load_existing_model: bool = False, logger: Logger=None, random_state: int=None):
        """Initializes LSTMForecaster.

        Args:
            name (str):
                Name of the LSTM forecaster.

            time_series (DataFrame):
                Time series on which the forecaster is trained and tested on.
            
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
        type_signature = [str, pd.DataFrame, bool, (Logger, type(None)), (int, type(None))]
        model_creator_utils.check_type(type_signature, **fun_params)

        if not re.match(r'^\w+$', name):
            raise ValueError('Please specify a valid name (can contain letters, numbers and underscore)')

        # Set passed attributes
        self.name = name
        self._time_series = time_series
        self._load_model = load_existing_model
        self._logger = logger
        self._random_state = random_state

        if self._random_state:
            tensorflow.random.set_seed(random_state)

        # Set internal attributes
        self._model_subdirectory = self.__class__.__name__
        self._model_path = PATHS['MODELS_DIR'] / self._model_subdirectory / self.name
        self._evaluation_dir_path = PATHS['EVALUATIONS_DIR'] / self._model_subdirectory / str(self.name)
        self._evaluation_file_path = self._evaluation_dir_path / str(self.name + '_evaluation.txt')
        self._result_dir_path = PATHS['RESULTS_DIR'] / self._model_subdirectory / str(self.name)
        
        # Define scaler range
        self._scaler_range = (-1, 1)

        # Set model and input_shape  initially to None
        self._final_model = None
        self._final_scaler = None
        self._input_shape = None
        
        # If a pre-existing model should be loaded
        if self._load_model:
            self._load_existing_LSTM_forecaster()
        else:
            # If no pre-existing model should be loaded but a model under the name exists a warning is printed
            if self._model_path.exists():
                warnings.warn("A Model with the name {0} already exists, model is overwritten if .initialize_forecaster() is called".format(self.name), stacklevel=2)



    ####################################################################################################
    # Public functions                                                                                 #
    ####################################################################################################
        
    def initialize_forecaster(self, test_size: Union[int, float], hyperparameters:dict = None, **kwargs):
        """Initializes LSTMForecaster instance.

        This function includes the following steps:
    
            - Set the input shape for the lstm model (n_timesteps, n_features)
            - Create a subfolder for the LSTMForecaster in the models and evaluation folder if it does not exist already
            - Create an evaluation file for the the forecaster
            - Fit the MinMaxScaler
            - Prepare data for subsequent steps (transform to sequence data, train-test split)
            - Perform grid search and find best model
                - Evaluate the model on the hold-out test set
            - Set the ``_final_model`` attribute with a model trained on the whole data

        Args:
            test_size (int or float):
                How many samples of the created sequences should be used as a hold-out test set. If float (should be between 0.0 and 1.0), the value represents the proportion of the dataset used as test samples. If int, the value represents the absolute number of test samples.

            hyperparameters (dict):
                Hyperparameters used for model building as dict of the form {'parameter_1': [value_1, value_2, ...], 'parameter_2': [...], ...}.

            **kwargs: 
                Additional parameters. Takes lstm_timesteps and lstm_features (not supported, default of 1 is used).
        Raises:
            ValueError: If hyperparmeters are not specified.
            TypeError: If lstm_timesteps or lstm_features are given and are not of type int.
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
            print('Forecaster already initialized, you can make forecasts with .make_forecast')
        else:
            if self._logger: self._logger.info(f'Initializing LSTMForecaster {self.name}')
            
            lstm_timesteps = 4
            lstm_features = 1

            if 'lstm_timesteps' in kwargs:
                lstm_timesteps = kwargs.get("lstm_timesteps")
                if not isinstance(lstm_timesteps, int):
                    raise TypeError(f'lstm_timesteps must be of type \'int\', but got {type(lstm_timesteps).__name__}.')

            ''' Multiple time series not supported
            if 'lstm_features' in kwargs:
                lstm_features = kwargs.get("lstm_features")
                if not isinstance(lstm_features, int):
                    raise TypeError(f'Beta must be of type \'int\', but got {type(lstm_features).__name__}.')
            '''
            self._input_shape = (lstm_timesteps, lstm_features)
            
            # Create the subfolder in models if it does not exist
            dir_path = PATHS['MODELS_DIR'] / self._model_subdirectory
            dir_path.mkdir(parents=True, exist_ok=True)

            # Create the subfolders in reports if it does not exist
            self._evaluation_dir_path.mkdir(parents=True, exist_ok=True)
            self._result_dir_path.mkdir(parents=True, exist_ok=True)

            # Create an evaluation file for the forecaster
            with open(self._evaluation_file_path, 'w') as evaluation:
                evaluation.write(model_creator_utils.get_title_line('General Model Info'))

            model_creator_utils.write_to_report_file('\tModel-Name: '+str(self.name),  self._evaluation_file_path)

            # Transform time series into sequences
            X, y = forecaster_utils.get_sequence_data(self._time_series, *self._input_shape)

            # Split into training and testing 
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

            if hyperparameters is not None:
                _, best_parameters = self._perform_grid_search(hyperparameters, X_train, y_train, X_test, y_test)

                # Before building and fitting the model to the whole data fit a MinMaxScaler to the whole time series
                self._final_scaler = MinMaxScaler(self._scaler_range).fit(self._time_series.values)

                # Building a final model based on the best_parameters and fit it to the whole data
                self._set_final_model(best_parameters, X, y)
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
            A time series of type pd.Series, where the index consists of the future n timesteps and the values quantify the future observations (aggregated sales) corresponding to the predicted timesteps.

        Raises: 
            NoTrainedModelError: If there is no trained model to make forecasts with.
        """

        # check input types
        fun_params = locals()
        fun_params.pop('self', None)
        type_signature = [int]
        model_creator_utils.check_type(type_signature, **fun_params)

        forecaster_utils.check_n_periods(n_periods)

        if self._load_model:
            n_timesteps = self._input_shape[0]

            # Get the last n_timesteps of the time series to build a sequence
            time_series_period = self._time_series.tail(n_timesteps)

            # Reshape the sequence for normalizing
            sequence = time_series_period.values.reshape(-1, 1)
            
            # Normalize with MinMaxScaler
            sequence = self._final_scaler.transform(sequence)

            # Reshape into lstm form (n_samples, n_timesteps, n_features)
            sequence = sequence.reshape(1, *self._input_shape)

            # Create empty numpy array for predictions
            predictions = np.empty((n_periods,1))
            
            # Create empty list for date index
            prediction_dates = []

            # Get the date of the last week in the time series
            last_week = time_series_period.tail(1).index[0]

            # Make n_periods one-step-ahead predictions
            for n in range(n_periods):
                prediction = self._final_model.predict(sequence)
                sequence = sequence[:,1:]
                sequence = np.concatenate((sequence, prediction.reshape(1, 1, self._input_shape[1])), axis=1)

                # Update last week to one week in the future
                last_week = last_week + timedelta(days=7)

                prediction_dates.append(last_week)
                predictions[n] = prediction
            
            # Inverse transform the normalized predictions
            inverse_predictions = self._final_scaler.inverse_transform(predictions).reshape(n_timesteps)
            prediction_series = pd.Series(data=inverse_predictions, index=prediction_dates)

            save_path = self._result_dir_path / str(self.name+f'_forecast_{n_periods}_periods.png')
            title = self.name+f' Forecast for {n_periods} Periods'
            forecaster_utils.plot_forecast(self._time_series, prediction_series, title, save_path)

            return prediction_series

        else:
            raise NoTrainedModelError('There is no trained model to make forecasts with, please call initialize_forecaster() first or set load_existing_model to True.')



    ####################################################################################################
    # Private functions                                                                                #
    ####################################################################################################

    def _load_existing_LSTM_forecaster(self):
        """Loads an existing LSTM forecaster.

        Raises: 
            ModelNotFoundError: If there is no pre-existing LSTM forecaster under the specified path ``_model_path``.
        """
        if self._logger: self._logger.info(f'({self.name}) Trying to load a pre-existing model.')

        if self._model_path.exists():

            # Create subfolder in results if it not exists already
            self._result_dir_path.mkdir(parents=True, exist_ok=True)

            # Fit the MinMaxScaler for the whole time series as initialize_forecaster() was not called
            self._final_scaler = MinMaxScaler(self._scaler_range).fit(self._time_series.values)

            self._final_model = tensorflow.keras.models.load_model(str(self._model_path))

            # Define input shape as initialize_forecaster is not called
            model_input_shape = self._final_model.layers[0].input_shape
            self._input_shape = (model_input_shape[1], model_input_shape[2])

            if self._logger: self._logger.info(f'({self.name}) Model successfully loaded.')
        else:
            if self._logger: self._logger.info(f'({self.name}) Model loading failed: ModelNotFoundError.')
            self._load_model = False
            error_message = 'There is no pre-existing forecaster with the name '+self.name+', specify an existing forecaster or build one with .initialize_forecaster().'
            raise ModelNotFoundError(error_message)


    # For bigger models parameters could be modfied to **kwargs
    def _build_model(self, neurons_layer_1: int = 1, neurons_layer_2: int = 0, recurrent_dropout: float = 0, learning_rate: float = 0.001) -> Sequential:
        """Builds an LSTM model and returns it.

        Args:
            neurons_layer_1 (int):
                Number of LSTM cells in the first layer, specified as integer.

            neurons_layer_2 (int):
                Number of LSTM cells in the second layer, specified as integer.

            recurrent_dropout (float):
                Percentage of dropout between recurrent units (applied on second layer), specified as float.

            learning_rate (float):
                Learning rate for the optimizer, specified as float.

        Returns: 
             A keras sequential LSTM model built according to the given hyperparameters.
        """

        model = Sequential(name=self.name)
        if neurons_layer_2 > 0:
            model.add(LSTM(neurons_layer_1, return_sequences=True, input_shape=self._input_shape))
            model.add(LSTM(neurons_layer_2, recurrent_dropout=recurrent_dropout))
        else:
            model.add(LSTM(neurons_layer_1, input_shape=self._input_shape))
        model.add(Dense(1))
        
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=[RootMeanSquaredError()])
        return model


    def _perform_grid_search(self, parameter_grid: dict, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> tuple:
        """Performs a grid-search with cross validation using the given hyperparameters.

        Based on the given parameter grid, a grid-search with CV (TimeSeriesSplit with 5 splits, predicting 8 time steps at each) is executed and the best resulting model in terms of RMSE and its paramters are returned. This function also takes care of evaluating the best model and plotting the predictions as time series (plots are stored on disk).

        Args:
            parameter_grid (dict):
                Hyperparameters used for model building as dict of the form {'hyperparameter_1': [value_1, value_2, ...], 'hyperparameter_2': [...], ...}.

            X_train (array):
                Training data as np.ndarray of shape (n_samples, n_timesteps).

            y_train (array):
                Target values of training data as np.ndarray of shape (n_samples, 1).

            X_test (array):
                Testing data for evaluating the performance of the model on a hold-out set as np.ndarray of shape (n_samples, n_timesteps)

            y_test (array):
                Target values of test data as np.ndarray of shape (n_samples, 1).

        Returns: 
            A tuple containing the best model and its parameters of form (best_model, best_params), where ``best_model`` is is a keras sequential model and ``params_best_model`` is a dict containing the hyperparamter configuration of the form {'hyperparameter_1': value, 'hyperparameter_2': value, ...}.
        """
        
        # lines to write into report file
        report_file_lines = []

        # Define CV strategy
        n_splits = 6
        test_size = 8
        time_series_split_cv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        print()

        # Scoring method for grid-search
        scoring = 'neg_root_mean_squared_error'

        # Jobs to run in parellel
        n_jobs = int(mp_cpu_count()/2)
        
        model = KerasRegressor(build_fn=self._build_model, verbose=0)
        # Wrap the Keras Regressor Wrapper in a TransformedTargetRegressor to enable scaling the target variable via the pipeline
        wrapped_model = TransformedTargetRegressor(regressor=model, transformer=MinMaxScaler(self._scaler_range))

        # Create a Pipeline with the following steps
        #   1. Reshape data back to time series format with one feature to enable appropriate scaling
        #   2. Scale data with MinMaxScaler
        #   3. Reshape to lstm format (n_samples, n_timesteps, n_features)
        #   4. Fit the model
        lstm_pipeline = Pipeline([
            ('time_series_reshape', forecaster_utils.TimeSeriesReshaper()),
            ('scale', MinMaxScaler(self._scaler_range)),
            ('lstm_reshape', forecaster_utils.LSTMReshaper(*self._input_shape)),
            ('lstm', wrapped_model)
        ])

        # Add prefix to parameters in grid to make it work with Pipeline and GridSearchCV
        pipeline_search_space = {}
        prefix = 'lstm__regressor__'

        for key in parameter_grid.keys():
            pipeline_search_space[prefix+key] = parameter_grid[key]

        pipeline_search_space = [pipeline_search_space]

        # Set up GridSearchCV
        grid_search = GridSearchCV(
                            estimator=lstm_pipeline, 
                            param_grid=pipeline_search_space, 
                            scoring=scoring, 
                            n_jobs=n_jobs, 
                            refit=True, 
                            cv=time_series_split_cv, 
                            verbose=0)

        if self._logger: 
            self._logger.info(f'({self.name}) Starting Hyperparameter tuning with grid search using {n_jobs} Jobs.')
            n_combinations = len(ParameterGrid(parameter_grid))
            self._logger.info(f'({self.name}) Fitting {n_splits} folds for each of {n_combinations} candidates, totalling {n_splits*n_combinations} fits.')

        # Search for the best model on the training data
        result = grid_search.fit(X_train, y_train)

        if self._logger: self._logger.info(f'({self.name}) Hyperparameter tuning finished.')

        # get model of pipeline
        best_model = result.best_estimator_.steps[3][1].regressor_.model

        # get best model paramters
        best_params = {}
        for key in result.best_params_.keys():
            new_key = key[len(prefix):]
            best_params[new_key] = result.best_params_[key]

        if self._logger: self._logger.info(f'({self.name}) Evaluating best model.')
        # Best mean cv test score
        best_score = result.best_score_ * -1

        # Predict training data and plot it against original time series
        train_predictions_best_estimator = result.best_estimator_.predict(X_train)
        train_score = mean_squared_error(y_train, train_predictions_best_estimator, squared=False)

        save_path = self._evaluation_dir_path / str(self.name+'_training_data_prediction.png')
        title = self.name+f' Training Data Prediction (RMSE = {train_score:.2f})'
        forecaster_utils.plot_predictions(time_series = self._time_series, predicted_data=train_predictions_best_estimator, title=title, save_path=save_path, mode='train', time_steps=self._input_shape[0])

        # Predict test data and plot it against hold-out test set
        n_periods = y_test.shape[0]
        test_predictions_best_estimator = result.best_estimator_.predict(X_test)
        test_score = mean_squared_error(y_test, test_predictions_best_estimator, squared=False)

        save_path = self._evaluation_dir_path / str(self.name+'_test_data_prediction.png')
        title = self.name+f' Hold-Out Set Prediction (RMSE = {test_score:.2f})'
        forecaster_utils.plot_predictions(time_series = self._time_series, predicted_data=test_predictions_best_estimator, title=title, save_path=save_path, mode='test')
        
        # Write lines to report file
        report_file_lines.append('\tModel-Structure:')
        best_model.summary(print_fn=lambda line: report_file_lines.append('\t\t'+line))
        report_file_lines.append('\n\tModel-Hyperparameter:')
        report_file_lines.append(self._format_parameter_output(best_params))
        report_file_lines.append(model_creator_utils.get_title_line(f'Mean CV Test-Score ({n_splits} Splits) ({test_size} Test Periods each)'))
        report_file_lines.append(f'\tRMSE: {best_score:.4f} \n')
        report_file_lines.append(model_creator_utils.get_title_line(f'Training-Score (Refit Model)'))
        report_file_lines.append(f'\tRMSE: {train_score:.4f} \n')
        report_file_lines.append(model_creator_utils.get_title_line(f'Test-Score (Refit Model) ({n_periods} Test Periods each)'))
        report_file_lines.append(f'\tRMSE: {test_score:.4f} \n')
        model_creator_utils.write_to_report_file(report_file_lines, self._evaluation_file_path)

        if self._logger: self._logger.info(f'({self.name}) Evaluation finished, evaluation file and plots are now accessible.')

        return best_model, best_params
        

    def _set_final_model(self, hyperparameters: dict, X: np.ndarray, y: np.ndarray) -> Sequential:
        """Set a final forecasting model.

        Builds a final forecasting model with the given hyperparameters and trains it on the given data (should include the values of the whole time series). The model is then stored on disk and can then be used for forecasting via the ``make_forecast()`` function.

        Args:
            hyperparameters (dict):
                A dict defining the hyperparameters of the form {'hyperparamer1': value, 'hyperparameter2': value, ...}.

            X (array):
                Values used for training in the form of sequences as np.ndarray with shape (n_samples, n_timesteps)
            
            y (array):
                Target values as np.ndarray with shape (n_samples, 1)
        """

        # Remove .fit() parameters from hyperparameter dict
        epochs = hyperparameters.pop('epochs', None)
        batch_size = hyperparameters.pop('batch_size', None)

        # Build the model with the given parameters and fit it on the given data
        model = self._build_model(**hyperparameters)

        # Reshape data for scaling
        X = X.reshape(-1,1)

        # Transform the data with the scaler fitted on the whole time series
        X = self._final_scaler.transform(X)
        
        # Reshape back to lstm format
        X = X.reshape(-1, *self._input_shape)

        # Scale the target with the same scaler 
        y = self._final_scaler.transform(y)
        
        # Fit the model
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

        if self._logger: self._logger.info(f'({self.name}) Storing final model to disk.')

        # Store the model on disk
        model.save(str(self._model_path))

        # set final model to model
        self._final_model = model


    def _check_hyperparameters_validity(self, hyperparameters: dict):
        """ Checks if the given hyperparameters are valid.

        Args:
            hyperparameters (dict):
                A dict of the form {'parameter_1': [value_1, value_2, ...], 'parameter_2': [...], ...}.

        Raises:
            TypeError: If the hyperparameters have the wrong type.
            ValueError: If hyperparameters are an empty dict or the a parameter has no attached value list.
            NameError: If invalid parameter names are present.
        """
        valid_keys = ['neurons_layer_1', 'neurons_layer_1', 'neurons_layer_2', 'recurrent_dropout', 'epochs', 'learning_rate', 'batch_size']

        if isinstance(hyperparameters, dict):
            if not hyperparameters:
                raise ValueError(f'Hyperparameters is an empty dict. Please specify supported hyperparameters. Supported parameters are: {", ".join(val_key for val_key in valid_keys)}')

            for key in hyperparameters.keys():
                if key in valid_keys:
                    parameter = hyperparameters[key]
                    if isinstance(parameter, list):
                        if not parameter:
                            raise ValueError(f'Valuelist for hyperparameter {key} is empty. Please specify values for the hyperparameter {key} or remove it from HYPERPARAMS.')
                        
                    else:
                        raise TypeError(f'The hyperparameter {key} must be of type \'list\', got type \'{type(parameter).__name__}\', make sure that every specified hyperparameter is wrapped in a list.')

                else:
                    raise NameError(f'The specified hyperparameter {key} is not a supported paramter of LSTMForecaster, make sure to only use supported parameters. Supported parameters are: {", ".join(val_key for val_key in valid_keys)}')
        else:
            raise TypeError(f'Hyperparameters must be of type \'dict\', passed hyperparameters are of type \'{type(hyperparameters).__name__}\'')


   
    def _format_parameter_output(self, parameters: dict) -> str:
        """Formats hyperparameters for printing.

        Helper function to format model hyperparameters for writing them into the evaluation file.

        Args:
            parameters (dict):
                A dict defining the hyperparameters of the form {'hyperparamer1': value, 'hyperparameter2': value, ...}.

        Returns:
            A formatted string
        """
        output = ''
        hyper_param_list = ['recurrent_dropout','epochs', 'learning_rate', 'batch_size']
        for key, value in parameters.items():
            if key in hyper_param_list:
                output = output + '\t\t' + str(key) + ': ' + str(value) + '\n'

        return output