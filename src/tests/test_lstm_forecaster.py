import pytest
import shutil
import pandas as pd

from path_definitions import PATHS

from modelcreator.datahandler.data_handler import DataHandler
from modelcreator.datahandler.preprocessor.sales_forecaster_preprocessor import SalesForecasterPreprocessor
from modelcreator.salesforecaster.lstm_forecaster import LSTMForecaster
from modelcreator.exception import ModelNotFoundError, NoTrainedModelError


class TestingUtils():

    @staticmethod
    def get_forecaster_name():
        return 'pytest_test_lstm_forecaster'

    @staticmethod
    def get_test_size():
        return 12

    @staticmethod
    def get_valid_hyperparameters():
        return  {'neurons_layer_1': [1], 'neurons_layer_2' : [0], 'recurrent_dropout': [0.1], 'epochs': [10], 'learning_rate': [0.01], 'batch_size': [1]}

    @staticmethod
    def get_value_error_hyperparameters():
        hyperparameters_1 = {}
        hyperparameters_2 = {'neurons_layer_1': [], 'learning_rate': [], 'batch_size': []}  
        return [hyperparameters_1, hyperparameters_2]

    @staticmethod
    def get_type_error_hyperparameters():
        hyperparameters_1 = [{'neurons_layer_1': [5], 'learning_rate': [0.01], 'batch_size': [50]}]
        hyperparameters_2 = {'neurons_layer_1': 5, 'learning_rate': 0.01, 'batch_size': 50}

        return [hyperparameters_1, hyperparameters_2]

    @staticmethod
    def get_n_periods():
        return 4

    @staticmethod
    def get_result_plot_path(name='pytest_test_lstm_forecaster'):
        return PATHS['EVALUATIONS_DIR'] / 'LSTMForecaster' / name / (name+'_test_data_prediction.png')

    @staticmethod
    def get_data():
        handler = DataHandler('online_retail_II.csv', SalesForecasterPreprocessor())
        data = handler.generate_preprocessed_data(False, True)
        return data

    @staticmethod
    def get_forecaster():
        name = TestingUtils.get_forecaster_name()
        data = TestingUtils.get_data()
        forecaster = LSTMForecaster(name, data)

        test_size = TestingUtils.get_test_size()
        hyperparameters = TestingUtils.get_valid_hyperparameters()

        forecaster.initialize_forecaster(test_size, hyperparameters)

        return forecaster

    @staticmethod
    def clean_directories(name='pytest_test_lstm_forecaster', mode=''):

        eval_dir_path = PATHS['EVALUATIONS_DIR'] / 'LSTMForecaster' / name
        eval_file_path = eval_dir_path / (name+'_evaluation.txt')
        eval_test_plot_file_path = eval_dir_path / (name+'_test_data_prediction.png')
        eval_train_plot_file_path = eval_dir_path / (name+'_training_data_prediction.png')
        
        model_dir_path = PATHS['MODELS_DIR'] / 'LSTMForecaster'
        model_files_path = model_dir_path / name
   
        results_dir_path = PATHS['RESULTS_DIR'] / 'LSTMForecaster' / name
        results_plot_file_path = results_dir_path / (name+f'_forecast_{TestingUtils.get_n_periods()}_periods.png')
        
        eval_file_path.unlink(missing_ok=True)
        eval_test_plot_file_path.unlink(missing_ok=True)
        eval_train_plot_file_path.unlink(missing_ok=True)

        if eval_dir_path.exists():
            eval_dir_path.rmdir()

        shutil.rmtree(model_files_path, ignore_errors = True)

        results_plot_file_path.unlink(missing_ok=True)

        if results_dir_path.exists():
            results_dir_path.rmdir()

        # also delete generated files at the end
        if mode == 'finalize':
            forecaster_data_path = PATHS['PROCESSED_DATA_DIR'] / 'SalesForecasterData.feather'
            forecaster_data_path.unlink(missing_ok=True)


@pytest.fixture
def testing_utils():
    return TestingUtils


@pytest.mark.dependency
def test_proper__init(testing_utils):
    """Tests a proper initialization of an LSTMForecaster instance.
        
        Expected: No error should be raised.
    """
    forecaster_data_path = PATHS['PROCESSED_DATA_DIR'] / 'SalesForecasterData.feather'
    forecaster_data_path.unlink(missing_ok=True)

    name = testing_utils.get_forecaster_name()
    data = testing_utils.get_data()
    try:
        LSTMForecaster(name, data)
    except:
        pytest.fail('Should initialize LSTMForecaster instance properly.')


@pytest.mark.dependency(depends=['test_proper__init'])
def test_empty__init():
    """Tests the behavior of intitializing an LSTMForecaster with no parameters.

        Expected: TypeError should be raised.
    """

    with pytest.raises(TypeError):
        LSTMForecaster()


@pytest.mark.dependency(depends=['test_proper__init'])
def test_wrongly_named__init(testing_utils):
    """Tests the behavior of intitializing an LSTMForecaster with not allowed names.

        Expected: ValueError should be raised in all cases.
    """

    data = testing_utils.get_data()

    with pytest.raises(ValueError):
        LSTMForecaster('', data)

    with pytest.raises(ValueError):
        LSTMForecaster('no_special_characters?!', data)

    with pytest.raises(ValueError):
        LSTMForecaster('no spaces', data)


@pytest.mark.dependency(depends=['test_proper__init'])
def test_proper__initialize_forecaster(testing_utils):
    """Tests the proper usage of the initialize_forecaster function.

        Expected: No error should be raised.
    """
    testing_utils.clean_directories()

    try:
        name = testing_utils.get_forecaster_name()

        eval_dir_path = PATHS['EVALUATIONS_DIR'] / 'LSTMForecaster' / name
        eval_file_path = eval_dir_path / (name+'_evaluation.txt')
        eval_test_plot_file_path = eval_dir_path / (name+'_test_data_prediction.png')
        eval_train_plot_file_path = eval_dir_path / (name+'_training_data_prediction.png')
        model_files_path = PATHS['MODELS_DIR'] / 'LSTMForecaster' / name

        data = testing_utils.get_data()
        test_size = testing_utils.get_test_size()
        hyperparameters = testing_utils.get_valid_hyperparameters()

        forecaster = LSTMForecaster(name, data)
        forecaster.initialize_forecaster(test_size, hyperparameters)

        assert eval_dir_path.exists(), 'An evaluation folder for the forcaster should have been created'
        assert eval_file_path.exists(), 'An evaluation file should have been created'
        assert eval_test_plot_file_path.exists(), 'A plot of the test data prediction should have been created'
        assert eval_train_plot_file_path.exists(), 'A plot of the training data prediction should have been created'
        assert model_files_path.exists(), 'A resulting model should have been stored on disk'

    except:
        pytest.fail('Should execute initialize_forecaster() properly.')


@pytest.mark.dependency(depends=['test_proper__initialize_forecaster'])
def test_wrongly_typed__initialize_forecaster(testing_utils):
    """Tests the behavior of calling initialize_forecaster with wrongly typed parameters.

        Expected: TypeError should be raised in all cases.
    """
    testing_utils.clean_directories()

    name = testing_utils.get_forecaster_name()
    data = testing_utils.get_data()
    hyperparameters = testing_utils.get_valid_hyperparameters()
    forecaster = LSTMForecaster(name, data)
    
    with pytest.raises(TypeError):
        forecaster.initialize_forecaster('12', hyperparameters)

    with pytest.raises(TypeError):
        forecaster.initialize_forecaster(12, ['wrong_parameters'])


@pytest.mark.dependency(depends=['test_proper__initialize_forecaster'])
def test_wrong_hyperparameters_name_error__initialize_forecaster(testing_utils):
    """Tests the behavior of calling initialize_forecaster() with invalid hyperparameter settings.

        Expected: NameError should be raised.
    """
    testing_utils.clean_directories()

    name = testing_utils.get_forecaster_name()
    data = testing_utils.get_data()
    test_size = testing_utils.get_test_size()

    hyperparameters = {'neurons_layer_1': [5], 'neurons_layer_2': [10], 'neurons_layer_3': [15]}

    forecaster = LSTMForecaster(name, data)
    
    with pytest.raises(NameError):
        forecaster.initialize_forecaster(test_size, hyperparameters)


@pytest.mark.dependency(depends=['test_proper__initialize_forecaster'])
@pytest.mark.parametrize('hyperparameter', TestingUtils.get_value_error_hyperparameters())
def test_wrong_hyperparameters_value_error__initialize_forecaster(hyperparameter, testing_utils):
    """Tests the behavior of calling initialize_forecaster() with invalid hyperparameter settings.

        Expected: ValueError should be raised.
    """
    testing_utils.clean_directories()

    name = testing_utils.get_forecaster_name()
    data = testing_utils.get_data()
    test_size = testing_utils.get_test_size()

    forecaster = LSTMForecaster(name, data)
    
    with pytest.raises(ValueError):
        forecaster.initialize_forecaster(test_size, hyperparameter)


@pytest.mark.dependency(depends=['test_proper__initialize_forecaster'])
@pytest.mark.parametrize('hyperparameter', TestingUtils.get_type_error_hyperparameters())
def test_wrong_hyperparameters_type_error__initialize_forecaster(hyperparameter, testing_utils):
    """Tests the behavior of calling initialize_forecaster() with invalid hyperparameter settings.

        Expected: TypeError should be raised.
    """
    testing_utils.clean_directories()

    name = testing_utils.get_forecaster_name()
    data = testing_utils.get_data()
    test_size = testing_utils.get_test_size()

    forecaster = LSTMForecaster(name, data)
    
    with pytest.raises(TypeError):
        forecaster.initialize_forecaster(test_size, hyperparameter)


@pytest.mark.dependency(depends=['test_proper__initialize_forecaster'])
def test_load_existing_model__init(testing_utils):
    """Tests the behavior of properly intitializing an LSTMForecaster with load_existing_model = True.

        Expected: No error should be raised.
    """
    testing_utils.clean_directories()

    name = testing_utils.get_forecaster_name()
    data = testing_utils.get_data()
    test_size = testing_utils.get_test_size()
    hyperparameters = testing_utils.get_valid_hyperparameters()

    LSTMForecaster(name, data).initialize_forecaster(test_size, hyperparameters)

    try:
        forecaster = LSTMForecaster(name, data, load_existing_model = True)
    except:
        pytest.fail('Should execute initialize_forecaster(load_existing_model = True) properly.')
        

@pytest.mark.dependency(depends=['test_load_existing_model__init'])
def test_load_not_existing_model__init(testing_utils):
    """Tests the behavior of intitializing an LSTMForecaster with load_existing_model = True if the model is not existing.

        Expected: ModelNotFoundError should be raised.
    """
    testing_utils.clean_directories()

    name = testing_utils.get_forecaster_name()
    data = testing_utils.get_data()

    with pytest.raises(ModelNotFoundError):
        LSTMForecaster(name, data, load_existing_model = True)


@pytest.mark.dependency(depends=['test_load_existing_model__init'])
def test_proper__make_forecast(testing_utils):
    """Tests the proper usage of the make_forecast function.
        
        Expected: No error should be raised.
    """
    testing_utils.clean_directories()

    forecaster = testing_utils.get_forecaster()
    n_periods = testing_utils.get_n_periods()
    result_plot_path = testing_utils.get_result_plot_path()

    try: 
        prediction = forecaster.make_forecast(n_periods)
        assert isinstance(prediction, pd.Series), 'should be pd.Series'
        assert len(prediction) == n_periods, f'predictions should have a lenght of {n_periods}'
        assert result_plot_path.exists(), 'path to result plot should exist after forecasting'

    except:
         pytest.fail('Should execute make_forecast() properly.')


@pytest.mark.dependency(depends=['test_proper__make_forecast'])
def test_random_state__make_forecast(testing_utils):
    """Tests the behavior of calling make_forecast() with a set random state.
        
        Expected: No error should be raised.
    """
    testing_utils.clean_directories()

    name = testing_utils.get_forecaster_name()
    name_random = testing_utils.get_forecaster_name() + 'random_state'
    
    data = testing_utils.get_data()
    test_size = testing_utils.get_test_size()
    hyperparamters = testing_utils.get_valid_hyperparameters()
    n_periods = testing_utils.get_n_periods()

    try:
        # check without set non_random
        forecaster = LSTMForecaster(name, data)
        forecaster.initialize_forecaster(test_size, hyperparamters)
        prediction_1 = forecaster.make_forecast(n_periods)

        testing_utils.clean_directories()

        # Overwrite old forecaster
        forecaster = LSTMForecaster(name, data)
        forecaster.initialize_forecaster(test_size, hyperparamters)
        prediction_2 = forecaster.make_forecast(n_periods)

        testing_utils.clean_directories()

        assert not prediction_1.equals(prediction_2), 'Predictions should be different'

        # check with set random_state
        forecaster = LSTMForecaster(name_random, data, random_state=1)
        forecaster.initialize_forecaster(test_size, hyperparamters)
        prediction_1 = forecaster.make_forecast(n_periods)

        testing_utils.clean_directories(name_random)

        # Overwrite old forecaster
        forecaster = LSTMForecaster(name_random, data, random_state=1)
        forecaster.initialize_forecaster(test_size, hyperparamters)
        prediction_2 = forecaster.make_forecast(n_periods)

        testing_utils.clean_directories(name_random)

        assert prediction_1.equals(prediction_2), 'Predictions should be equal'

    except:
        testing_utils.clean_directories()
        testing_utils.clean_directories(name_random)
        pytest.fail('Should execute make_forecast() properly with a set random state.')


@pytest.mark.dependency(depends=['test_proper__make_forecast'])
def test_wrongly_typed__make_forecast(testing_utils):
    """Tests the behavior of calling make_forecast() with wrongly typed parameters.
        
        Expected: TypeError should be raised.
    """
    testing_utils.clean_directories()

    forecaster = testing_utils.get_forecaster()

    with pytest.raises(TypeError):
        forecaster.make_forecast('5')
    
    with pytest.raises(TypeError):
        forecaster.make_forecast(4.32)


@pytest.mark.dependency(depends=['test_proper__make_forecast'])
def test_too_large_n_periods__make_forecast(testing_utils):
    """Tests the behavior of calling make_forecast() with a too large n_periods parameter.
        
        Expected: ValueError should be raised.
    """
    testing_utils.clean_directories()

    forecaster = testing_utils.get_forecaster()

    with pytest.raises(ValueError):
        forecaster.make_forecast(-5)

    with pytest.raises(ValueError):
        forecaster.make_forecast(53)


@pytest.mark.dependency(depends=['test_proper__make_forecast'])
def test_wrong_instance__make_forecast(testing_utils):
    """Tests the behavior of calling make_forecast() when it is called from an uninitialized LSTMForecaster.
        
        Expected: NoTrainedModelError should be raised.
    """
    testing_utils.clean_directories()

    name = testing_utils.get_forecaster_name()
    data = testing_utils.get_data()
    n_periods = testing_utils.get_n_periods()
    forecaster = LSTMForecaster(name, data)

    # when loading an existing model which originally had apply_box_cox = False
    with pytest.raises(NoTrainedModelError):
        forecaster.make_forecast(n_periods)


@pytest.fixture(scope="session", autouse=True)
def final_cleanup(request):
    """Cleans directories when all tests are finished."""
    request.addfinalizer(lambda: TestingUtils.clean_directories(mode='finalize'))
