import pytest
import pandas as pd

from path_definitions import PATHS
from modelcreator.datahandler.data_handler import DataHandler
from modelcreator.datahandler.preprocessor.sales_forecaster_preprocessor import SalesForecasterPreprocessor
from modelcreator.salesforecaster.arima_forecaster import ARIMAForecaster
from modelcreator.exception import IncompatibleDataError, ModelNotFoundError, NoTrainedModelError


class TestingUtils():

    @staticmethod
    def get_forecaster_name():
        return 'pytest_test_arima_forecaster'

    @staticmethod
    def get_test_size():
        return 12

    @staticmethod
    def get_valid_hyperparameters():
        return {'p': [1], 'd': [1], 'q': [1], 'P': [0], 'D': [0], 'Q': [0], 'm': [0]}

    @staticmethod
    def get_value_error_hyperparameters():
        hyperparameters_1 = {}
        hyperparameters_2 = {'p': [], 'd': [], 'q': [], 'P': [], 'D': [], 'Q': [], 'm': []} 

        return [hyperparameters_1, hyperparameters_2]

    @staticmethod
    def get_type_error_hyperparameters():
        hyperparameters_1 = [{'p': [1], 'd': [1], 'q': [1], 'P': [0], 'D': [0], 'Q': [0], 'm': [0]}]
        hyperparameters_2 = {'p': ['1'], 'd': ['1'], 'q': ['1'], 'P': ['0'], 'D': ['0'], 'Q': ['0'], 'm': ['0']}
        hyperparameters_3 = {'p': 1, 'd': 1, 'q': 1, 'P': 0, 'D': 0, 'Q': 0, 'm': 0}

        return [hyperparameters_1, hyperparameters_2, hyperparameters_3]

    @staticmethod
    def get_n_periods():
        return 4

    @staticmethod
    def get_result_plot_path(name='pytest_test_arima_forecaster'):
        return PATHS['EVALUATIONS_DIR'] / 'ARIMAForecaster' / name / (name+'_test_data_prediction.png')

    @staticmethod
    def get_data():
        handler = DataHandler('online_retail_II.csv', SalesForecasterPreprocessor())
        return handler.generate_preprocessed_data(False, True)

    @staticmethod
    def get_forecaster():
        name = TestingUtils.get_forecaster_name()
        data = TestingUtils.get_data()
        forecaster = ARIMAForecaster(name, data)

        test_size = TestingUtils.get_test_size()
        hyperparameters = TestingUtils.get_valid_hyperparameters()

        forecaster.initialize_forecaster(test_size, hyperparameters)

        return forecaster

    @staticmethod
    def clean_directories(name='pytest_test_arima_forecaster', mode=''):

        eval_dir_path = PATHS['EVALUATIONS_DIR'] / 'ARIMAForecaster' / name
        eval_file_path = eval_dir_path / (name+'_evaluation.txt')
        eval_plot_file_path = eval_dir_path / (name+'_test_data_prediction.png')
        
        model_dir_path = PATHS['MODELS_DIR'] / 'ARIMAForecaster'
        model_file_path = model_dir_path / (name+'.pickle')
   
        results_dir_path = PATHS['RESULTS_DIR'] / 'ARIMAForecaster' / name
        results_plot_file_path = results_dir_path / (name+f'_forecast_{TestingUtils.get_n_periods()}_periods.png')

        eval_file_path.unlink(missing_ok=True)
        eval_plot_file_path.unlink(missing_ok=True)

        if eval_dir_path.exists():
            eval_dir_path.rmdir()

        model_file_path.unlink(missing_ok=True)
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
    """Tests a proper initialization of an ARIMAForecaster instance.
        
        Expected: No error should be raised.
    """
    forecaster_data_path = PATHS['PROCESSED_DATA_DIR'] / 'SalesForecasterData.feather'
    forecaster_data_path.unlink(missing_ok=True)

    name = testing_utils.get_forecaster_name()
    data = testing_utils.get_data()
    try:
        ARIMAForecaster(name, data)
    except:
        pytest.fail('Should initialize ARIMAForecaster instance properly.')


@pytest.mark.dependency(depends=['test_proper__init'])
def test_empty__init():
    """Tests the behavior of intitializing an ARIMAForecaster with no parameters.

        Expected: TypeError should be raised.
    """

    with pytest.raises(TypeError):
        ARIMAForecaster()


@pytest.mark.dependency(depends=['test_proper__init'])
def test_wrongly_named__init(testing_utils):
    """Tests the behavior of intitializing an ARIMAForecaster with not allowed names.

        Expected: ValueError should be raised in all cases.
    """

    data = testing_utils.get_data()

    with pytest.raises(ValueError):
        ARIMAForecaster('', data)

    with pytest.raises(ValueError):
        ARIMAForecaster('no_special_characters?!', data)

    with pytest.raises(ValueError):
        ARIMAForecaster('no spaces', data)


@pytest.mark.dependency(depends=['test_proper__init'])
def test_proper__initialize_forecaster(testing_utils):
    """Tests the proper usage of the initialize_forecaster function.

        Expected: No error should be raised.
    """

    testing_utils.clean_directories()

    try:
        name = testing_utils.get_forecaster_name()

        eval_dir_path = PATHS['EVALUATIONS_DIR'] / 'ARIMAForecaster' / name
        eval_file_path = eval_dir_path / (name+'_evaluation.txt')
        eval_plot_file_path = eval_dir_path / (name+'_test_data_prediction.png')
        
        model_dir_path = PATHS['MODELS_DIR'] / 'ARIMAForecaster'
        model_file_path = model_dir_path / (name+'.pickle')

        data = testing_utils.get_data()
        test_size = testing_utils.get_test_size()
        hyperparameters = testing_utils.get_valid_hyperparameters()

        forecaster = ARIMAForecaster(name, data)
        forecaster.initialize_forecaster(test_size, hyperparameters)

        assert eval_dir_path.exists(), 'An evaluation folder for the forcaster should have been created'
        assert eval_file_path.exists(), 'An evaluation file should have been created'
        assert eval_plot_file_path.exists(), 'A plot of the test data prediction should have been created'
        assert model_file_path.exists(), 'A resulting model should have been stored on disk'

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
    forecaster = ARIMAForecaster(name, data)
    
    with pytest.raises(TypeError):
        forecaster.initialize_forecaster('12', hyperparameters)

    with pytest.raises(TypeError):
        forecaster.initialize_forecaster(12, ['wrong_parameters'])


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

    forecaster = ARIMAForecaster(name, data)

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

    forecaster = ARIMAForecaster(name, data)

    with pytest.raises(TypeError):
        forecaster.initialize_forecaster(test_size, hyperparameter)

@pytest.mark.dependency(depends=['test_proper__initialize_forecaster'])
def test_wrong_hyperparameters_name_error__initialize_forecaster(testing_utils):
    """Tests the behavior of calling initialize_forecaster() with invalid hyperparameter settings.

        Expected: NameError should be raised.
    """

    testing_utils.clean_directories()

    name = testing_utils.get_forecaster_name()
    data = testing_utils.get_data()
    test_size = testing_utils.get_test_size()

    hyperparameters = {'p': [1], 'd': [1], 'q': [1], 'P': [0], 'D': [0], 'Q': [0], 'm': [0], 'additional': [10]}

    forecaster = ARIMAForecaster(name, data)

    with pytest.raises(NameError):
        forecaster.initialize_forecaster(test_size, hyperparameters)


@pytest.mark.dependency(depends=['test_proper__initialize_forecaster'])
def test_wrong_hyperparameters_key_error__initialize_forecaster(testing_utils):
    """Tests the behavior of calling initialize_forecaster() with invalid hyperparameter settings.

        Expected: KeyError should be raised.
    """
    testing_utils.clean_directories()

    name = testing_utils.get_forecaster_name()
    data = testing_utils.get_data()
    test_size = testing_utils.get_test_size()

    hyperparameters = {'p': [1], 'd': [1], 'q': [1]}

    forecaster = ARIMAForecaster(name, data)

    with pytest.raises(KeyError):
        forecaster.initialize_forecaster(test_size, hyperparameters)


@pytest.mark.dependency(depends=['test_proper__initialize_forecaster'])
def test_load_existing_model__init(testing_utils):
    """Tests the behavior of properly intitializing an ARIMAForecaster with load_existing_model = True.

        Expected: No error should be raised.
    """
    testing_utils.clean_directories()

    name = testing_utils.get_forecaster_name()
    data = testing_utils.get_data()
    test_size = testing_utils.get_test_size()
    hyperparameters = testing_utils.get_valid_hyperparameters()

    forecaster = ARIMAForecaster(name, data, load_existing_model = False)
    forecaster.initialize_forecaster(test_size, hyperparameters)

    try:
        ARIMAForecaster(name, data, load_existing_model = True)
    except:
        pytest.fail('Should execute initialize_forecaster(load_existing_model = True) properly.')


@pytest.mark.dependency(depends=['test_load_existing_model__init'])
def test_load_not_existing_model__init(testing_utils):
    """Tests the behavior of intitializing an ARIMAForecaster with load_existing_model = True if the model is not existing.

        Expected: ModelNotFoundError should be raised.
    """
    testing_utils.clean_directories()

    name = testing_utils.get_forecaster_name()
    data = testing_utils.get_data()

    with pytest.raises(ModelNotFoundError):
        ARIMAForecaster(name, data, load_existing_model = True)


@pytest.mark.dependency(depends=['test_load_existing_model__init'])
def test_load_existing_model_opposite_transformation__init(testing_utils):
    """Tests the behavior of intitializing an ARIMAForecaster with load_existing_model = True, but with the opposite setting for apply_box_cox opposed to how it was first initialized.

        Expected: IncompatibleDataError should be raised in both cases.
    """
    testing_utils.clean_directories()

    name = testing_utils.get_forecaster_name()
    data = testing_utils.get_data()
    test_size = testing_utils.get_test_size()
    hyperparameters = testing_utils.get_valid_hyperparameters()

    ARIMAForecaster(name, data, apply_box_cox=True, load_existing_model=False).initialize_forecaster(test_size, hyperparameters)

    with pytest.raises(IncompatibleDataError):
        ARIMAForecaster(name, data, apply_box_cox=False, load_existing_model=True)

    testing_utils.clean_directories()

    ARIMAForecaster(name, data, apply_box_cox=False, load_existing_model=False).initialize_forecaster(test_size, hyperparameters)

    with pytest.raises(IncompatibleDataError):
        ARIMAForecaster(name, data, apply_box_cox=True, load_existing_model=True)


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
    """Tests the behavior of calling make_forecast() when it is called from an uninitialized ARIMAForecaster.
        
        Expected: NoTrainedModelError should be raised.
    """
    testing_utils.clean_directories()

    name = testing_utils.get_forecaster_name()
    data = testing_utils.get_data()
    n_periods = testing_utils.get_n_periods()
    forecaster = ARIMAForecaster(name, data)

    # when loading an existing model which originally had apply_box_cox = False
    with pytest.raises(NoTrainedModelError):
        forecaster.make_forecast(n_periods)


@pytest.fixture(scope="session", autouse=True)
def final_cleanup(request):
    """Cleans directories when all tests are finished."""
    request.addfinalizer(lambda: TestingUtils.clean_directories(mode='finalize'))
