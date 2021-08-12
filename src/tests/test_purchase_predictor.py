import pytest
import pandas as pd
import numpy as np

from path_definitions import PATHS
from modelcreator.datahandler.data_handler import DataHandler
from modelcreator.datahandler.preprocessor.purchase_predictor_preprocessor import PurchasePredictorPreprocessor
from modelcreator.purchasepredictor.purchase_predictor import PurchasePredictor
from modelcreator.exception import IncompatibleDataError, NoTrainedModelError, ModelNotFoundError


class TestingUtils():

    @staticmethod
    def get_test_size():
        return 0.2

    @staticmethod
    def get_predictor_name():
        return 'pytest_test_purchase_predictor'

    @staticmethod
    def get_valid_settings():
        return [{'CLF_NAME': 'LogisticRegression', 'HYPERPARAMS': {'penalty': ['l2']}}]

    @staticmethod
    def get_key_error_settings():
        setting_1 = [{}, {}]
        setting_2 = [{'CLF': 'RandomForestClassifier'}]
        return [setting_1, setting_2]
    
    @staticmethod
    def get_name_error_settings():
        setting_1 = [{'CLF_NAME': 'RandomForestClassifier', 'HYPERPARAMS': {'make_my_model_good': [True]}}]
        setting_2 = [{'CLF_NAME': 'MagicModel'}]
        return [setting_1, setting_2]
    
    @staticmethod
    def get_type_error_settings():
        setting_1 = {}
        setting_2 = [{'CLF_NAME': 'RandomForestClassifier', 'HYPERPARAMS': {'n_estimators': 1}}]
        setting_3 = [{'CLF_NAME': 'RandomForestClassifier'}, [{'CLF_NAME': 'LogisticRegression'}]]
        return [setting_1, setting_2, setting_3]

    @staticmethod
    def get_value_error_settings():
        setting_1 = []
        setting_2 = [{'CLF_NAME': 'SVC'}]
        setting_2 = [{'CLF_NAME': 'RandomForestClassifier', 'HYPERPARAMS': []}]
        return [setting_1, setting_2]

    @staticmethod
    def get_customer():
        return 12352.0

    @staticmethod 
    def get_customer_list():
        return [12347.0, 12352.0, 12356.0]

    @staticmethod
    def get_data():
        handler = DataHandler('online_retail_II.csv', PurchasePredictorPreprocessor())
        data = handler.generate_preprocessed_data(False, True)
        return data

    @staticmethod
    def get_predictor():
        name = TestingUtils.get_predictor_name()
        data = TestingUtils.get_data()

        predictor = PurchasePredictor(name, data, True)

        test_size = TestingUtils.get_test_size()
        settings = TestingUtils.get_valid_settings()

        predictor.initialize_purchase_predictor(test_size, settings)

        return predictor

    @staticmethod
    def clean_directories(name='pytest_test_purchase_predictor', mode=''):

        name = TestingUtils.get_predictor_name()

        eval_dir_path = PATHS['EVALUATIONS_DIR'] / 'PurchasePredictor' / name
        eval_file_path = eval_dir_path / (name+'_evaluation.txt')
        summed_conf_mat_path = eval_dir_path / (name+'_summed_cv_confusion_matrix.png')
        conf_mat_path = eval_dir_path / (name+'_test_confusion_matrix.png')
        model_file_path = PATHS['MODELS_DIR'] / 'PurchasePredictor'  / (name + '.pickle')

        eval_file_path.unlink(missing_ok=True)
        summed_conf_mat_path.unlink(missing_ok=True)
        conf_mat_path.unlink(missing_ok=True)

        if eval_dir_path.exists():
            eval_dir_path.rmdir()

        model_file_path.unlink(missing_ok=True)

        # also delete generated files at the end
        if mode=='finalize':
            predictor_data_path = PATHS['PROCESSED_DATA_DIR'] / 'PurchasePredictorData.feather'
            predictor_data_path.unlink(missing_ok=True)



@pytest.fixture
def testing_utils():
    return TestingUtils


@pytest.mark.dependency
def test_proper__init(testing_utils):
    """Tests a proper initialization of a PurchasePredictor instance.
        
        Expected: No error should be raised.
    """
    name = testing_utils.get_predictor_name()
    data = testing_utils.get_data()

    try:
        predictor = PurchasePredictor(name, data, apply_standardization=True)
    except:
        pytest.fail('Should initialize PurchasePredictor instance properly.')


@pytest.mark.dependency(depends=['test_proper__init'])
def test_empty__init():
    """Tests the behavior of intitializing a PurchasePredictor with no parameters.

        Expected: TypeError should be raised.
    """
    with pytest.raises(TypeError):
        predictor = PurchasePredictor()


@pytest.mark.dependency(depends=['test_proper__init'])
def test_wrongly_typed__init(testing_utils):
    """Tests the behavior of intitializing a PurchasePredictor with wrongly typed parameters.

        Expected: TypeError should be raised in all cases.
    """
    name = testing_utils.get_predictor_name()
    data = testing_utils.get_data()

    with pytest.raises(TypeError):
        predictor = PurchasePredictor(1, data, True)

    with pytest.raises(TypeError):
        predictor = PurchasePredictor(name, data, 'True')

    with pytest.raises(TypeError):
        predictor = PurchasePredictor(name, [1,2,3], True)


@pytest.mark.dependency(depends=['test_proper__init'])
def test_wrongly_named__init(testing_utils):
    """Tests the behavior of intitializing a PurchasePredictor with not allowed names.

        Expected: ValueError should be raised in all cases.
    """

    data = testing_utils.get_data()

    with pytest.raises(ValueError):
        predictor = PurchasePredictor('', data, True)

    with pytest.raises(ValueError):
        predictor = PurchasePredictor('no_special_characters?!', data, True)

    with pytest.raises(ValueError):
        predictor = PurchasePredictor('no spaces', data, True)


@pytest.mark.dependency(depends=['test_proper__init'])
def test_proper__initialize_purchase_predictor(testing_utils):
    """Tests the proper usage of the initialize_purchase_predictor function.

        Expected:  No error should be raised.
    """
    testing_utils.clean_directories()

    try:
        name = testing_utils.get_predictor_name()

        eval_dir_path = PATHS['EVALUATIONS_DIR'] / 'PurchasePredictor' / name
        eval_file_path = eval_dir_path / (name+'_evaluation.txt')
        summed_conf_mat_path = eval_dir_path / (name+'_summed_cv_confusion_matrix.png')
        conf_mat_path = eval_dir_path / (name+'_test_confusion_matrix.png')
        model_file_path = PATHS['MODELS_DIR'] / 'PurchasePredictor'  / (name + '.pickle')

        data = testing_utils.get_data()
        test_size = testing_utils.get_test_size()
        settings = testing_utils.get_valid_settings()

        predictor = PurchasePredictor(name, data, True)
        predictor.initialize_purchase_predictor(test_size, settings)

        assert eval_dir_path.exists(), 'An evaluation folder for the purchase predictor should have been created'
        assert eval_file_path.exists(), 'An evaluation file should have been created'
        assert summed_conf_mat_path.exists(), 'A plot of a summed cv confusion matrix should have been stored on disk'
        assert conf_mat_path.exists(), 'A plot of a test confusion matrix should have been stored on disk'
        assert model_file_path.exists(), 'A purchase predictor model should have been stored on disk'

    except:

        pytest.fail('Should execute initialize_purchase_predictor() properly.')


@pytest.mark.dependency(depends=['test_proper__initialize_purchase_predictor'])
def test_wrongly_typed__initialize_purchase_predictor(testing_utils):
    """Tests the behavior of calling initialize_purchase_predictor() with wrongly typed parameters.

        Expected: TypeError should be raised in all cases.
    """
    testing_utils.clean_directories()

    name = testing_utils.get_predictor_name()
    data = testing_utils.get_data()
    test_size = testing_utils.get_test_size()
    settings = testing_utils.get_valid_settings()
    predictor = PurchasePredictor(name, data, True)

    with pytest.raises(TypeError):
        predictor.initialize_purchase_predictor('0.2', settings)

    with pytest.raises(TypeError):
        predictor.initialize_purchase_predictor(test_size, 'settings')


@pytest.mark.dependency(depends=['test_proper__initialize_purchase_predictor'])
@pytest.mark.parametrize('settings', TestingUtils.get_key_error_settings())
def test_wrong_hyperparameters_key_error__initialize_purchase_predictor(settings, testing_utils):
    """Tests the behavior of calling initialize_purchase_predictor() with invalid hyperparameter settings.

        Expected: KeyError should be raised.
    """
    testing_utils.clean_directories()

    name = testing_utils.get_predictor_name()
    data = testing_utils.get_data()
    test_size = testing_utils.get_test_size()
    
    predictor = PurchasePredictor(name, data, True)
    
    with pytest.raises(KeyError):
        predictor.initialize_purchase_predictor(test_size, settings)


@pytest.mark.dependency(depends=['test_proper__initialize_purchase_predictor'])
@pytest.mark.parametrize('settings', TestingUtils.get_name_error_settings())
def test_wrong_hyperparameters_name_error__initialize_purchase_predictor(settings, testing_utils):
    """Tests the behavior of calling initialize_purchase_predictor() with invalid hyperparameter settings.

        Expected: NameError should be raised.
    """
    testing_utils.clean_directories()

    name = testing_utils.get_predictor_name()
    data = testing_utils.get_data()
    test_size = testing_utils.get_test_size()
    
    predictor = PurchasePredictor(name, data, True)
    
    with pytest.raises(NameError):
        predictor.initialize_purchase_predictor(test_size, settings)


@pytest.mark.dependency(depends=['test_proper__initialize_purchase_predictor'])
@pytest.mark.parametrize('settings', TestingUtils.get_value_error_settings())
def test_wrong_hyperparameters_value_error__initialize_purchase_predictor(settings, testing_utils):
    """Tests the behavior of calling initialize_purchase_predictor() with invalid hyperparameter settings.

        Expected: ValuError should be raised.
    """
    testing_utils.clean_directories()

    name = testing_utils.get_predictor_name()
    data = testing_utils.get_data()
    test_size = testing_utils.get_test_size()
    
    predictor = PurchasePredictor(name, data, True)
    
    with pytest.raises(ValueError):
        predictor.initialize_purchase_predictor(test_size, settings)


@pytest.mark.dependency(depends=['test_proper__initialize_purchase_predictor'])
@pytest.mark.parametrize('settings', TestingUtils.get_type_error_settings())
def test_wrong_hyperparameters_type_error__initialize_purchase_predictor(settings, testing_utils):
    """Tests the behavior of calling initialize_purchase_predictor() with invalid hyperparameter settings.

        Expected: TypeError should be raised.
    """
    testing_utils.clean_directories()

    name = testing_utils.get_predictor_name()
    data = testing_utils.get_data()
    test_size = testing_utils.get_test_size()
    
    predictor = PurchasePredictor(name, data, True)
    
    with pytest.raises(TypeError):
        predictor.initialize_purchase_predictor(test_size, settings)

@pytest.mark.dependency(depends=['test_proper__initialize_purchase_predictor'])
def test_load_existing_model__init(testing_utils):
    """Tests the behavior of properly intitializing a PurchasePredictor with load_existing_model = True.

        Expected: No error should be raised.
    """
    testing_utils.clean_directories()

    name = testing_utils.get_predictor_name()
    data = testing_utils.get_data()
    test_size = testing_utils.get_test_size()
    settings = testing_utils.get_valid_settings()

    PurchasePredictor(name, data, apply_standardization=True).initialize_purchase_predictor(test_size, settings)

    try:
        PurchasePredictor(name, data, apply_standardization=True, load_existing_model=True)
    
    except:
        pytest.fail('Should execute initialize_recommender(load_existing_model = True) properly.')
   
@pytest.mark.dependency(depends=['test_load_existing_model__init'])
def test_load_not_existing_model__init(testing_utils):
    """Tests the behavior of intitializing a PurchasePredictor with load_existing_model = True if the model is not existing.

        Expected: ModelNotFoundError should be raised.
    """
    testing_utils.clean_directories()

    name = testing_utils.get_predictor_name()
    data = testing_utils.get_data()
    
    with pytest.raises(ModelNotFoundError):
        PurchasePredictor(name, data, apply_standardization=True, load_existing_model=True)


@pytest.mark.dependency(depends=['test_load_existing_model__init'])
def test_load_existing_model_opposite_transformation__init(testing_utils):
    """Tests the behavior of intitializing an PurchasePredictor with load_existing_model = True, but with the opposite setting for apply_standardization opposed to how it was first initialized.

        Expected: IncompatibleDataError should be raised in both cases.
    """
    testing_utils.clean_directories()

    name = testing_utils.get_predictor_name()
    data = testing_utils.get_data()
    test_size = testing_utils.get_test_size()
    settings = testing_utils.get_valid_settings()

    PurchasePredictor(name, data, apply_standardization=False).initialize_purchase_predictor(test_size, settings)

    with pytest.raises(IncompatibleDataError):
        PurchasePredictor(name, data, apply_standardization=True, load_existing_model=True)

    testing_utils.clean_directories()

    PurchasePredictor(name, data, apply_standardization=True).initialize_purchase_predictor(test_size, settings)

    with pytest.raises(IncompatibleDataError):
        PurchasePredictor(name, data, apply_standardization=False, load_existing_model=True)


@pytest.mark.dependency(depends=['test_proper__initialize_purchase_predictor'])
def test_proper_simple_customer_id__predict_if_customer_purchases_next_quarter(testing_utils):
    """Tests the proper usage of the predict_if_customer_purchases_next_quarter function with a given customer_id.

        Expected: No error should be raised.
    """
    testing_utils.clean_directories()

    predictor = testing_utils.get_predictor()
    customer = testing_utils.get_customer()

    try:
        label = predictor.predict_if_customer_purchases_next_quarter(customer)
        assert isinstance(label, int), 'Should be of type int'
        assert label == 0 or label == 1, 'Should be either one or zero'

    except:
        pytest.fail('Should execute initialize_recommender(load_existing_model = True) properly.')
   

@pytest.mark.dependency(depends=['test_proper_simple_customer_id__predict_if_customer_purchases_next_quarter'])
def test_proper_list_customer_id__predict_if_customer_purchases_next_quarter(testing_utils):
    """Tests the proper usage of the predict_if_customer_purchases_next_quarter function with a given list of customer_id's.

        Expected: No error should be raised.
    """
    testing_utils.clean_directories()

    predictor = testing_utils.get_predictor()
    customers = testing_utils.get_customer_list()

    try:
        labels = predictor.predict_if_customer_purchases_next_quarter(customers)
        assert isinstance(labels, pd.Series), 'Should be of type Series'
        assert len(labels) == len(customers), 'Should be of same length'

    except:
        pytest.fail('Should execute initialize_recommender(load_existing_model = True) properly.')


@pytest.mark.dependency(depends=['test_proper_simple_customer_id__predict_if_customer_purchases_next_quarter'])
def test_wrong_simple_customer_id__predict_if_customer_purchases_next_quarter(testing_utils):
    """Tests the behavior of calling predict_if_customer_purchases_next_quarter() with an invalid customer_id.

        Expected: ValueError should be raised.
    """
    testing_utils.clean_directories()

    predictor = testing_utils.get_predictor()
    customer = 999999.99

    with pytest.raises(ValueError):
        predictor.predict_if_customer_purchases_next_quarter(customer)


@pytest.mark.dependency(depends=['test_proper_simple_customer_id__predict_if_customer_purchases_next_quarter'])
def test_wrong_simple_customer_id_ignore_if_not_exists__predict_if_customer_purchases_next_quarter(testing_utils):
    """Tests the behavior of calling predict_if_customer_purchases_next_quarter() with an invalid customer_id but ignore_if_not_exists is set to True.

        Expected: No error should be raised.
    """
    testing_utils.clean_directories()

    predictor = testing_utils.get_predictor()
    customer = 999999.99

    try:
        label = predictor.predict_if_customer_purchases_next_quarter(customer, ignore_if_not_exists=True)
        assert label is None, 'label should be None'

    except:
        pytest.fail('Should execute predict_if_customer_purchases_next_quarter() properly.')


@pytest.mark.dependency(depends=['test_proper_simple_customer_id__predict_if_customer_purchases_next_quarter'])
def test_wrong_list_customer_id__predict_if_customer_purchases_next_quarter(testing_utils):
    """Tests the behavior of calling predict_if_customer_purchases_next_quarter() with a list with invalid customer_id's.

        Expected: ValueError should be raised.
    """

    testing_utils.clean_directories()

    predictor = testing_utils.get_predictor()
    customers = [12352.0, 888888.88, 999999.99]

    with pytest.raises(ValueError):
        predictor.predict_if_customer_purchases_next_quarter(customers)


@pytest.mark.dependency(depends=['test_proper_simple_customer_id__predict_if_customer_purchases_next_quarter'])
def test_wrong_list_customer_id_ignore_if_not_exists__predict_if_customer_purchases_next_quarter(testing_utils):
    """Tests the behavior of calling predict_if_customer_purchases_next_quarter() with a list with invalid customer_id's but ignore_if_not_exists is set to True.

        Expected: No error should be raised.
    """

    testing_utils.clean_directories()

    predictor = testing_utils.get_predictor()
    customers = [12352.0, 888888.88, 999999.99]

    try:
        labels = predictor.predict_if_customer_purchases_next_quarter(customers, ignore_if_not_exists=True)
        assert isinstance(labels, pd.Series), 'Should be of type Series'
        assert len(labels) == len(customers), 'Should be of same length'
        assert labels.isnull().sum() == 2, 'Should be 2'

    except:
        pytest.fail('Should execute predict_if_customer_purchases_next_quarter() properly.')


@pytest.mark.dependency(depends=['test_proper_simple_customer_id__predict_if_customer_purchases_next_quarter'])
def test_wrongly_typed__predict_if_customer_purchases_next_quarter(testing_utils):
    """Tests the behavior of calling predict_if_customer_purchases_next_quarter() with wrongly typed parameters.

        Expected: TypeError should be raised.
    """
    testing_utils.clean_directories()

    predictor = testing_utils.get_predictor()

    with pytest.raises(TypeError):
        predictor.predict_if_customer_purchases_next_quarter('customer')


@pytest.mark.dependency(depends=['test_proper_simple_customer_id__predict_if_customer_purchases_next_quarter'])
def test_wrong_instance__predict_if_customer_purchases_next_quarter(testing_utils):
    """Tests the behavior of calling predict_if_customer_purchases_next_quarter() with an instance that is not initialized yet.
    
        Expected: NoTrainedModelError should be raised.
    """
    testing_utils.clean_directories()

    name = testing_utils.get_predictor_name()
    data = testing_utils.get_data()
    customer = testing_utils.get_customer()
    
    with pytest.raises(NoTrainedModelError):
        PurchasePredictor(name, data, True).predict_if_customer_purchases_next_quarter(customer)


@pytest.mark.dependency(depends=['test_proper__initialize_purchase_predictor'])
def test_proper__get_all_customers_purchasing_next_quarter(testing_utils):
    """Tests the proper usage of the get_all_customers_purchasing_next_quarter function.

        Expected: No error should be raised.
    """
    testing_utils.clean_directories()

    predictor = testing_utils.get_predictor()

    try:
        customers = predictor.get_all_customers_purchasing_next_quarter()
        assert isinstance(customers, list), 'Should be list'
        assert len(customers) > 0, 'Should be not empty'

    except:
        pytest.fail('Should execute get_all_customers_purchasing_next_quarter() properly.')


@pytest.mark.dependency(depends=['test_proper__get_all_customers_purchasing_next_quarter'])
def test_random_state__get_all_customers_purchasing_next_quarter(testing_utils):
    """Tests the behavior of calling get_all_customers_purchasing_next_quarter() with an instance using random_state.

        Expected: No error should be raised.
    """

    testing_utils.clean_directories()

    name = testing_utils.get_predictor_name()
    name_random = testing_utils.get_predictor_name() + 'random_state'
    
    data = testing_utils.get_data()
    test_size = testing_utils.get_test_size()
    hyperparamters = testing_utils.get_valid_settings()

    try:
        # check without set non_random
        predictor = PurchasePredictor(name, data, True)
        predictor.initialize_purchase_predictor(test_size, hyperparamters)
        customers_1 = predictor.get_all_customers_purchasing_next_quarter()

        testing_utils.clean_directories()

        # Overwrite old forecaster
        predictor = PurchasePredictor(name, data, True)
        predictor.initialize_purchase_predictor(test_size, hyperparamters)
        customers_2 = predictor.get_all_customers_purchasing_next_quarter()

        testing_utils.clean_directories()

        if len(customers_1) == len(customers_2):
            assert (customers_1 != customers_2), 'Predicted customers should be different'
        else:
            assert (len(customers_1) != len(customers_2)), 'Predicted customers should be different'

        # check with set random_state
        predictor = PurchasePredictor(name, data, True, random_state=1)
        predictor.initialize_purchase_predictor(test_size, hyperparamters)
        customers_1 = predictor.get_all_customers_purchasing_next_quarter()

        testing_utils.clean_directories(name_random)

        # Overwrite old forecaster
        predictor = PurchasePredictor(name, data, True, random_state=1)
        predictor.initialize_purchase_predictor(test_size, hyperparamters)
        customers_2 = predictor.get_all_customers_purchasing_next_quarter()

        testing_utils.clean_directories(name_random)

        assert (customers_1 == customers_2), 'Predicted customers should be equal'

    except:
        testing_utils.clean_directories()
        testing_utils.clean_directories(name_random)
        pytest.fail('Should execute get_all_customers_purchasing_next_quarter() properly with a set random state.')


@pytest.mark.dependency(depends=['test_proper__get_all_customers_purchasing_next_quarter'])
def test_wrong_instance__get_all_customers_purchasing_next_quarter(testing_utils):
    """Tests the behavior of calling get_all_customers_purchasing_next_quarter() with an instance that is not initialized yet.

        Expected: NoTrainedModelError should be raised.
    """

    testing_utils.clean_directories()

    name = testing_utils.get_predictor_name()
    data = testing_utils.get_data()

    with pytest.raises(NoTrainedModelError):
        PurchasePredictor(name, data, True).get_all_customers_purchasing_next_quarter()


@pytest.mark.dependency(depends=['test_proper__initialize_purchase_predictor'])
def test_proper__get_all_customers_not_purchasing_next_quarter(testing_utils):
    """Tests the proper usage of the get_all_customers_not_purchasing_next_quarter function.

        Expected: No error should be raised.
    """

    testing_utils.clean_directories()

    predictor = testing_utils.get_predictor()

    try:
        customers = predictor.get_all_customers_not_purchasing_next_quarter()
        assert isinstance(customers, list), 'Should be list'
        assert len(customers) > 0, 'Should be not empty'

    except:
        pytest.fail('Should execute get_all_customers_not_purchasing_next_quarter() properly.')


@pytest.mark.dependency(depends=['test_proper__get_all_customers_not_purchasing_next_quarter'])
def test_wrong_instance__get_all_customers_not_purchasing_next_quarter(testing_utils):
    """Tests the behavior of calling get_all_customers_not_purchasing_next_quarter() with an instance that is not initialized yet.

        Expected: NoTrainedModelError should be raised.
    """

    testing_utils.clean_directories()

    name = testing_utils.get_predictor_name()
    data = testing_utils.get_data()

    with pytest.raises(NoTrainedModelError):
        PurchasePredictor(name, data, True).get_all_customers_not_purchasing_next_quarter()


@pytest.fixture(scope="session", autouse=True)
def final_cleanup(request):
    """Cleans directories when all tests are finished."""
    request.addfinalizer(lambda: TestingUtils.clean_directories(mode='finalize'))
