import pytest
import pandas as pd

from path_definitions import PATHS
from modelcreator.datahandler.data_handler import DataHandler
from modelcreator.datahandler.preprocessor.product_recommender_preprocessor import ProductRecommenderPreprocessor
from modelcreator.productrecommender.product_recommender import ProductRecommender
from modelcreator.exception import CorruptRecommenderError, NoTrainedModelError, ModelNotFoundError


class TestingUtils():

    @staticmethod
    def get_recommender_name():
        return 'pytest_test_product_recommender'

    @staticmethod
    def get_number_of_recommendations():
        return 15

    @staticmethod
    def get_product():
        return '79303A'

    @staticmethod
    def get_product_list():
        return ['85150', '79303A', '47579']

    @staticmethod
    def get_customer():
        return 12352.0

    @staticmethod 
    def get_customer_list():
        return [12347.0, 12352.0, 12356.0]

    @staticmethod
    def get_data():
        handler = DataHandler('online_retail_II.csv', ProductRecommenderPreprocessor())
        data = handler.generate_preprocessed_data(False, True)
        return data

    @staticmethod
    def get_loaded_recommender():
        recommender = ProductRecommender(TestingUtils.get_recommender_name(), True)
        return recommender

    @staticmethod
    def clean_directories(name='pytest_test_product_recommender', mode='all'):

        eval_dir_path = PATHS['EVALUATIONS_DIR'] / 'ProductRecommender' / name
        eval_file_path = eval_dir_path / (name+'_evaluation.txt')
        
        model_dir_path = PATHS['MODELS_DIR'] / 'ProductRecommender' / name
        cir_matrix_file_path = model_dir_path / 'customer_item_recommendation_matrix.feather'
        in_matrix_file_path = model_dir_path / 'item_neighborhood_matrix.feather'
        
        if mode == 'all':
            eval_file_path.unlink(missing_ok=True)

            if eval_dir_path.exists():
                eval_dir_path.rmdir()

            cir_matrix_file_path.unlink(missing_ok=True)
            in_matrix_file_path.unlink(missing_ok=True)

            if model_dir_path.exists():
                model_dir_path.rmdir()

        elif mode == 'model_files':
            cir_matrix_file_path.unlink(missing_ok=True)
            in_matrix_file_path.unlink(missing_ok=True)

        elif mode == 'finalize':
            eval_file_path.unlink(missing_ok=True)

            if eval_dir_path.exists():
                eval_dir_path.rmdir()

            cir_matrix_file_path.unlink(missing_ok=True)
            in_matrix_file_path.unlink(missing_ok=True)

            if model_dir_path.exists():
                model_dir_path.rmdir()

            recommender_data_path = PATHS['PROCESSED_DATA_DIR'] / 'ProductRecommenderData.feather'
            recommender_data_path.unlink(missing_ok=True)
    
@pytest.fixture
def testing_utils():
    return TestingUtils


@pytest.mark.dependency
def test_proper__init():
    """Tests a proper initialization of a ProductRecommender instance.
        
        Expected: No error should be raised.
    """

    try:
        recommender = ProductRecommender('pytest_test_product_recommender')
    
    except:
        pytest.fail('Should initialize ProductRecommender instance properly.')


@pytest.mark.dependency(depends=['test_proper__init'])
def test_empty__init():
    """Tests the behavior of intitializing a ProductRecommender with no parameters.

        Expected: TypeError should be raised.
    """

    with pytest.raises(TypeError):
        recommender = ProductRecommender()


@pytest.mark.dependency(depends=['test_proper__init'])
def test_wrongly_typed__init():
    """Tests the behavior of intitializing a ProductRecommender with wrongly typed parameters.

        Expected: TypeError should be raised in all cases.
    """
    
    with pytest.raises(TypeError):
        recommender = ProductRecommender(1)

    with pytest.raises(TypeError):
        recommender = ProductRecommender('pytest_test_product_recommender', 'True')

    with pytest.raises(TypeError):
        recommender = ProductRecommender('pytest_test_product_recommender', True, 10)


@pytest.mark.dependency(depends=['test_proper__init'])
def test_wrongly_named__init():
    """Tests the behavior of intitializing a ProductRecommender with not allowed names.

        Expected: ValueError should be raised in all cases.
    """

    with pytest.raises(ValueError):
        recommender = ProductRecommender('')

    with pytest.raises(ValueError):
        recommender = ProductRecommender('no_special_characters?!')

    with pytest.raises(ValueError):
        recommender = ProductRecommender('no spaces')


@pytest.mark.dependency(depends=['test_proper__init'])
def test_proper__initialize_recommender(testing_utils):
    """Tests the proper usage of the initialize_recommender function.

        Expected:  No error should be raised.
    """
    recommender_data_path = PATHS['PROCESSED_DATA_DIR'] / 'ProductRecommenderData.feather'
    recommender_data_path.unlink(missing_ok=True)

    name = testing_utils.get_recommender_name()

    eval_dir_path = PATHS['EVALUATIONS_DIR'] / 'ProductRecommender' / name
    eval_file_path = eval_dir_path / (name+'_evaluation.txt')
    model_dir_path = PATHS['MODELS_DIR'] / 'ProductRecommender' / name
    cir_matrix_file_path = model_dir_path / 'customer_item_recommendation_matrix.feather'
    in_matrix_file_path = model_dir_path / 'item_neighborhood_matrix.feather'

    data = testing_utils.get_data()
    number_of_recommendations = testing_utils.get_number_of_recommendations()
    recommender = ProductRecommender(name)

    try:
        recommender.initialize_recommender(data, number_of_recommendations)
        
        assert eval_dir_path.exists(), 'An evaluation folder for the product recommender should have been created'
        assert eval_file_path.exists(), 'An evaluation file should have been created'
        assert model_dir_path.exists(), 'A model folder for the product recommender should have been created'
        assert cir_matrix_file_path.exists(), 'A resulting customer-item-recommendation matrix should have been stored on disk'
        assert in_matrix_file_path.exists(), 'A resulting item-neighborhood matrix should have been stored on disk'

    except:
        pytest.fail('Should execute initialize_recommender() properly.')


@pytest.mark.dependency(depends=['test_proper__initialize_recommender'])
def test_wrongly_typed__initialize_recommender(testing_utils):
    """Tests the behavior of calling initialize_recommender() with wrongly typed parameters.

        Expected: TypeError should be raised in all cases.
    """
    
    name = testing_utils.get_recommender_name()
    data = testing_utils.get_data()
    recommender = ProductRecommender(name)

    with pytest.raises(TypeError):
        recommender.initialize_recommender([1,2,3], 5)

    with pytest.raises(TypeError):
        recommender.initialize_recommender(data, '5')


@pytest.mark.dependency(depends=['test_proper__initialize_recommender'])
def test_wrong_number_of_recommendations__initialize_recommender(testing_utils):
    """Tests the behavior of calling initialize_recommender() with a wrong number of recommendations.

        Expected: ValueError should be raised in all cases.
    """

    name = testing_utils.get_recommender_name()
    data = testing_utils.get_data()
    recommender = ProductRecommender(name)

    with pytest.raises(ValueError):
        recommender.initialize_recommender(data, -5)

    with pytest.raises(ValueError):
        recommender.initialize_recommender(data, 0)
    
    with pytest.raises(ValueError):
        recommender.initialize_recommender(data, data.shape[1]+1)


@pytest.mark.dependency(depends=['test_proper__initialize_recommender'])
def test_load_existing_model__init(testing_utils):
    """Tests the behavior of properly intitializing a ProductRecommender with load_existing_model = True.

        Expected: No error should be raised.
    """
    name = testing_utils.get_recommender_name()
    try:
        recommender = ProductRecommender(name, load_existing_model = True)
    except:
        pytest.fail('Should execute initialize_recommender(load_existing_model = True) properly.')


@pytest.mark.dependency(depends=['test_load_existing_model__init'])
def test_load_not_existing_model__init(testing_utils):
    """Tests the behavior of intitializing a ProductRecommender with load_existing_model = True if the model is not existing.

        Expected: ModelNotFoundError should be raised.
    """
    name = testing_utils.get_recommender_name() + '_new'
    with pytest.raises(ModelNotFoundError):
        recommender = ProductRecommender(name, load_existing_model = True)


@pytest.mark.dependency(depends=['test_load_existing_model__init'])
def test_load_existing_model_without_files__init(testing_utils):
    """Tests the behavior of intitializing a ProductRecommender with load_existing_model = True, but without model files being present.

        Expected: CorruptRecommenderError should be raised.
    """

    name = testing_utils.get_recommender_name() + '_new'
    recommender = ProductRecommender(name)

    data = testing_utils.get_data()
    number_of_recommendations = testing_utils.get_number_of_recommendations()
    recommender.initialize_recommender(data, number_of_recommendations)

    testing_utils.clean_directories(name, 'model_files')
    
    with pytest.raises(CorruptRecommenderError):
        recommender = ProductRecommender(name, load_existing_model = True)

    testing_utils.clean_directories(name, 'all')


@pytest.mark.dependency(depends=['test_load_existing_model__init'])
def test_proper_single_product__get_top_recommended_products_for_product(testing_utils):
    """Tests the proper usage of the get_top_recommended_products_for_product function for a single stock_code.

        Expected:  No error should be raised.
    """
    recommender = testing_utils.get_loaded_recommender()
    product = testing_utils.get_product()
    
    try:
        recommendations = recommender.get_top_recommended_products_for_product(product)
        assert isinstance(recommendations, pd.Series), 'Should be of type Series'
        assert recommendations.shape[0] == testing_utils.get_number_of_recommendations(), f'Number of rows should be {testing_utils.get_number_of_recommendations()}'
        
    except:
        pytest.fail(f'Should execute get_top_recommended_products_for_product(\'{product}\') properly.')


@pytest.mark.dependency(depends=['test_load_existing_model__init'])
def test_proper_product_list__get_top_recommended_products_for_product(testing_utils):
    """Tests the proper usage of the get_top_recommended_products_for_product function for a list of stock_codes.

        Expected:  No error should be raised.
    """

    recommender = testing_utils.get_loaded_recommender()
    products = testing_utils.get_product_list()

    try:
        recommendations = recommender.get_top_recommended_products_for_product(products)
        assert isinstance(recommendations, pd.DataFrame), 'Should be of type DataFrame'
        assert recommendations.shape[0] == len(products), f'Number of rows should be {len(products)}'
        assert recommendations.shape[1] == testing_utils.get_number_of_recommendations(), f'Number of columns should be {testing_utils.get_number_of_recommendations()}'
        
    except:
        pytest.fail(f'Should execute get_top_recommended_products_for_product({products}) properly.')


@pytest.mark.dependency(depends=['test_load_existing_model__init'])
def test_wrongly_typed__get_top_recommended_products_for_product(testing_utils):
    """Tests the behavior of calling get_top_recommended_products_for_product() with wrongly typed parameters.

        Expected:  TypeError should be raised.
    """
    recommender = testing_utils.get_loaded_recommender()

    with pytest.raises(TypeError):
        recommender.get_top_recommended_products_for_product(1)


@pytest.mark.dependency(depends=['test_proper_single_product__get_top_recommended_products_for_product'])
def test_wrong_single_product__get_top_recommended_products_for_product(testing_utils):
    """Tests the behavior of calling get_top_recommended_products_for_product() with a wrong stock_code.

        Expected:  ValueError should be raised.
    """

    recommender = testing_utils.get_loaded_recommender()

    with pytest.raises(ValueError):
        recommender.get_top_recommended_products_for_product('wrong_product')


@pytest.mark.dependency(depends=['test_proper_product_list__get_top_recommended_products_for_product'])
def test_wrong_product_list__get_top_recommended_products_for_product(testing_utils):
    """Tests the behavior of calling get_top_recommended_products_for_product() with a list containing wrong stock codes.

        Expected:  ValueError should be raised.
    """
    recommender = testing_utils.get_loaded_recommender()

    with pytest.raises(ValueError):
        recommender.get_top_recommended_products_for_product(['wrong_1', 'wrong_2', 'wrong_3'])


@pytest.mark.dependency(depends=['test_proper_single_product__get_top_recommended_products_for_product'])
def test_wrong_single_product_ignore_if_not_exists__get_top_recommended_products_for_product(testing_utils):
    """Tests the behavior of calling get_top_recommended_products_for_product() with a wrong stock_code but ignore_if_not_exist = True.

        Expected:  No error should be raised.
    """
    recommender = testing_utils.get_loaded_recommender()

    try:
        recommendations = recommender.get_top_recommended_products_for_product('wrong_product', ignore_if_not_exists=True)
        assert recommendations is None, 'Should be none'
    except:
        pytest.fail('Should execute get_top_recommended_products_for_product(\'wrong_product\', ignore_if_not_exists=True) properly.')


@pytest.mark.dependency(depends=['test_proper_product_list__get_top_recommended_products_for_product'])
def test_wrong_product_list_ignore_if_not_exist__get_top_recommended_products_for_product(testing_utils):
    """Tests the proper usage of the get_top_recommended_products_for_product with a list containing wrong stock codes but ignore_if_not_exist = True.

        Expected:  No error should be raised.
    """

    recommender = testing_utils.get_loaded_recommender()
    wrong_products = ['85150', '79303A', 'wrong_1']

    try:
        recommendations = recommender.get_top_recommended_products_for_product(wrong_products, ignore_if_not_exists=True)
        assert isinstance(recommendations, pd.DataFrame), 'Should be of type DataFrame'
        assert recommendations.shape[0] == 3, 'Number of rows should be 3'
        assert recommendations.shape[1] == testing_utils.get_number_of_recommendations(), f'Number of columns should be {testing_utils.get_number_of_recommendations()}'
        assert recommendations.isnull().sum().sum() == testing_utils.get_number_of_recommendations(), f'There should be {testing_utils.get_number_of_recommendations()} nan values'

    except:
        pytest.fail(f'Should execute get_top_recommended_products_for_product({wrong_products}, ignore_if_not_exists=True) properly.')


@pytest.mark.dependency(depends=['test_load_existing_model__init'])
def test_proper_single_customer__get_top_recommended_products_for_customer(testing_utils):
    """Tests the proper usage of the get_top_recommended_products_for_customer function for a single customer.

        Expected:  No error should be raised.
    """

    recommender = testing_utils.get_loaded_recommender()
    customer = testing_utils.get_customer()
    
    try:
        recommendations = recommender.get_top_recommended_products_for_customer(customer)
        assert isinstance(recommendations, pd.Series), 'Should be of type Series'
        assert recommendations.shape[0] == testing_utils.get_number_of_recommendations(), f'Should be {testing_utils.get_number_of_recommendations()}'
        
    except:
        pytest.fail(f'Should execute get_top_recommended_products_for_customer({customer}) properly.')


@pytest.mark.dependency(depends=['test_load_existing_model__init'])
def test_proper_customer_list__get_top_recommended_products_for_customer(testing_utils):
    """Tests the proper usage of the get_top_recommended_products_for_customer function for a list of customers.

        Expected:  No error should be raised.
    """

    recommender = testing_utils.get_loaded_recommender()
    customers = testing_utils.get_customer_list()

    try:
        recommendations = recommender.get_top_recommended_products_for_customer(customers)
        assert isinstance(recommendations, pd.DataFrame), 'Should be of type DataFrame'
        assert recommendations.shape[0] == len(customers), f'Number of rows should be {len(customers)}'
        assert recommendations.shape[1] == testing_utils.get_number_of_recommendations(), f'Number of columns should be {testing_utils.get_number_of_recommendations()}'
        
    except:
        pytest.fail(f'Should execute get_top_recommended_products_for_product({customers}) properly.')


@pytest.mark.dependency(depends=['test_load_existing_model__init'])
def test_wrongly_typed__get_top_recommended_products_for_customer(testing_utils):
    """Tests the behavior of calling get_top_recommended_products_for_customer() with wrongly typed parameters.

        Expected:  TypeError should be raised.
    """
    recommender = testing_utils.get_loaded_recommender()

    with pytest.raises(TypeError):
        recommender.get_top_recommended_products_for_customer(1)


@pytest.mark.dependency(depends=['test_proper_single_customer__get_top_recommended_products_for_customer'])
def test_wrong_single_customer__get_top_recommended_products_for_customer(testing_utils):
    """Tests the behavior of calling get_top_recommended_products_for_customer() with a wrong customer_id.

        Expected:  ValueError should be raised.
    """
    recommender = testing_utils.get_loaded_recommender()

    with pytest.raises(ValueError):
        recommender.get_top_recommended_products_for_customer(999999.99)


@pytest.mark.dependency(depends=['test_proper_customer_list__get_top_recommended_products_for_customer'])
def test_wrong_customer_list__get_top_recommended_products_for_customer(testing_utils):
    """Tests the behavior of calling get_top_recommended_products_for_customer() with a list containing wrong customer IDs.

        Expected:  ValueError should be raised.
    """
    recommender = testing_utils.get_loaded_recommender()

    with pytest.raises(ValueError):
        recommender.get_top_recommended_products_for_customer([777777.77, 888888.88, 999999.99])


@pytest.mark.dependency(depends=['test_proper_single_customer__get_top_recommended_products_for_customer'])
def test_wrong_single_customer_ignore_if_not_exists__get_top_recommended_products_for_customer(testing_utils):
    """Tests the behavior of calling get_top_recommended_products_for_customer() with a wrong customer_id but ignore_if_not_exist = True.

        Expected:  No error should be raised.
    """
    recommender = testing_utils.get_loaded_recommender()

    try:
        recommendations = recommender.get_top_recommended_products_for_customer(99999.99, ignore_if_not_exists=True)
        assert recommendations is None, 'Should be none'

    except:
        pytest.fail('Should execute get_top_recommended_products_for_customer(99999.99, ignore_if_not_exists=True) properly.')


@pytest.mark.dependency(depends=['test_proper_customer_list__get_top_recommended_products_for_customer'])
def test_wrong_customer_list_ignore_if_not_exist__get_top_recommended_products_for_customer(testing_utils):
    """Tests the proper usage of the get_top_recommended_products_for_customer with a list containing wrong customer IDs but ignore_if_not_exist = True.

        Expected:  No error should be raised.
    """

    recommender = testing_utils.get_loaded_recommender()
    wrong_products = [12347.0, 12352.0, 99999.99]

    try:
        recommendations = recommender.get_top_recommended_products_for_customer(wrong_products, ignore_if_not_exists=True)
        assert recommendations.shape[0] == 3, 'Number of rows should be 3'
        assert recommendations.shape[1] == testing_utils.get_number_of_recommendations(), f'Number of columns should be {testing_utils.get_number_of_recommendations()}'
        assert recommendations.isnull().sum().sum() == testing_utils.get_number_of_recommendations(), f'There should be {testing_utils.get_number_of_recommendations()} nan values'

    except:
        pytest.fail(f'Should execute get_top_recommended_products_for_customer({wrong_products}, ignore_if_not_exists=True) properly.')


@pytest.mark.dependency(depends=[
                                'test_proper_single_product__get_top_recommended_products_for_product',
                                'test_proper_single_customer__get_top_recommended_products_for_customer'
                                ])
def test_not_initialized_recommender__get_top_recommended_products_for_product_or_customer(testing_utils):
    
    name = testing_utils.get_recommender_name()
    recommender = ProductRecommender(name)

    with pytest.raises(NoTrainedModelError):
        recommender.get_top_recommended_products_for_product('79303A')

    with pytest.raises(NoTrainedModelError):
        recommender.get_top_recommended_products_for_customer(12352.0)


@pytest.fixture(scope="session", autouse=True)
def final_cleanup(request):
    """Cleans directories when all tests are finished."""
    request.addfinalizer(lambda: TestingUtils.clean_directories(mode='finalize'))
