
import pytest
from pathlib import Path
import pandas as pd

from path_definitions import PATHS
from modelcreator.datahandler.data_handler import DataHandler
from modelcreator.datahandler.preprocessor.sales_forecaster_preprocessor import SalesForecasterPreprocessor
from modelcreator.datahandler.preprocessor.product_recommender_preprocessor import ProductRecommenderPreprocessor
from modelcreator.datahandler.preprocessor.purchase_predictor_preprocessor import PurchasePredictorPreprocessor

class TestingUtils():

    @staticmethod
    def get_sales_forecast_handler():
        return DataHandler('online_retail_II.csv', SalesForecasterPreprocessor())

    @staticmethod
    def get_purchase_prediction_handler():
        return DataHandler('online_retail_II.csv', PurchasePredictorPreprocessor())

    @staticmethod
    def get_product_recommender_handler():
        return DataHandler('online_retail_II.csv', ProductRecommenderPreprocessor())

    @staticmethod
    def get_forecaster_data_name():
        return 'SalesForecasterData.feather'

    @staticmethod
    def get_predictor_data_name():
        return 'PurchasePredictorData.feather'

    @staticmethod
    def get_recommender_data_name():
        return 'ProductRecommenderData.feather'

    @staticmethod
    def clear_files(mode='all'):
        forecaster_data_path = PATHS['PROCESSED_DATA_DIR'] / TestingUtils.get_forecaster_data_name()
        predictor_data_path = PATHS['PROCESSED_DATA_DIR'] / TestingUtils.get_predictor_data_name()
        recommender_data_path = PATHS['PROCESSED_DATA_DIR'] / TestingUtils.get_recommender_data_name()
        
        if mode == 'all':
            forecaster_data_path.unlink(missing_ok=True)
            predictor_data_path.unlink(missing_ok=True)
            recommender_data_path.unlink(missing_ok=True)

        elif mode == 'forecaster':
            forecaster_data_path.unlink(missing_ok=True)
        
        elif mode == 'predictor':
            predictor_data_path.unlink(missing_ok=True)

        elif mode == 'recommender':
            recommender_data_path.unlink(missing_ok=True)

@pytest.fixture
def testing_utils():
    return TestingUtils


@pytest.mark.dependency
def test_proper__init():
    """Tests a proper initialization of a DataHandler instance.
        
        Expected: No error should be raised.
    """
    try:
        handler = DataHandler('online_retail_II.csv', SalesForecasterPreprocessor())
    except:
        pytest.fail('Should initialize DataHandler instance properly with SalesForecasterPreprocessor.')
    
    try:
        handler = DataHandler('online_retail_II.csv', PurchasePredictorPreprocessor())
    except:
        pytest.fail('Should initialize DataHandler instance properly with PurchasePredictorPreprocessor.')

    try:
        handler = DataHandler('online_retail_II.csv', ProductRecommenderPreprocessor())
    except:
        pytest.fail('Should initialize DataHandler instance properly with ProductRecommenderPreprocessor.')


@pytest.mark.dependency(depends=['test_proper__init'])
def test_empty__init():
    """Tests the behavior of intitializing a DataHandler with no parameters.

        Expected: TypeError should be raised.
    """

    with pytest.raises(TypeError) as err:
        handler = DataHandler()


@pytest.mark.dependency(depends=['test_proper__init'])
def test_wrongly_typed__init():
    """Tests the behavior of intitializing a DataHandler with wrongly typed parameters.

        Expected: TypeError should be raised in all cases.
    """
    
    with pytest.raises(TypeError) as err:
        handler = DataHandler('wrong', 'init')

    with pytest.raises(TypeError) as err:
        handler = DataHandler(1, SalesForecasterPreprocessor())

    with pytest.raises(TypeError) as err:
        handler = DataHandler('online_retail_II.csv', 'some preprocessor')


@pytest.mark.dependency(depends=['test_proper__init'])
def test_wrong_file__init():
    """Tests the behavior of intitializing a DataHandler with a wrong file.

        Expected: FileNotFoundError should be raised.
    """
    
    with pytest.raises(FileNotFoundError) as err:
        handler = DataHandler('not_existent_file.csv', SalesForecasterPreprocessor())


@pytest.mark.dependency(depends=['test_proper__init'])
def test_proper_return_path__generate_preprocessed_data(testing_utils):
    """Tests the proper usage of the generate_preprocessed_data function with return_path=True.
        
        Expected: No error should be raised.
    """

    testing_utils.clear_files()

    try:
        path = testing_utils.get_sales_forecast_handler().generate_preprocessed_data(return_path=True)
        assert isinstance(path, Path), 'Should be a Path object'
        assert path.exists(), 'The returned path should exist'
    except:
        pytest.fail('Should execute generate_preprocessed_data(return_path=True) with SalesForecasterPreprocessor properly.')
    
    try:
        path = testing_utils.get_purchase_prediction_handler().generate_preprocessed_data(return_path=True)
        assert isinstance(path, Path), 'Should be a Path object'
        assert path.exists(), 'The returned path should exist'
    except:
        pytest.fail('Should execute generate_preprocessed_data(return_path=True) with PurchasePredictorPreprocessor properly.')
        
    try:
        path = testing_utils.get_product_recommender_handler().generate_preprocessed_data(return_path=True)
        assert isinstance(path, Path), 'Should be a Path object'
        assert path.exists(), 'The returned path should exist'
    except:
        pytest.fail('Should execute generate_preprocessed_data(return_path=True) with ProductRecommenderPreprocessor properly.')


@pytest.mark.dependency(depends=['test_proper__init'])
def test_proper_return_dataset__generate_preprocessed_data(testing_utils):
    """Tests the proper usage of the generate_preprocessed_data function with return_dataset=True.
        
        Expected: No error should be raised.
    """
    testing_utils.clear_files()

    try:
        data = testing_utils.get_sales_forecast_handler().generate_preprocessed_data(return_dataset=True)
        assert isinstance(data, pd.DataFrame), 'Should be a DataFrame'
        assert not data.empty , 'DataFrame should not be empty'
    except:
        pytest.fail('Should execute generate_preprocessed_data(return_dataset=True) with SalesForecasterPreprocessor properly.')
    
    try:
        data = testing_utils.get_purchase_prediction_handler().generate_preprocessed_data(return_dataset=True)
        assert isinstance(data, pd.DataFrame), 'Should be a DataFrame'
        assert not data.empty, 'DataFrame should not be empty'
    except:
        pytest.fail('Should execute generate_preprocessed_data(return_dataset=True) with PurchasePredictorPreprocessor properly.')
        
    try:
        data = testing_utils.get_product_recommender_handler().generate_preprocessed_data(return_dataset=True)
        assert isinstance(data, pd.DataFrame), 'Should be a DataFrame'
        assert not data.empty, 'DataFrame should not be empty'
    except:
        pytest.fail('Should execute generate_preprocessed_data(return_dataset=True) with ProductRecommenderPreprocessor properly.')


@pytest.mark.dependency(depends=['test_proper__init'])
def test_proper_overwrite_if_exists__generate_preprocessed_data(testing_utils):
    """Tests the proper usage of the generate_preprocessed_data function with overwrite_if_exists = True.
        
        Expected: No error should be raised.
    """
    testing_utils.clear_files()

    try:
        return_value = testing_utils.get_sales_forecast_handler().generate_preprocessed_data(overwrite_if_exists = True)
        assert return_value is None, 'Should be None'
        
        data_name = 'SalesForecasterData.feather'
        data_path = PATHS['PROCESSED_DATA_DIR'] / data_name

        # call function again and check modification timestamp
        last_modification = data_path.stat().st_mtime
        testing_utils.get_sales_forecast_handler().generate_preprocessed_data(overwrite_if_exists = True)
        new_modification = data_path.stat().st_mtime
        assert last_modification < new_modification, 'Modification date of file should have changed'

    except:
        pytest.fail('Should execute generate_preprocessed_data(overwrite_if_exists = True) with SalesForecasterPreprocessor properly.')
    
    try:
        return_value = testing_utils.get_purchase_prediction_handler().generate_preprocessed_data(overwrite_if_exists = True)
        assert return_value is None, 'Should be None'
        
        data_name = 'PurchasePredictorData.feather'
        data_path = PATHS['PROCESSED_DATA_DIR'] / data_name
        
        # call function again and check modification timestamp
        last_modification = data_path.stat().st_mtime
        testing_utils.get_purchase_prediction_handler().generate_preprocessed_data(overwrite_if_exists = True)
        new_modification = data_path.stat().st_mtime
        assert last_modification < new_modification, 'Modification date of file should have changed'

    except:
        pytest.fail('Should execute generate_preprocessed_data(overwrite_if_exists = True) with PurchasePredictorPreprocessor properly.')
        
    try:
        return_value = testing_utils.get_product_recommender_handler().generate_preprocessed_data(overwrite_if_exists = True)
        assert return_value is None, 'Should be None'
        
        data_name = 'ProductRecommenderData.feather'
        data_path = PATHS['PROCESSED_DATA_DIR'] / data_name

         # call function again and check modification timestamp
        last_modification = data_path.stat().st_mtime
        testing_utils.get_product_recommender_handler().generate_preprocessed_data(overwrite_if_exists = True)
        new_modification = data_path.stat().st_mtime
        assert last_modification < new_modification, 'Modification date of file should have changed'

    except:
        pytest.fail('Should execute generate_preprocessed_data(overwrite_if_exists = True) with ProductRecommenderPreprocessor properly.')


@pytest.mark.dependency(depends=[
                                'test_proper_return_path__generate_preprocessed_data',
                                'test_proper_return_dataset__generate_preprocessed_data',
                                'test_proper_overwrite_if_exists__generate_preprocessed_data'
                                ])
def test_proper_all_true__generate_preprocessed_data(testing_utils):
    """Tests the proper usage of the generate_preprocessed_data function with all parameters set to true.
        
        Expected: No error should be raised.
    """
    testing_utils.clear_files()

    try:
        return_value = testing_utils.get_sales_forecast_handler().generate_preprocessed_data(True, True, True)
        assert isinstance(return_value, tuple), 'Should be a tuple'
        assert len(return_value) == 2, 'Should be a tuple with a length of 2'
        assert isinstance(return_value[0], Path), 'Should be a Path object'
        assert isinstance(return_value[1], pd.DataFrame), 'Should be a DataFrame'
        
        data_name = 'SalesForecasterData.feather'
        data_path = PATHS['PROCESSED_DATA_DIR'] / data_name

        # call function again and check modification timestamp
        last_modification = data_path.stat().st_mtime
        testing_utils.get_sales_forecast_handler().generate_preprocessed_data(True, True, True)
        new_modification = data_path.stat().st_mtime
        assert last_modification < new_modification, 'Modification date of file should have changed'

    except:
        pytest.fail('Should execute generate_preprocessed_data(True, True, True) with SalesForecasterPreprocessor properly.')
    
    try:
        return_value = testing_utils.get_purchase_prediction_handler().generate_preprocessed_data(True, True, True)
        assert isinstance(return_value, tuple), 'Should be a tuple'
        assert len(return_value) == 2, 'Should be a tuple with a length of 2'
        assert isinstance(return_value[0], Path), 'Should be a Path object'
        assert isinstance(return_value[1], pd.DataFrame), 'Should be a DataFrame'
        
        data_name = 'PurchasePredictorData.feather'
        data_path = PATHS['PROCESSED_DATA_DIR'] / data_name
        
        # call function again and check modification timestamp
        last_modification = data_path.stat().st_mtime
        testing_utils.get_purchase_prediction_handler().generate_preprocessed_data(True, True, True)
        new_modification = data_path.stat().st_mtime
        assert last_modification < new_modification, 'Modification date of file should have changed'

    except:
        pytest.fail('Should execute generate_preprocessed_data(True, True, True) with PurchasePredictorPreprocessor properly.')
        
    try:
        return_value = testing_utils.get_product_recommender_handler().generate_preprocessed_data(True, True, True)
        assert isinstance(return_value, tuple), 'Should be a tuple'
        assert len(return_value) == 2, 'Should be a tuple with a length of 2'
        assert isinstance(return_value[0], Path), 'Should be a Path object'
        assert isinstance(return_value[1], pd.DataFrame), 'Should be a DataFrame'
        
        data_name = 'ProductRecommenderData.feather'
        data_path = PATHS['PROCESSED_DATA_DIR'] / data_name

         # call function again and check modification timestamp
        last_modification = data_path.stat().st_mtime
        testing_utils.get_product_recommender_handler().generate_preprocessed_data(True, True, True)
        new_modification = data_path.stat().st_mtime
        assert last_modification < new_modification, 'Modification date of file should have changed'
        
    except:
        pytest.fail('Should execute generate_preprocessed_data(True, True, True) with ProductRecommenderPreprocessor properly.')


@pytest.mark.dependency(depends=['test_proper_all_true__generate_preprocessed_data'])
def test_wrongly_typed__generate_preprocessed_data(testing_utils):
    """Tests the behavior of calling generate_preprocessed_data() with wrongly typed parameters.

        Expected: TypeError should be raised in all cases.
    """
    testing_utils.clear_files()

    with pytest.raises(TypeError) as err:
        testing_utils.get_sales_forecast_handler().generate_preprocessed_data(1, False, False)

    with pytest.raises(TypeError) as err:
        testing_utils.get_purchase_prediction_handler().generate_preprocessed_data(False, 1, False)
    
    with pytest.raises(TypeError) as err:
        testing_utils.get_product_recommender_handler().generate_preprocessed_data(False, False, 1)


@pytest.mark.dependency(depends=['test_proper_all_true__generate_preprocessed_data'])
def test_proper__get_preprocessed_data(testing_utils):
    """Tests the proper usage of the get_preprocessed_data function.

        Expected: No error should be raised.
    """
    testing_utils.clear_files()
    testing_utils.get_sales_forecast_handler().generate_preprocessed_data(False, False)

    try:
        data = testing_utils.get_sales_forecast_handler().get_preprocessed_data()
        assert isinstance(data, pd.DataFrame), 'Should be a DataFrame'
        assert not data.empty, 'DataFrame should not be empty'

    except:
        pytest.fail('Should execute get_preprocessed_data() with SalesForecasterPreprocessor properly.')

    testing_utils.get_purchase_prediction_handler().generate_preprocessed_data(False, False)

    try:
        data = testing_utils.get_purchase_prediction_handler().get_preprocessed_data()
        assert isinstance(data, pd.DataFrame) == True, 'Should be a DataFrame'
        assert not data.empty, 'DataFrame should not be empty'

    except:
        pytest.fail('Should execute get_preprocessed_data() with PurchasePredictorPreprocessor properly.')
    
    testing_utils.get_product_recommender_handler().generate_preprocessed_data(False, False)

    try:
        data = testing_utils.get_product_recommender_handler().get_preprocessed_data()
        assert isinstance(data, pd.DataFrame) == True, 'Should be a DataFrame'
        assert not data.empty, 'DataFrame should not be empty'

    except:
        pytest.fail('Should execute get_preprocessed_data() with ProductRecommenderPreprocessor properly.')


@pytest.mark.dependency(depends=['test_proper__get_preprocessed_data'])
def test_file_does_not_exist__get_preprocessed_data(testing_utils):
    """Tests the behavior of the get_preprocessed_data function if a file does not exist.

        Expected: No error should be raised.
    """
    testing_utils.clear_files()

    with pytest.raises(FileNotFoundError) as err:
        testing_utils.get_sales_forecast_handler().get_preprocessed_data()

    with pytest.raises(FileNotFoundError) as err:
        testing_utils.get_purchase_prediction_handler().get_preprocessed_data()

    with pytest.raises(FileNotFoundError) as err:
        testing_utils.get_product_recommender_handler().get_preprocessed_data()


@pytest.fixture(scope="session", autouse=True)
def final_cleanup(request):
    """Removes files when all tests are finished."""
    request.addfinalizer(TestingUtils.clear_files)
