import os
import warnings

# Disable warnings and tensorflow output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from modelcreator.datahandler.data_handler import DataHandler
from modelcreator.datahandler.preprocessor import ProductRecommenderPreprocessor
from modelcreator.datahandler.preprocessor import PurchasePredictorPreprocessor
from modelcreator.datahandler.preprocessor import SalesForecasterPreprocessor
from modelcreator.salesforecaster import ARIMAForecaster
from modelcreator.salesforecaster import LSTMForecaster
from modelcreator.purchasepredictor import PurchasePredictor
from modelcreator.productrecommender import ProductRecommender
from modelcreator.logger import setup_info_logger

"""
This module provides a ``create()`` function, which creates predefined models.

"""

def create():
    """Creates models with different configurations to compare them and to pick a final model for the tasks."""

    ####################################################################################################
    # Model Configuarations                                                                            #
    ####################################################################################################


    LR_PURCHASE_PREDICTOR_SETTINGS = [
        {
            'CLF_NAME': 'LogisticRegression',
            'HYPERPARAMS': [
                {
                    'penalty': ['l2'],
                    'C': [1, 0.75, 0.5, 0.25, 0.1, 0.05],
                    'class_weight': ['balanced', None]
                },
                {
                    'penalty': ['none'],
                    'class_weight': ['balanced', None]
                }
            ]
        }
    ]

    RF_PURCHASE_PREDICTOR_SETTINGS = [
        {
            'CLF_NAME': 'RandomForestClassifier',
            'HYPERPARAMS': 
            {
                'n_estimators': [100, 200, 300],
                'criterion': ['gini', 'entropy'],
                'max_depth': [5, 7, 10, 15, None],
                'min_samples_split': [2, 4, 8, 16, 32, 64, 128, 256],
                'class_weight': ['balanced', 'balanced_subsample', None]
            }
        }
    ]

    ARIMA_SETTINGS = {
            'p': [0, 1, 2],
            'd': [1],
            'q': [0, 1, 2],
            'P': [0, 1, 2],
            'D': [0],
            'Q': [0],
            'm': [0, 52]
        }

    LSTM_SINGLE_SMALL_SETTINGS = {
            'neurons_layer_1': [1, 2, 4],
            'neurons_layer_2' : [0],
            'epochs': [50, 100, 150, 200],
            'learning_rate': [0.001],
            'batch_size': [1]
    }

    LSTM_SINGLE_MEDIUM_SETTINGS = {
            'neurons_layer_1': [8, 16, 32],
            'neurons_layer_2' : [0],
            'epochs': [50, 100, 150, 200],
            'learning_rate': [0.001],
            'batch_size': [1]
    }

    LSTM_SINGLE_LARGE_SETTINGS = {
            'neurons_layer_1': [64, 128, 256],
            'neurons_layer_2' : [0],
            'epochs': [50, 100, 150, 200],
            'learning_rate': [0.001],
            'batch_size': [1]
    }

    LSTM_STACKED_SMALL_SETTINGS = {
            'neurons_layer_1': [1, 2, 4],
            'neurons_layer_2' : [1, 2, 4],
            'recurrent_dropout': [0, 0.1],
            'epochs': [50, 100, 150, 200],
            'learning_rate': [0.001],
            'batch_size': [1]
    }

    LSTM_STACKED_MEDIUM_SETTINGS = {
            'neurons_layer_1': [8, 16, 32],
            'neurons_layer_2' : [8, 16, 32],
            'recurrent_dropout': [0, 0.1],
            'epochs': [50, 100, 150, 200],
            'learning_rate': [0.001],
            'batch_size': [1]
    }

    LSTM_STACKED_LARGE_SETTINGS = {
            'neurons_layer_1': [64, 128, 256],
            'neurons_layer_2' : [64, 128, 256],
            'recurrent_dropout': [0.1, 0.2],
            'epochs': [50, 100, 150, 200],
            'learning_rate': [0.001],
            'batch_size': [1]
    }


    ####################################################################################################
    # Data Preprocessing                                                                               #
    ####################################################################################################

    # Initialize DataHandler with different preprocessing strategies
    raw_data_name = 'online_retail_II.csv'
    sales_forecasting_handler = DataHandler(raw_data_name, SalesForecasterPreprocessor())
    purchase_prediction_handler = DataHandler(raw_data_name, PurchasePredictorPreprocessor())
    product_recommendation_handler = DataHandler(raw_data_name, ProductRecommenderPreprocessor())

    # Generate the preprocessed data
    sales_forecasting_data = sales_forecasting_handler.generate_preprocessed_data(return_dataset=True)
    purchase_prediction_data = purchase_prediction_handler.generate_preprocessed_data(return_dataset=True)
    product_recommendation_data = product_recommendation_handler.generate_preprocessed_data(return_dataset=True)
    
    # Setup a logger
    logger = setup_info_logger() 

    # Set a random_state for reproducible results
    random_state = 95

    ####################################################################################################
    # Create Sales Forecasting Models                                                                  #
    ####################################################################################################
    test_size = 8

    # Arima model
    arima_1 = ARIMAForecaster(name='arima_model', time_series=sales_forecasting_data, apply_box_cox=True, load_existing_model=False, logger=logger)
    arima_1.initialize_forecaster(test_size=test_size, hyperparameters=ARIMA_SETTINGS)

    # Single Layer LSTM
    lstm_single_s = LSTMForecaster(name='single_small_model', time_series=sales_forecasting_data, load_existing_model=False, logger=logger, random_state=random_state)
    lstm_single_s.initialize_forecaster(test_size=test_size, hyperparameters=LSTM_SINGLE_SMALL_SETTINGS, lstm_timesteps=4)

    lstm_single_m = LSTMForecaster(name='single_medium_model', time_series=sales_forecasting_data, load_existing_model=False, logger=logger, random_state=random_state)
    lstm_single_m.initialize_forecaster(test_size = test_size, hyperparameters = LSTM_SINGLE_MEDIUM_SETTINGS, lstm_timesteps=4)

    lstm_single_l = LSTMForecaster(name='single_large_model', time_series=sales_forecasting_data, load_existing_model=False, logger=logger, random_state=random_state)
    lstm_single_l.initialize_forecaster(test_size = test_size, hyperparameters = LSTM_SINGLE_LARGE_SETTINGS, lstm_timesteps=4)

    # Stacked LSTM

    lstm_stacked_s = LSTMForecaster(name='stacked_small_model', time_series=sales_forecasting_data, load_existing_model=False, logger=logger, random_state=random_state)
    lstm_stacked_s.initialize_forecaster(test_size=test_size, hyperparameters=LSTM_STACKED_SMALL_SETTINGS, lstm_timesteps=4)

    lstm_stacked_m = LSTMForecaster(name='stacked_medium_model', time_series=sales_forecasting_data, load_existing_model=False, logger=logger, random_state=random_state)
    lstm_stacked_m.initialize_forecaster(test_size = test_size, hyperparameters = LSTM_STACKED_MEDIUM_SETTINGS, lstm_timesteps=4)

    lstm_stacked_l = LSTMForecaster(name='stacked_large_model', time_series=sales_forecasting_data, load_existing_model=False, logger=logger, random_state=random_state)
    lstm_stacked_l.initialize_forecaster(test_size = test_size, hyperparameters = LSTM_STACKED_LARGE_SETTINGS, lstm_timesteps=4)
    

    ####################################################################################################
    # Create Purchase Prediction Model                                                                 #
    ####################################################################################################

    # Create Purchase Prediction Model
    test_size = 0.2
    
    #Try logistic regression

    # Best Purchase predictor in terms of F1 score
    lr_fbeta_1 = PurchasePredictor(name='lr_model', data=purchase_prediction_data, apply_standardization=True, load_existing_model=False, logger=logger, random_state=random_state)
    lr_fbeta_1.initialize_purchase_predictor(test_size=test_size, settings=LR_PURCHASE_PREDICTOR_SETTINGS, beta=1)

    # Best Purchase predictor in terms of F0.5 score
    lr_fbeta_0_5 = PurchasePredictor(name='more_precise_lr_model', data=purchase_prediction_data, apply_standardization=True, load_existing_model=False, logger=logger, random_state=random_state)
    lr_fbeta_0_5.initialize_purchase_predictor(test_size=test_size, settings=LR_PURCHASE_PREDICTOR_SETTINGS, beta=0.5)


    #Try random forest

    # Best Purchase predictor in terms of F1 score
    rf_fbeta_1 = PurchasePredictor(name='rf_model', data=purchase_prediction_data, apply_standardization=True, load_existing_model=False, logger=logger, random_state=random_state)
    rf_fbeta_1.initialize_purchase_predictor(test_size=test_size, settings=RF_PURCHASE_PREDICTOR_SETTINGS, beta=1)

    # Best Purchase predictor in terms of F0.5 score
    rf_fbeta_0_5 = PurchasePredictor(name='more_precise_rf_model', data=purchase_prediction_data, apply_standardization=True, load_existing_model=False, logger=logger, random_state=random_state)
    rf_fbeta_0_5.initialize_purchase_predictor(test_size=test_size, settings=RF_PURCHASE_PREDICTOR_SETTINGS, beta=0.5)


    ####################################################################################################
    # Create Product Recommender                                                                       #
    ####################################################################################################

    product_recommender_1 = ProductRecommender(name='kNN_IBCF_recommender', load_existing_model=False, logger=logger)
    product_recommender_1.initialize_recommender(customer_item_matrix=product_recommendation_data, number_of_recommendations=15)

    
if __name__ == '__main__':
    print('Creates defined models, this may take a while...')
    create()
    print('Defined models were successfully created!')
