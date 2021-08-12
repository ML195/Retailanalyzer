import datetime

from modelcreator.datahandler import DataHandler
from modelcreator.datahandler.preprocessor import SalesForecasterPreprocessor
from modelcreator.datahandler.preprocessor import PurchasePredictorPreprocessor
from modelcreator.datahandler.preprocessor import ProductRecommenderPreprocessor
from modelcreator.salesforecaster import LSTMForecaster
from modelcreator.purchasepredictor import PurchasePredictor
from modelcreator.productrecommender import ProductRecommender


#  This module shows example use case utilizing the modelcreator package.
#
#  It tackles the following use cases:
#      - Predicting the expected aggregated sales for the next month
#      - Get customers that are predicted to buy next quarter to send them an email with prodcuts they might be interested in (e.g maybe there is a sale for especially those products)
#      - Get customers that are predicted to not buy next quarter and send those customers that have also a recency of < 365 days (naive churning threshold) a discount code to give them an 
#        incentive to buy at the shop.
#      - Show an unregistered customer recommendations based on the articles he got in his shopping cart based on what others have bought. 


######################################################################
# Create the models                                                  #
######################################################################

# First step is to create some models which can be defined in model_creater.py
# This creates the models specified in the file and stores the models and their evaluation (model parameters, plots and achieved metrics) onto disk.

# commented as models were already created
#model_creator.create()

######################################################################
# 1. Predict expected aggregated sales for the next month            #
######################################################################

def predict_sales():
    
    # Load or create the data with the needed Preprocessing Strategy
    data_file_name = 'online_retail_II.csv'
    sales_forecasting_handler = DataHandler(data_file_name, SalesForecasterPreprocessor())

    # If we can't get prepocessed data it wasn't created already so we should generate it first
    try:
        sales_forecasting_data = sales_forecasting_handler.get_preprocessed_data()

    except(FileNotFoundError):
        sales_forecasting_data = sales_forecasting_handler.generate_preprocessed_data(return_dataset=True)


    # For predicting the sum of sales, the LSTM "stacked_small_model" is loaded as it achieved the overall best result on the training data. 
    # Based on that an LSTMForecaster instance with the name is created and load_existing_model is set to true to load the stored model
    forecaster = LSTMForecaster('stacked_small_model', sales_forecasting_data, load_existing_model=True)

    # When calling make_forecast() the forecast for the given periods is returned and a plot is stored to disk, visualizing the forecast (can be found in the /reports/evaluations/results folder for the specified model)
    expected_sales = forecaster.make_forecast(n_periods=4)

    print('The expected sales for the next four weeks are:')
    count = 0
    for date, value in expected_sales.items():
        start_of_week = date - datetime.timedelta(days=7)
        print(f'Week {count} ({start_of_week:%d/%m/%Y} - {date:%d/%m/%Y}) Expected Sales: $ {value:2f}')

    print(f'\nIn total the expected sales for the next four weeks are: $ {expected_sales.sum():2f}.')


######################################################################
# 2. Send product recommendation mail      	                         #
######################################################################

def send_product_recommendations():

    # As above, load or create the needed data using a DataHandler instance with an appropriate preprocessing strategy
    data_file_name = 'online_retail_II.csv'
    purchase_prediction_handler = DataHandler(data_file_name, PurchasePredictorPreprocessor())

    try:
        purchase_prediction_data = purchase_prediction_handler.get_preprocessed_data()

    except FileNotFoundError:
        purchase_prediction_data = purchase_prediction_handler.generate_preprocessed_data(return_dataset=True)

    # To predict customers that purchase next quarter we have to decide what the actual marketing strategy should be. Do we want a lower false negative rate (customers that are predicted to not purchase but actually do it) but can deal with a higher false positive rate (customers that are predicted to purchase don't actually do it) or vice versa? In the case of just sending product recommendations to assist a future buyer in his decision for example, we mabye want to reach as many of the customers that actually buy but can tolerate a higher fals positive rate as it is not costly to send recommendations out to the "wrong" customers. 


    # Based on that we can load the random forest model with the best F1 score ("rf_model") as it achieved the overall best result on the training data.
    predictor = PurchasePredictor('rf_model', purchase_prediction_data, True, True)
    
    # For getting the recommendations to the customers we load the trainded ProductRecommender
    recommender = ProductRecommender('kNN_IBCF_recommender', True)

    # select three customers for which to search the top recommended products
    buying_customers = predictor.get_all_customers_purchasing_next_quarter()[:3]
    recommendations = recommender.get_top_recommended_products_for_customer(buying_customers, 3)

    # Generate artifical mail text for printing
    customer_mails = []
    template = 'Hey, we hope you were satisifed with your last order and would like to thank you for shopping with us.\nBy the way, in the meantime, we found some products you may also be interested in: '

    for customer in buying_customers:
        # get the top three recommended products
        stock_codes = recommendations.loc[customer].values[:3]
        customer_template = template

        for stock_code in stock_codes:
            description = get_description_to_stock_code(purchase_prediction_handler, stock_code)
            customer_template = customer_template + '\n\t-' + description
        
        customer_mails.append(customer_template)
        
    for i, val in enumerate(buying_customers):
        print(f'Mail for customer {val}')
        print(customer_mails[i],'\n')


######################################################################
# 3. Send discount mail      	                                     #
######################################################################

def send_discounts():
    
    # Load or create the data
    data_file_name = 'online_retail_II.csv'
    purchase_prediction_handler = DataHandler(data_file_name, PurchasePredictorPreprocessor())

    try:
        purchase_prediction_data = purchase_prediction_handler.get_preprocessed_data()

    except FileNotFoundError:
        purchase_prediction_data = purchase_prediction_handler.generate_preprocessed_data(return_dataset=True)

     # As in the example above we need to think about what the marketing strategy should be. When we send discounts we want to target as many customers that are actually not buying (true negatives) and don't care as much about false negatives as losing customers because we are applying no marketing strategy (sending discounts), is more costly in the long term as sending some discounts to customers that would buy nevertheless.

    # Based on that we load the random forest model with the best F0.5 (recall is half as important) score ("more_precise_rf_model") as it achieved the overall best result on the training data. 
    predictor = PurchasePredictor('more_precise_rf_model', purchase_prediction_data, True, True)

    # For adding the recommendations at the end of the mail we also need our trained ProductRecommender
    recommender = ProductRecommender('kNN_IBCF_recommender', True)
    
    # Select three customers for which to search the top recommended products
    not_buying_customers = predictor.get_all_customers_not_purchasing_next_quarter()[:3]
    recommendations = recommender.get_top_recommended_products_for_customer(not_buying_customers, 3)

    customer_mails = []

    for customer in not_buying_customers:
        recency = get_recency_to_customer(purchase_prediction_data, customer)

        # get the top three recommended products
        stock_codes = recommendations.loc[customer].values[:3]

        # Naive churning check
        if recency <= 365:

            # Generate artifical mail text for printing
            customer_template = 'Hey, we just thought about you and wanted to give you a 20% discount!\nHere are some products you may be interested in:'
            for stock_code in stock_codes:
                description = get_description_to_stock_code(purchase_prediction_handler, stock_code)
                customer_template = customer_template + '\n\t-' + description
            
            customer_mails.append(customer_template)
    
    for i, val in enumerate(not_buying_customers):
        print(f'Mail for customer {val}')
        print(customer_mails[i],'\n')


######################################################################
# 4. Shopping Cart Recommendation      	                             #
######################################################################


def get_shopping_cart_recommendations():
    
    # Load or create the data
    data_file_name = 'online_retail_II.csv'
    product_recommednation_handler = DataHandler(data_file_name, ProductRecommenderPreprocessor())

    example_products_in_cart = ['17038', '17084N', '46118']

    # create artifical text to show
    cart_template_text = 'Your cart contains:'

    for product in example_products_in_cart:
        description = get_description_to_stock_code(product_recommednation_handler, product)
        cart_template_text = cart_template_text + '\n\t-' + description

    # load recommender
    recommender = ProductRecommender('kNN_IBCF_recommender', True)

    # find top 3 recommended products to shopping cart item
    recommendations = recommender.get_top_recommended_products_for_product(example_products_in_cart).iloc[:,:3]
    
    # create artifical text to show
    recommended_products = []
    recommendation_template_text = 'Based on your shopping cart you may also like:'

    for products_row in recommendations.values.tolist():
        if products_row[0] in recommended_products:
            for product in products_row[1:]:
                if not product in recommended_products:
                    recommended_products.append(product)
        else:
            recommended_products.append(products_row[0])
        
    for product in recommended_products:
        description = get_description_to_stock_code(product_recommednation_handler, product)
        recommendation_template_text = recommendation_template_text + '\n\t-' + description

    print(cart_template_text,'\n')
    print(recommendation_template_text)


######################################################################
# Convenience Functions      	                                     #
######################################################################

def get_description_to_stock_code(data_handler, stock_code):
    stock_description = data_handler.raw_data.loc[:,['StockCode', 'Description']]
    stock_description.set_index('StockCode', inplace=True)
    stock_description.dropna(inplace=True)
    stock_description = stock_description.groupby(by=['StockCode']).last()

    return stock_description.loc[stock_code].values[0]


def get_recency_to_customer(data, customer_id):
    customer_recency = data.loc[:,['recency', 'target']]
    customer_recency = customer_recency.loc[customer_recency.target == 'predict']
    customer_recency.drop('target', axis=1, inplace=True)

    return customer_recency.loc[customer_id].values[0]

def run_use_cases():
    print('--------------------------------------------------')
    print('Use Case 1: Predict Aggregated Sales (4 Weeks)')
    print('--------------------------------------------------')
    predict_sales()

    print('\n--------------------------------------------------')
    print('Use Case 2: Product Recommendation Mail')
    print('--------------------------------------------------')
    send_product_recommendations()

    print('\n--------------------------------------------------')
    print('Use Case 3: Discount Mail')
    print('--------------------------------------------------')
    send_discounts()

    print('--------------------------------------------------')
    print('Use Case 4: Shopping Cart Recommnedation')
    print('--------------------------------------------------')
    get_shopping_cart_recommendations()

if __name__ == '__main__':
    run_use_cases()

# prints the following lines
# Notice the customer with ID 12356.0, which is classifier differently by the two models 
'''
--------------------------------------------------
Use Case 1: Predict Aggregated Sales (4 Weeks)
--------------------------------------------------
The expected sales for the next four weeks are:
Week 0 (11/12/2011 - 18/12/2011) Expected Sales: $ 262352.065843
Week 0 (18/12/2011 - 25/12/2011) Expected Sales: $ 273736.944491
Week 0 (25/12/2011 - 01/01/2012) Expected Sales: $ 278152.563347
Week 0 (01/01/2012 - 08/01/2012) Expected Sales: $ 272273.716490

In total the expected sales for the next four weeks are: $ 1086515.290170.

--------------------------------------------------
Use Case 2: Product Recommendation Mail
--------------------------------------------------
Mail for customer 12347.0
Hey, we hope you were satisifed with your last order and would like to thank you for shopping with us.
By the way, in the meantime, we found some products you may also be interested in:
        -SET/2 TEA TOWELS MODERN VINTAGE
        -F FAIRY POTPOURRI CUSHIONS SUMMER
        -PINK PAINTED KASHMIRI TABLE

Mail for customer 12352.0
Hey, we hope you were satisifed with your last order and would like to thank you for shopping with us.
By the way, in the meantime, we found some products you may also be interested in:
        -LADIES & GENTLEMEN METAL SIGN
        -N0 SINGING METAL SIGN
        -FANNY'S REST STOPMETAL SIGN

Mail for customer 12356.0
Hey, we hope you were satisifed with your last order and would like to thank you for shopping with us.
By the way, in the meantime, we found some products you may also be interested in:
        -TEA TIME BREAKFAST BASKET
        -CERAMIC CAKE STAND + HANGING CAKES
        -CERAMIC BOWL WITH STRAWBERRY DESIGN


--------------------------------------------------
Use Case 3: Discount Mail
--------------------------------------------------
Mail for customer 12348.0
Hey, we just thought about you and wanted to give you a 20% discount!
Here are some products you may be interested in:
        -PACK OF 12 PINK POLKADOT TISSUES
        -PACK OF 12 PINK PAISLEY TISSUES
        -SWEET PUDDING STICKER SHEET

Mail for customer 12349.0
Hey, we just thought about you and wanted to give you a 20% discount!
Here are some products you may be interested in:
        -LILY BROOCH WHITE/SILVER COLOUR
        -RED RETROSPOT TRADITIONAL TEAPOT
        -UTILTY CABINET WITH HOOKS

Mail for customer 12356.0
Hey, we just thought about you and wanted to give you a 20% discount!
Here are some products you may be interested in:
        -TEA TIME BREAKFAST BASKET
        -CERAMIC CAKE STAND + HANGING CAKES
        -CERAMIC BOWL WITH STRAWBERRY DESIGN

--------------------------------------------------
Use Case 4: Shopping Cart Recommnedation
--------------------------------------------------
Your cart contains:
        -PORCELAIN BUDAH INCENSE HOLDER
        -FAIRY DREAMS INCENSE
        -FUNKY MONKEY CUSHION COVER 

Based on your shopping cart you may also like:
        -ASSORTED LAQUERED INCENSE HOLDERS
        -DRAGONS BLOOD INCENSE
        -KID'S CHALKBOARD/EASEL
'''