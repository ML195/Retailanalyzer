from interface import implements
import pandas as pd

from modelcreator.datahandler.preprocessor import IPreprocessor, preprocessor_utils
from modelcreator import model_creator_utils


class ProductRecommenderPreprocessor(implements(IPreprocessor)):
    """Preprocessor for product recommending tasks (implements the IPreprocessor interface).
    
    Enables the transformation of the given data into a customer-item matrix with CustomerID as rows and StockCodes (products) as columns, where purchased products are designated as 1.0, whereas the rest is set to 0.0.
    """

    def __init__(self):
        """Initializes ProductRecommenderPreprocessor."""
        pass

    ####################################################################################################
    # Public functions                                                                                 #
    ####################################################################################################

    def get_preprocessed_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for product recommending.

        The preprocessing consists of:
            - Cleaning the data
            - Taking a subset of the data with columns CustomerID and StockCode
            - Drop duplicate rows as it is only important to know which products a customer bought at least one time
            - Create a additional column called "purchased", which designates that a customer bought that product
            - Create a customer-item-matrix with CustomerID as rows and StockCodes (products) as columns (if a customer bought an item there is a 1.0 at cell [CustomerID, StockCode] otherwise the value is 0.0)

        Args:
            raw_data (DataFrame): 
                The data in its orignal format.
            
        Returns:
            The preprocessed data as pd.DataFrame, representing a customer-item-matrix with CustomerID as rows and StockCodes as columns
        """

        # check input types
        fun_params = locals()
        fun_params.pop('self', None)
        type_signature = [pd.DataFrame]
        model_creator_utils.check_type(type_signature, **fun_params)

        cleaned_data = self._get_cleaned_data(raw_data)
        
        sub_data = cleaned_data.loc[:,['CustomerID', 'StockCode']]
        sub_data.drop_duplicates(subset=['CustomerID', 'StockCode'], inplace=True)

        sub_data['purchased'] = 1

        #create customer-item-matrix
        customer_item_matrix = sub_data.pivot(index='CustomerID', columns='StockCode', values='purchased')
        customer_item_matrix.fillna(0, inplace=True)
        customer_item_matrix.rename_axis(None, axis=1, inplace=True)
        
        return customer_item_matrix



    ####################################################################################################
    # Private functions                                                                                #
    ####################################################################################################

    def _get_cleaned_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Cleans the data and returns it.

        The data cleaning consists of:
        
            - Dropping rows with missing CustomerID
            - Removing rows with StockCodes that not affect the number of purchases (e.g. postage, charity, discounts, internal adjustments, etc.)
            - Removing purchases that are directly (within 14 days) followed by their full cancellation
            - Removing all cancellations as they have nothing to do with purchase numbers

        Args:
            raw_data (DataFrame):
                The data in its orignal format.

        Returns:
            The cleaned data as pd.DataFrame.
        """

        # Remove NaN customer IDs
        cleaned_data = raw_data.dropna(subset=['CustomerID'])

        # StockCode categories to remove
        rem_stock = ['ADJUST', 'ADJUST2', 'BANK CHARGES', 'C2', 'CRUK', 'D', 'DOT', 'M', 'POST', 'TEST001', 'TEST002', ]
        cleaned_data = cleaned_data[~cleaned_data.StockCode.isin(rem_stock)]

        # Remove all purchases that are directly followed by their cancellation
        cleaned_data.drop(preprocessor_utils.get_indices_of_purchases_with_instant_cancellations(cleaned_data), inplace=True)

        # Remove all cancellations
        cleaned_data = cleaned_data.loc[~cleaned_data.Invoice.str.startswith('C')]

        return cleaned_data

    