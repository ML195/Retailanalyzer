from interface import implements
import pandas as pd

from modelcreator.datahandler.preprocessor import IPreprocessor, preprocessor_utils
from modelcreator import model_creator_utils

class SalesForecasterPreprocessor(implements(IPreprocessor)):
    """ Preprocessor for sales forecasting tasks (implements the IPreprocessor interface).
    
    Enables the transformation of the given data into a time series with weekly aggregated sales.
    """

    def __init__(self):
        """Initializes SalesForecasterPreprocessor."""
        pass



    ####################################################################################################
    # Public functions                                                                                 #
    ####################################################################################################
    
    def get_preprocessed_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """ Preprocesses data for sales forecasting.

        The preprocessing consists of:

            - Cleaning the data
            - Calculatíng the total price (quantity * price) for a purchase
            - Generating a time series as a pd.DataFrame with time steps as index and the total price as values
            - Aggregate the total price per week

        Args:
            raw_data (DataFrame): 
                The data in its orignal format.
            
        Returns:
            The preprocessed data as pd.DataFrame, representing the time series with aggregated total price per week.
        """

        # check input types
        fun_params = locals()
        fun_params.pop('self', None)
        type_signature = [pd.DataFrame]
        model_creator_utils.check_type(type_signature, **fun_params)
        
        
        cleaned_data = self._get_cleaned_data(raw_data)

        # Calculate total price
        cleaned_data['TotalPrice'] = cleaned_data.loc[:,'Quantity'] * cleaned_data.loc[:,'Price']

        time_series_data = pd.DataFrame(
                            cleaned_data.loc[:,'TotalPrice'].tolist(), 
                            index=pd.to_datetime(cleaned_data['InvoiceDate']),
                            columns=['TotalPrice'])

        weekly_data = time_series_data.resample(rule='W')['TotalPrice'].sum()
        return weekly_data.to_frame()  
    


    ####################################################################################################
    # Private functions                                                                                #
    ####################################################################################################

    def _get_cleaned_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """ Cleans the data and returns it.

        The data cleaning consists of:
        
            - Dropping rows with missing CustomerID
            - Remove rows with StockCodes that not affect sales numbers (e.g. postage, charity, internal adjustments, etc.)

        Args:
            raw_data (DataFrame):
                The data in its orignal format.

        Returns:
            The cleaned data as pd.DataFrame.
        """
        
        # Remove NaN customer IDs as it is not sure where the corresponding column values come from
        cleaned_data = raw_data.dropna(subset=['CustomerID'])

        # StockCode categories to remove
        # (Discounts are not removed as they affect sales numbers)
        rem_stock = ['ADJUST', 'ADJUST2', 'BANK CHARGES', 'C2', 'CRUK', 'DOT', 'M', 'POST', 'TEST001', 'TEST002']
        cleaned_data = cleaned_data[~cleaned_data.StockCode.isin(rem_stock)]

        # Remove all purchases that are directly followed by their cancellation
        cleaned_data.drop(preprocessor_utils.get_indices_of_purchases_with_instant_cancellations(cleaned_data), inplace=True)

        # Remove all cancellations
        cleaned_data = cleaned_data.loc[~cleaned_data.Invoice.str.startswith('C')]

        return cleaned_data

    