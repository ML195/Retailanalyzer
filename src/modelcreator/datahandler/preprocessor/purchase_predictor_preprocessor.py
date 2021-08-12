
from functools import reduce
from interface import implements
import pandas as pd

from modelcreator.datahandler.preprocessor import IPreprocessor, preprocessor_utils
from modelcreator import model_creator_utils


class PurchasePredictorPreprocessor(implements(IPreprocessor)):
    """Preprocessor for purchase prediction tasks (implements the IPreprocessor interface).
    
    Enables the transformation of the given data into a data set with the following features for each customer:

        - 'recency' -> recency of last pruchase 
        - 'frequency' -> frequency of pruchases 
        - 'diff_purchase_t_1', ..., 'diff_purchase_t_n-1' -> differences between the latest purchase and the last n-1 purchases 
        - 'purchase_subsequent_quarters' -> number of times the customer bought in two subsequent quarters 
        - 'y_label' -> binary label (1.0 = yes, 0.0 = no) defining if the customer placed orders in the latest quarter and in the previous one
        - 'target' designation if data is for training ('train') or should be used to predict ('predict')
    """

    def __init__(self):
        """Initializes PurchasePredictorPreprocessor."""
        pass



    ####################################################################################################
    # Public functions                                                                                 #
    ####################################################################################################
    
    def get_preprocessed_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """ Preprocess data for purchase prediction.

        The preprocessing consists of:
        
            - Cleaning the data 
            - Splitting the data into current quarter (latest quarter in the data) and past quarters (quarters prior to the latest one)
            - Calculating the features with
                - ``_get_RF()``
                - ``_get_days_between_last_n_purchases()``
                - ``_get_times_customer_bought_next_quarter()``
            - Calculate the label with
                - ``_get_next_quarter_purchase_label()``
            - Concatenate the features and labels into a pd.DataFrame
            - Drop customers thar are below a frequency of 3
            - Designate the created data with 'target' = 'train' to indicate that the data should be used for training the predictor
            - Repeat the same steps for the whole data (not split between current and past quarters), to use it to predict if a customer will buy next quarter (outside the given data), designated with 'target' = 'predict'
            - Concatenate the data for training and prediction and return it

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
        
        # Split the data into past quarter and current quarter, where past_quarters serves as observation period
        past_quarters, current_quarter = self._separate_current_quarter_from_past_quarters(cleaned_data)

        train_recency_frequency = self._get_RF(past_quarters)

        # How many pruchases to consider when calculating the difference in days between the latest pruchase and the last n-1 ones
        last_purchases = 3

        train_days_between_purchases = self._get_days_between_last_n_purchases(past_quarters, last_purchases)
        train_bnought_next_quarter = self._get_times_customer_bought_next_quarter(past_quarters)
        train_labels = self._get_next_quarter_purchase_label(past_quarters, current_quarter)

        prepared_train_data = pd.concat([train_recency_frequency, train_days_between_purchases, train_bnought_next_quarter, train_labels],axis=1)

        # Remove customers that have a frequency below last_purchases to remove NaN values
        prepared_train_data = prepared_train_data.loc[prepared_train_data.frequency >= last_purchases]
        prepared_train_data['target'] = 'train'

        # Calculate features for the whole dataset, which can be used to predict if a customer purchases a product next quarter (outside the given dataset)
        predict_recency_frequency = self._get_RF(cleaned_data)
        predict_days_between_purchases = self._get_days_between_last_n_purchases(cleaned_data, last_purchases)
        predict_test_bnought_next_quarter = self._get_times_customer_bought_next_quarter(cleaned_data)
        
        prepared_predict_data = pd.concat([predict_recency_frequency, predict_days_between_purchases, predict_test_bnought_next_quarter],axis=1)

        prepared_predict_data = prepared_predict_data.loc[prepared_predict_data.frequency >= last_purchases]
        prepared_predict_data['target'] = 'predict'

        prepared_data = pd.concat([prepared_train_data, prepared_predict_data],axis=0)
        
        return prepared_data



    ####################################################################################################
    # Private functions                                                                                #
    ####################################################################################################

    def _get_cleaned_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """ Cleans the data and returns it.

        The data cleaning consists of:
            - Dropping rows with missing CustomerID
            - Removing rows with StockCodes that not affect the number of purchases (e.g. postage, charity, discounts, internal adjustments, etc.)
            - Removing purchases that are directly (within 14 days) followed by their full cancellation
            - Removing all cancellations as they have nothing to do with purchase numbers
            - To make working with InvoiceDates more convenient (as only the day of purchase is important) they are normalized to 00:00:00

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
        
        # Set timestamp to 00:00:00
        cleaned_data.InvoiceDate = pd.to_datetime(cleaned_data.InvoiceDate).dt.normalize()

        return cleaned_data


    def _separate_current_quarter_from_past_quarters(self, data: pd.DataFrame) -> tuple:
        """Separate ``data`` into latest quarter and quarters prior to the latest one.

        Args:
            data (DataFrame):
                Data containing the columns 'InvoiceDate' and 'CustomerID' as pd.DataFrame.

        Returns:
            A tuple of the form (pd.DataFrame, pd.DataFrame) that contains the quarters prior to the latest one in ``data`` as first element and the data to the latest quarter as second element.
        """

        max_date = data.InvoiceDate.max()
    
        # Get end date of the quarter previous to that of the maximum observed date
        end_past_quarters = (max_date - pd.tseries.offsets.QuarterEnd())

        past_quarters =  data[data.InvoiceDate <= end_past_quarters].reset_index(drop=True)
        last_quarter = data[data.InvoiceDate > end_past_quarters].reset_index(drop=True)
        
        return past_quarters, last_quarter

    
    def _get_next_quarter_purchase_label(self, past_quarters: pd.DataFrame, current_quarter: pd.DataFrame) -> pd.Series:    
        """Get binary labels (1.0 = yes, 0.0 = no) if customers placed orders in the last quarter of ``past_quarters`` and in the current quarter.
        
        Args:
            past_quarters (DataFrame):
                Data representing the quarters prior to the last one in the data as pd.DataFrame containing the columns 'InvoiceDate' and 'CustomerID'.

            current_quarter (DataFrame):
                Data representing the last quarter in the data as pd.DataFrame containing the columns 'InvoiceDate' and 'CustomerID'.
            
        Returns:
            The binary labels (1.0 = yes, 0.0 = no) defining if customers placed orders in the last quarter of ``past_quarters`` as well as in the current quarter as a pd.Series, with CustomerID as index and the corresponding binary label 'y_label' as values.    
        """
        # Get last purchase date in past quarters for every customer
        past_quarters_last_purchases = past_quarters.groupby(by=['CustomerID']).InvoiceDate.max()
        
        # Get first purchase date in current quarter for every customer
        current_quarter_first_purchases = current_quarter.groupby(by=['CustomerID']).InvoiceDate.min()
        
        # Rename columns
        past_quarters_last_purchases.name = 'past_quarter_last_purchase'
        current_quarter_first_purchases.name = 'current_quarter_first_purchase'
        
        # Concat the two series to indentify which customers bought in the past and current quarter
        purchase_dates = pd.concat([past_quarters_last_purchases, current_quarter_first_purchases], axis=1)
        
        # Remove customers who purchased a product the first time in the current quarter 
        purchase_dates = purchase_dates[~((purchase_dates.past_quarter_last_purchase.isnull()) & ~(purchase_dates.current_quarter_first_purchase.isnull()))]
        
        # Get begin and end of the quarter before the maximum purchase date
        prev_quarter_begin, prev_quarter_end = self._get_previous_quarter_begin_and_end_to_date(current_quarter.InvoiceDate.max())
        
        # Set labels
        purchase_dates['y_label'] = 0
        
        # For all customers for whom the last purchase data is in the previous quarter and for which there is a purchase date in the current quarter set label to 1 (yes)
        purchase_dates.loc[((purchase_dates.past_quarter_last_purchase >= prev_quarter_begin) & (purchase_dates.past_quarter_last_purchase <= prev_quarter_end)) & ~(purchase_dates.current_quarter_first_purchase.isnull()), 'y_label'] = 1

        return purchase_dates.y_label


    def _get_RF(self, data: pd.DataFrame) -> pd.DataFrame:
        """Get the recency and frequency of purchases for customers.

        Args:
            data (DataFrame):
                Data containing the columns 'InvoiceDate' and 'CustomerID' as pd.DataFrame.
            
        Returns:
            The recency and frequency of purchases for each customer as a pd.DataFrame with CustomerID as index and 'recency' and 'frequency' as columns.
        """

        observation_period_end_date = data.InvoiceDate.max() + pd.tseries.offsets.QuarterEnd(n=0)

        recency_frequency = data.groupby(by=['CustomerID']).agg(
            recency=('InvoiceDate', lambda date: (observation_period_end_date-date.max()).days),
            frequency=('InvoiceDate', 'nunique'))
            
        return recency_frequency


    def _get_days_between_last_n_purchases(self, data: pd.DataFrame, n: int) -> pd.DataFrame:
        """Get the difference in days between the latest purchase and the last ``n-1`` purchases for customers.

        For example: if ``n=3`` the days between the latest purchase (n=1) and the purchase at t-1 (n=2) as well as the days between the latest pruchase (n=1) and the purchase at t-2 (n=3) are calculated.

        Args:
            data (DataFrame):
                Data containing the columns 'InvoiceDate' and 'CustomerID' as pd.DataFrame.

            n (int):
                Number of purchases to include.
            
        Returns:
            The difference in days between the latest purchase and the last ``n-1`` purchases for each customer as pd.DataFrame, with CustomerID as index and the ``n-1`` differences as columns. For example with ``n = 3``: index is a Float64Index([12347.0], dtype='float64', name='CustomerID') with columns [diff_purchase_t_1  diff_purchase_t_2] and values [37.0, 127.0].
        """

        # Get the data ready for grouping by CustomerID
        sub_data = data.loc[:,['CustomerID','InvoiceDate']]
        sub_data.sort_values(by = ['CustomerID','InvoiceDate'], inplace=True)
        sub_data.drop_duplicates(subset=['CustomerID','InvoiceDate'], inplace=True)
        
        # Only get the recent n purchases per customer
        last_n_purchases = sub_data.groupby(by=['CustomerID']).tail(n)
        
        # Names of columns containing the difference in days
        diff_columns = []

        for i in range(1,n):
            # Get date of previous prurchase at t-i
            purchase_column = 'purchase_t_'+str(i)
            last_n_purchases[purchase_column] = last_n_purchases.groupby(by=['CustomerID']).InvoiceDate.shift(i)
            
            # Get differences of latest purchase and prurchase at t-i
            diff_column = 'diff_purchase_t_'+str(i)
            last_n_purchases[diff_column] = (last_n_purchases.InvoiceDate - last_n_purchases[purchase_column]).dt.days

            diff_columns.append(diff_column)
        
        # Return the last entry for every difference column grouped by customer, it is worth mentioning that customers that have bought less than n products have NaN entries
        return last_n_purchases.groupby(by=['CustomerID'])[diff_columns].last().astype('Int64')


    def _get_times_customer_bought_next_quarter(self, data: pd.DataFrame) -> pd.Series:
        """Get the number of times a customer placed orders in two subsequent months over the timespan of the given data.

        Args:
            data (DataFrame):
                Data containing the columns 'InvoiceDate' and 'CustomerID' as pd.DataFrame.

        Returns:
            The number of times a customer bought in the next quarter as pd.Series, where the CustomerID is the index and the values represent the times the customer placed orders in two subsequent months over the timespan of ``data``.
        """

        # Get minimum and maximum purchase date
        min_purchase_date = data.InvoiceDate.min()
        max_purchase_date = data.InvoiceDate.max()
        
        counter_list = []
        
        # Get begin and end of quarter of the minimum purchase date (fist purchase)
        curr_quarter_begin, curr_quarter_end = self._get_current_quarter_begin_and_end_to_date(min_purchase_date)

        # As long as the end of the current quarter is before the maximum purchase date (last purchase), this is important as curr_quarter_end gets updated in the loop
        while curr_quarter_end <= max_purchase_date:

            # Get the next quarter to the current one 
            next_quarter_begin, next_quarter_end = self._get_next_quarter_begin_and_end_to_date(curr_quarter_end)
            
            # Get the last purchase of the current quarter for each customer
            purchase_curr_quarter = data.loc[(data.InvoiceDate >= curr_quarter_begin) & (data.InvoiceDate <= curr_quarter_end)].groupby('CustomerID').InvoiceDate.last()

            # Get the first purchase of the next quarter for each customer
            purchase_next_quarter = data.loc[(data.InvoiceDate >= next_quarter_begin) & (data.InvoiceDate <= next_quarter_end)].groupby('CustomerID').InvoiceDate.first()
            
            # Set series column name
            purchase_curr_quarter.name = 'date_curr_quarter'
            purchase_next_quarter.name = 'date_next_quarter'

            # Concatenate the two series horizontally
            purchase_in_two_quarters = pd.concat([purchase_curr_quarter, purchase_next_quarter], axis=1)

            # Create a new column that reflects if the customer pruchased in both quarters and set it to 0 (for no)
            purchase_in_two_quarters['purchase_subsequent_quarters'] = 0

            # As the series containing the pruchases for the current and the next quarter can contain different customers it is easy to spot if a customer bought in both quarters (customerID existed in both) or not (customer existed only in one). If a customer existed in both series a purchase was made in both quarters and the value is set to 1
            purchase_in_two_quarters.loc[~(purchase_in_two_quarters.date_curr_quarter.isnull()) & ~(purchase_in_two_quarters.date_next_quarter.isnull()), 'purchase_subsequent_quarters'] = 1
            
            # Append the column to a list so that there is no need to contionously append to a dataframe (slow)
            counter_list.append(purchase_in_two_quarters.purchase_subsequent_quarters)
            
            # Update the current quarter and set it to the next and repeat the process
            curr_quarter_begin, curr_quarter_end = next_quarter_begin, next_quarter_end
        
        # Sum up the columns in the counter_list to get the number of times a customer bought two quarters in a row
        times_bought_next_quarter = reduce(lambda x, y: x.add(y, fill_value=0), counter_list)

        return times_bought_next_quarter.astype('Int64')


    def _get_previous_quarter_begin_and_end_to_date(self, date: pd.Timestamp) -> tuple:
        """Get beginning and end date of previous quarter in relation to the given date.

        Args:
            date (Timestamp):
                Date of type pd.Timestamp.

        Returns:
            A tuple of type (pd.Timestamp, pd.Timestamp) containing the date of the beginning and end of the previous quarter in relation to ``date``.
        """
        prev_quarter_end = date - pd.tseries.offsets.QuarterEnd()
        prev_quarter_begin = prev_quarter_end - pd.tseries.offsets.QuarterBegin(startingMonth=1)
        
        return prev_quarter_begin, prev_quarter_end   


    def _get_current_quarter_begin_and_end_to_date(self, date: pd.Timestamp) -> tuple:
        """Get beginning and end date of current quarter in relation to the given date.

        Args:
            date (Timestamp):
                Date of type pd.Timestamp.

        Returns:
            A tuple of type (pd.Timestamp, pd.Timestamp) containing the date of the beginning and end of the current quarter in relation to ``date``.
        """
        curr_quarter_end = date +  pd.tseries.offsets.QuarterEnd(n=0)
        curr_quarter_begin = curr_quarter_end -  pd.tseries.offsets.QuarterBegin(startingMonth=1)
    
        return curr_quarter_begin, curr_quarter_end


    def _get_next_quarter_begin_and_end_to_date(self, date: pd.Timestamp) -> tuple: 
        """Get beginning and end date of next quarter in relation to the given date.

        Args:
            date (Timestamp):
                Date of type pd.Timestamp.

        Returns:
            A tuple of type (pd.Timestamp, pd.Timestamp) containing the date of the beginning and end of the next quarter in relation to ``date``.
        """
        next_quarter_begin = date + pd.tseries.offsets.QuarterBegin(startingMonth=1)
        next_quarter_end = next_quarter_begin + pd.tseries.offsets.QuarterEnd()

        return next_quarter_begin, next_quarter_end