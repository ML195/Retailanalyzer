import pandas as pd

def get_indices_of_purchases_with_instant_cancellations(data: pd.DataFrame) -> list:
        """ Get indices for purchases that are followed by their cancellation within 14 days.

        Utility function that can be used by preprocessors. It was written for data cleaning purposes.

        Args:
            data (DataFrame):
                Data that contains the columns 'CustomerID', 'Invoice', 'InvoiceDate', 'StockCode', 'Quantity' and 'Price'.

        Returns:
                A list containing the indices of pruchases that are followed by their cancellation within 14 days.
        """
        
        temp_data = data.copy()
        
        # transform negative quantities to positive to enable comparison
        temp_data.Quantity = temp_data.Quantity.abs()
        
        # reset index and add the previous one as column
        temp_data.reset_index(inplace=True)
        
        # split data into purchases and pruchase cancellations
        cancellations = temp_data.loc[temp_data.Invoice.str.startswith('C')]
        non_cancellations = temp_data.loc[~temp_data.Invoice.str.startswith('C')]
        
        merge_on = ['StockCode', 'Quantity','Price', 'CustomerID']
        
        cancellation_purchases = cancellations.merge(non_cancellations, on=merge_on, how='inner', suffixes = ('_cancel', '_purchase'))
        
        cancellation_purchases.InvoiceDate_cancel = pd.to_datetime(cancellation_purchases.InvoiceDate_cancel)
        cancellation_purchases.InvoiceDate_purchase = pd.to_datetime(cancellation_purchases.InvoiceDate_purchase)
        
        # Filter out cancellations that are before the actual purchase
        instant_cancellations = cancellation_purchases[cancellation_purchases.InvoiceDate_cancel >= cancellation_purchases.InvoiceDate_purchase]
        
        # Filter out cancellations that are 14 days after the purchase (official cancellation span in UK)
        instant_cancellations = instant_cancellations[(instant_cancellations.InvoiceDate_cancel - instant_cancellations.InvoiceDate_purchase).dt.days <= 14]

        return instant_cancellations.index_purchase.tolist()