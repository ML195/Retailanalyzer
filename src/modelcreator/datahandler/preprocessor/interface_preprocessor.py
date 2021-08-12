from interface import Interface
import pandas as pd

class IPreprocessor(Interface):
    """ Preprocessor interface."""
    
    def get_preprocessed_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """ Preprocesses the data.

        Args:
            raw_data (DataFrame): 
                The data in its orignal format.

        Returns:
            The preprocessed data as pd.DataFrame.
        """
        pass
