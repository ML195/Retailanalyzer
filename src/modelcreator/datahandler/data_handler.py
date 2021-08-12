from pathlib import Path
from typing import Union

import pandas as pd

from path_definitions import PATHS
from modelcreator import model_creator_utils
from modelcreator.datahandler.preprocessor import IPreprocessor, SalesForecasterPreprocessor, PurchasePredictorPreprocessor, ProductRecommenderPreprocessor

class DataHandler():
    """Generates preprocessed data according to a passed preprocessor strategy.
    
    A DataHandler instance facilitates the data prepocessing by automatically transforming the data according to a given preprocessor strategy, storing the data on disk as a .feather and making it conveniently accessible.

    Attributes:
        _raw_data (DataFrame): Unprocessed data
        _preprocessor_strategy (IPreprocessor): Preprocessor strategy to use to generate the data
        _data_name (str): The name of the processed data's file
    """

    def __init__(self, file_name: str, preprocessor_strategy: IPreprocessor):
        """Initializes DataHandler.
        
        Args:
            file_name (str):
                The name of the original data file located in data/raw/ with file ending.

            preprocessor_strategy (IPreprocessor):
                The preprocessor strategy of type IPreprocessor that should be used.
        """

        # check input types
        fun_params = locals()
        fun_params.pop('self', None)
        type_signature = [str, (SalesForecasterPreprocessor, PurchasePredictorPreprocessor, ProductRecommenderPreprocessor)]
        model_creator_utils.check_type(type_signature, **fun_params)

        raw_file_path = PATHS['RAW_DATA_DIR'] / file_name

        if raw_file_path.exists():
            # read the file from csv to pandas dataframe
            self._raw_data = pd.read_csv(raw_file_path)

            # set the preprocessor strategy 
            self._preprocessor_strategy = preprocessor_strategy

            # name the data, which is stored ultimately, with the given preprocessor strategy's name 
            self._data_name = self._preprocessor_strategy.__class__.__name__.replace('Preprocessor', '') + 'Data'

            # rename Customer ID
            self._raw_data.rename(columns={'Customer ID': 'CustomerID'}, inplace=True)
        else:
            raise FileNotFoundError(f'The specified file {file_name} cannot be found in the data/raw/ folder, check the name of the file or check if file exists in the folder.')


    @property
    def raw_data(self):
        """Getter for the _raw_data attribute. 

        Returns:
            The unprocessed data as pd.DataFrame.
        """
        return self._raw_data.copy()

    
    ####################################################################################################
    # Public functions                                                                                 #
    ####################################################################################################

    def generate_preprocessed_data(self, return_path: bool = False, return_dataset: bool = False, overwrite_if_exists: bool = False) -> Union[tuple, Path, pd.DataFrame, None]:
        """Generates a preprocessed data set based on the given preprocessor strategy

        Args:
            return_path (bool):
                Defines if the function should return the path to the generated data. Default is False (no).

            return_dataset (bool):
                Defines if the function should return the generated data. Default is False (no).
           
            overwrite_if_exists (bool):
                Defines if the function should overwrite an already existing dataset with the same name. Default is False (no).

        Returns:
            Depending on the passed parameters. If ``return_path`` and ``return_dataset`` are True a tuple of tpye (Path, pd.DataFrame) is returned, containing the path to the generated data and the data itself. If only ``return_path`` is True the path to the data is returned as a Path object. If only ``return_dataset`` is True, the generated data itself is returned as pd.DataFrame. If ``return_path`` and ``return_dataset`` are both False, None is returned.
        """
        # check input types
        fun_params = locals()
        fun_params.pop('self', None)
        type_signature = [bool, bool, bool]
        model_creator_utils.check_type(type_signature, **fun_params)
        
        preprocessed_data = self._preprocessor_strategy.get_preprocessed_data(self.raw_data)

        path_to_data = PATHS['PROCESSED_DATA_DIR'] / (self._data_name + '.feather')

        path_to_data_exists = path_to_data.exists()
        
        if path_to_data_exists and overwrite_if_exists:
            # Reset index as feather can only handle primitive indices
            data_to_store = preprocessed_data.reset_index()
            data_to_store.to_feather(path_to_data)

        elif not path_to_data_exists:
            # Reset index as feather can only handle primitive indices
            data_to_store = preprocessed_data.reset_index()
            data_to_store.to_feather(path_to_data)

        # Return if-cascade
        if return_path and return_dataset:
            return (path_to_data, preprocessed_data)

        elif return_path:
            return path_to_data

        elif return_dataset:
            return preprocessed_data

    def get_preprocessed_data(self) -> pd.DataFrame:
        """Returns the preprocessed data.

        Returns:
            The preprocessed data as pd.DataFrame.

        Raises:
            FileNotFoundError: When there is no file of the preprocessed data, implying ``generate_preprocessed_data()`` was not called first.
        """

        path_to_data = PATHS['PROCESSED_DATA_DIR'] / (self._data_name + '.feather')

        if path_to_data.exists():
            preprocessed_data = pd.read_feather(path_to_data)
            # Set the index to the first column and drop the column afterwards
            preprocessed_data.set_index(preprocessed_data.columns[0], drop=True, inplace=True)
            return preprocessed_data
        else:
            raise FileNotFoundError('There is no file to the preprocessed data, please call generate_preprocessed_data() first.')
    
    
        
        




        
    
    
    