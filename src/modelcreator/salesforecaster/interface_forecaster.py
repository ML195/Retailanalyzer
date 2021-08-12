from interface import Interface
from typing import Union
import pandas as pd

class IForecaster(Interface):
    """Forecaster interface."""

    def initialize_forecaster(self, test_size: Union[int, float], hyperparameters: dict = None, **kwargs):
        """Initializes forecaster instance.

        Args:
            test_size (int or float):
                How many samples of the given time series should be used as a hold-out test set. If float (should be between 0.0 and 1.0), the value represents the proportion of the dataset used as test samples. If int the value represents the absolute number of test samples.

            hyperparameters (dict):
                Hyperparameters used for model building as dict of the form {'parameter_1': [value_1, value_2, ...], 'parameter_2': [...], ...}.

            **kwargs: 
                Additional parameters.
        """
        pass
    
    def make_forecast(self, n_periods: int) -> pd.Series:
        """Makes a forecast for ``n_periods``.
       
        Args:
            n_periods (int):
                Defines the number of future periods to forecast.

        Returns:
            A time series of type pd.Series, where the index consists of the future n timesteps and the values quantify the future observations corresponding to the predicted timesteps.
        """
        pass
