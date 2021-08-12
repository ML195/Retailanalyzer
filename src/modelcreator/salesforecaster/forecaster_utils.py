from multiprocessing import Value
from re import T
from typing import Union
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

####################################################################################################
# Utility functions for forecaster                                                                 #
####################################################################################################

def check_n_periods(n_periods):
    """Checks if n_periods is in a certain range.

    Args:
        n_periods (int):
            Defines the number of future periods to forecast.

    Raises: 
        ValueError: If n_periods is not in the expected range (> 0 and <= 52).
    """
    
    if n_periods > 52:
        raise ValueError('It is not supported to make forecasts for n_periods > 52.')
        
    elif n_periods <= 0:
        raise ValueError('n_periods must be at least 1.')

def get_sequence_data(time_series: Union[pd.Series, pd.DataFrame], input_lags: int, output_lags: int) -> tuple:
    """ Get the time series as sequence.

    Transforms the time series into a sequence format for prediction with an LSTM model. For example with ``input_lags=4`` and ``output_lags=1``, X1 = [t0, t1, t2, t3], y1 = [t4], X2 = [t1, t2, t3, t4], y2 = [t5], X3 = ... 

    Utility function that is used by a LSTMForecaster in the data preparation steps.

    Args:
        time_series (Series or DataFrame):
            The time series that should be normalized as pd.Series or as pd.DataFrame with one column.

        input_lags (int):
            Lags to use in the sequence.

        output_lags (int):
            Lags to use in the traget.

    Returns:
        A tuple of the form (np.ndarray, np.ndarray) containing the input sequences as first element with shape (n_samples, ``input_lags``) and the target as second element with shape (n_samples, 1).
    """
    
    nd_time_series = time_series.to_numpy().ravel()
    max_i = nd_time_series.size - (input_lags + output_lags) + 1

    X_data = np.empty((0,input_lags))
    y_data = np.empty((0,output_lags))
    
    for i, val in np.ndenumerate(nd_time_series):
        if i[0] < max_i:
            X_observations = []
            y_observations = []
            for in_lag in range(input_lags):
                X_observations.append(nd_time_series[i[0]+in_lag])
                
            for out_lag in range(output_lags):
                y_observations.append(nd_time_series[i[0]+input_lags+out_lag])

            X_data = np.vstack((X_data, X_observations))
            y_data = np.vstack((y_data, y_observations))

    return X_data, y_data


def time_series_train_test_split(time_series: np.ndarray, test_size: Union[int, float]) -> tuple:
    """Split time series into train and test sets.

    Utility function is used by the ARIMAForecaster 

    Args:
        time_series (array):
            The time series that should be normalized as np.ndarray with shape (n_samples, 1)

        test_size  (int or float):
            How many samples of the time series should be used as a hold-out test set. If float (should be between 0.0 and 1.0), the value represents the proportion of the dataset used as test samples. If int, the value represents the absolute number of test samples.

    Returns:
        A tuple of the form (np.ndarray, np.ndarray), where the training set is the first element with shape (n_train_samples,) and the testing set the second element with shape (n_test_samples). 

    Raises:
        ValueError: If test_size is out of range.
            
    """

    time_series_len = len(time_series)

    if test_size <= 0 or test_size >= time_series_len:
        raise ValueError(f'test_size must be greater than 0 but cannot exceed {time_series_len}')


    if test_size < 1:
        from_idx = int(np.ceil((time_series_len*(1-test_size))))
    else:
        from_idx = time_series_len - test_size
    
    train = time_series[:from_idx]
    test = time_series[from_idx:]

    return train, test


def plot_predictions(time_series: pd.DataFrame, predicted_data: np.ndarray, title: str, save_path: Path, mode: str = 'test', time_steps: int = 4):
    """Generates plots for training or test predictions and stores them on disk.

    Utility function that can be used by a forecaster.

    Args:
        time_series (DataFrame):
            The original time series to plot against as pd.DataFrame.

        predicted_data (array): 
            The predicted values as np.ndarray.

        title (str):  
            The title of the plot as string.

        save_path (Path):  
            The path to which the plot should be saved to as Path object.

        mode (str):
            Determines for which type of prediction to plot (train = training data prediction, test = test data prediction).

        time_steps (int): 
            Time steps (sequence length) of the data used for LSTM-based forecaster.  
    """

    plt.figure(figsize=(8, 6))
    
    # This mode is to plot the training data prediction for LSTMs
    if mode == 'train':
        prediction_dates = time_series.iloc[time_steps:].head(predicted_data.shape[0]).index
        data_to_plot = time_series.loc[prediction_dates]
        data_to_plot['Predicted Sales (Training)'] = predicted_data
        
        ax = sns.lineplot(data=data_to_plot, legend='full')
        ax.set_xlabel('Timesteps', labelpad=10, fontsize=12)
        ax.set_ylabel('Aggregated Sales', labelpad=10, fontsize=12)
        ax.tick_params(labelsize=9.5)
        ax.tick_params('x', labelrotation=90)
        ax.set_title(title, pad=15, fontsize=14)

        leg = ax.get_legend()
        leg.get_texts()[0].set_text('Actual Sales')
        
    elif mode == 'test':
        prediction_dates = time_series.tail(predicted_data.shape[0]).index
        data_to_plot = time_series.loc[prediction_dates]
        data_to_plot['Predicted Sales (Test)'] = predicted_data
        data_to_plot.index = data_to_plot.index.astype(str, copy = False)
        
        ax = sns.lineplot(data=data_to_plot, legend='full')
        ax.set_xlabel('Timesteps', labelpad=10, fontsize=12)
        ax.set_ylabel('Aggregated Sales', labelpad=10, fontsize=12)
        ax.tick_params(labelsize=9.5)
        ax.tick_params('x', labelrotation=90)
        ax.set_title(title, pad=15, fontsize=14)

        leg = ax.get_legend()
        leg.get_texts()[0].set_text('Actual Sales')

    if not save_path.suffix :
        save_path = str(save_path)+'.png'

    plt.savefig(save_path, bbox_inches='tight')


def plot_forecast(time_series: pd.DataFrame, forecast_series: pd.Series, title: str, save_path: Path, past_timesteps: int = 12):
    """Generates a plot for a forecast and stores them on disk.

    Utility function that can be used by a forecaster.

    Args:
        time_series (DataFrame):
            The original time series from which to take the last ``past_timesteps`` observations.

        forecast_series (array): 
            The forecasted values as pd.Series, where the index determines the future timesteps and the values the predicted sales. 

        title (str):  
            The title of the plot as string.

        save_path (Path):  
            The path to which the plot should be saved to as Path object.

        past_timesteps (int):
            Determines the number of timesteps from the original time series to include in the plot.
    """
    past_observations = time_series.iloc[-past_timesteps:]

    # Get custom x_ticks
    dates = past_observations.index.tolist()
    dates.extend(forecast_series.index.tolist())

    if len(dates) % 2 == 0:
        x_ticks = dates[::3]
    else:
        x_ticks = dates[::2]

    
    plt.figure(figsize=(8, 6))

    
    # Connection line between last observations and first forecast
    last_obeservation = past_observations.iloc[-1:]
    connection = pd.DataFrame(data=[forecast_series[0]], index=[forecast_series.index[0]], columns=[last_obeservation.columns[0]])
    last_obeservation = pd.concat([last_obeservation, connection])


    # Plot past observations and forecasts
    forecast_color = 'orange'

    sns.lineplot(data=past_observations)
    sns.lineplot(data=last_obeservation)
    ax = sns.lineplot(data=forecast_series, legend=False, color=forecast_color, markers=True)
    ax.set_xlabel('Timesteps', labelpad=10, fontsize=12)
    ax.set_ylabel('Aggregated Sales', labelpad=10, fontsize=12)
    ax.set_xticks(x_ticks)
    ax.tick_params(labelsize=9.5)
    ax.tick_params('x', labelrotation=90)
    ax.set_title(title, pad=15, fontsize=14)
    ax.lines[2].set_linestyle("--")
    
    handles, labl = ax.get_legend_handles_labels()
    ax.legend(handles, ['Past Sales', 'Forecasted Sales'])
    leg = ax.get_legend()
    leg.legendHandles[1].set_color(forecast_color)

    if not save_path.suffix :
        save_path = str(save_path)+'.png'

    plt.savefig(save_path, bbox_inches='tight')


class TimeSeriesReshaper():
    """ Helper class for LSTM Pipeline that reshapes the sequence data (n_samples, n_timesteps) to time series format (n_samples, 1)."""

    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X.reshape(-1, 1)


class LSTMReshaper():
    """ Helper class for LSTM Pipeline that reshapes the data to LSTM format sequence data (n_samples, n_timesteps, n_features)."""

    def __init__(self, n_timesteps, n_features):
        self.n_timesteps = n_timesteps
        self.n_features = n_features

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X.reshape(-1, self.n_timesteps, self.n_features)

