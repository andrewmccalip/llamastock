import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from itertools import islice
import pandas as pd
import random
import numpy as np

# Global variables to store forecasts and time series data
global forecasts, tss

def load_forecasts(file_path):
    """
    Load forecast and time series data from a pickle file.

    Parameters:
    file_path (str): Path to the pickle file containing the data.

    Returns:
    tuple: A tuple containing forecasts and time series data.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    forecasts = data['forecasts']
    tss = data['tss']

    print(f"Number of series in the forecasts dataset: {len(forecasts)}")
    return forecasts, tss

def debug_forecasts_tss(forecasts, tss):
    """
    Print debug information about the forecasts and time series data.

    Parameters:
    forecasts (list): List of forecast objects.
    tss (list): List of time series data.
    """
    print(f"Number of series in forecasts: {len(forecasts)}")
    print(f"Number of series in tss: {len(tss)}")
    
    for i, (forecast, ts) in enumerate(zip(forecasts, tss)):
        print(f"\nSeries {i}:")
        print(f"  Forecast length: {len(forecast.mean)}")
        print(f"  Time series length: {len(ts)}")

def smooth_series(series, window_size):
    """
    Apply a rolling mean to smooth the series.

    Parameters:
    series (pd.Series): The time series data to be smoothed.
    window_size (int): The window size for the rolling mean.

    Returns:
    pd.Series: The smoothed time series.
    """
    return series.rolling(window=window_size, min_periods=1).mean()


def plot_time_series(forecasts, tss, context_length, prediction_length, max_samples):
    """
    Plot the time series data along with forecasts.

    Parameters:
    forecasts (list): List of forecast objects.
    tss (list): List of time series data.
    context_length (int): Length of the context window.
    prediction_length (int): Length of the prediction window.
    max_samples (int): Maximum number of samples to plot.
    """
    plt.figure(figsize=(20, 15))
    date_formatter = mdates.DateFormatter('%H:%M')  # Format to display hours and minutes
    plt.rcParams.update({'font.size': 15, 'xtick.labelsize': 10, 'ytick.labelsize': 10})

    # Ensure max_samples does not exceed the number of available series
    max_samples = min(max_samples, len(forecasts), len(tss))

    # Randomly select indices for the samples
    random_indices = random.sample(range(len(forecasts)), max_samples)

    # Calculate the number of rows and columns for the subplots
    n_cols = 3
    n_rows = (max_samples + n_cols - 1) // n_cols  # This ensures enough rows to fit max_samples subplots

    for idx, random_idx in enumerate(random_indices):
        forecast = forecasts[random_idx]
        ts = tss[random_idx]
        ax = plt.subplot(n_rows, n_cols, idx + 1)  # Adjust subplot grid size based on max_samples

        # Convert PeriodIndex to Timestamp
        ts = ts.to_timestamp()

        # Print debug information about the series
        print(f"\nSeries {random_idx}:")
        print(f"  Total length of time series: {len(ts)}")
        print(f"  Forecast length: {len(forecast.mean)}")

        # Adjust context and prediction lengths if the series is too short
        total_length = len(ts)
        if total_length < context_length + prediction_length:
            print(f"Adjusting lengths for series {random_idx} because it is too short.")
            context_length = total_length // 2
            prediction_length = total_length - context_length

        # Calculate indices for context and prediction
        context_end_idx = ts.index[-prediction_length - 1]
        context_start_idx = context_end_idx - pd.Timedelta(minutes=context_length - 1)
        ground_truth_start_idx = context_end_idx + pd.Timedelta(minutes=1)
        ground_truth_end_idx = ground_truth_start_idx + pd.Timedelta(minutes=prediction_length - 1)

        # Extract context and ground truth series
        context_series = ts[context_start_idx:context_end_idx]
        ground_truth_series = ts[ground_truth_start_idx:ground_truth_end_idx]

        # Create forecast series
        forecast_start_idx = ground_truth_start_idx
        forecast_end_idx = forecast_start_idx + pd.Timedelta(minutes=len(forecast.mean) - 1)
        forecast_index = pd.date_range(start=forecast_start_idx, end=forecast_end_idx, freq='T')
        forecast_series = pd.Series(forecast.mean, index=forecast_index)

        # Plot the context window
        ax.plot(context_series.index, context_series, color='blue', label="context")

        # Plot the ground truth
        ax.plot(ground_truth_series.index, ground_truth_series, color='red', label="ground truth")

    
        # Plot the forecast samples
        forecast_samples = forecasts[random_idx].samples.real
        mean_forecast = forecasts[random_idx].median

        # Generate a color map for the forecast samples
        color_map = plt.cm.get_cmap('viridis', len(forecast_samples))

        # Apply smoothing to the forecast samples
        smoothed_lower_bound = pd.Series(forecast_samples[0], index=forecast_index).rolling(window=10, min_periods=1).mean()
        smoothed_upper_bound = pd.Series(forecast_samples[-1], index=forecast_index).rolling(window=10, min_periods=1).mean()

        # Plot the forecast samples as a shaded area
        ax.fill_between(forecast_index, smoothed_lower_bound, smoothed_upper_bound, color='gray', alpha=0.3, label="forecast range")

        # Plot the mean forecast as a dashed line
        #ax.plot(forecast_index, mean_forecast, color='green', linestyle='--', label="forecast mean")

        # Combine context and ground truth to set y-axis limits
        combined_series = pd.concat([context_series, ground_truth_series])
        y_min, y_max = combined_series.min(), combined_series.max()
        y_range = y_max - y_min
        y_padding = y_range * 0.1
        ax.set_ylim((y_min - y_padding).item(), (y_max + y_padding).item())

        # Set x-axis limits to match the exact range of the context and forecast periods
        ax.set_xlim(context_series.index[0], forecast_series.index[-1])

        # Format the x-axis to show hours and minutes
        ax.xaxis.set_major_formatter(date_formatter)

        # Add a legend
        ax.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()       
        
if __name__ == "__main__":
    # Load and plot fine-tuned predictions
    forecasts, tss = load_forecasts('pickle/tuned_forecasts_tss.pkl')
    plot_time_series(forecasts, tss, context_length=960, prediction_length=360, max_samples=6)
    print('done')

