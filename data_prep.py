# Import necessary libraries
from gluonts.dataset.common import ListDataset  # For creating GluonTS datasets
import json  # For handling JSON data
import numpy as np  # For numerical operations
from pathlib import Path  # For handling file paths
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For plotting
import matplotlib.dates as mdates  # For handling dates in plots
from datetime import datetime, time  # For date and time operations
import pytz  # For timezone handling
import matplotlib  # For plotting configurations
from gluonts.dataset.common import TrainDatasets, MetaData, CategoricalFeatureInfo  # For GluonTS dataset metadata
import os  # For operating system interactions
import pickle  # For serializing objects
from scipy.spatial.distance import euclidean  # For calculating Euclidean distance
from concurrent.futures import ProcessPoolExecutor  # For parallel processing
import zipfile  # Add this import at the beginning of your script


# Define the prediction length
prediction_length = 120

# Function to filter, prepare data, and plot
def filter_prepare_and_plot_data(df):
    # Combine 'date' and 'time' into a single datetime column
    df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
    
    # Set 'datetime' as the index and filter to only include times from 12:01 AM to 5:00 PM
    df = df.set_index('datetime').between_time('00:01', '16:00').reset_index()
    
    # Normalize prices based on the price at 9:30 AM
    normalized_prices = []
    for date, group in df.groupby(df['datetime'].dt.date):
        base_price = group[group['datetime'].dt.time == time(9, 30)]['normalized_price'].values
        if base_price.size > 0:
            normalized_price = group['normalized_price'] / base_price[0]
            normalized_prices.extend(normalized_price)
        else:
            normalized_prices.extend([np.nan] * len(group))  # Append NaN if no base price at 9:30
    
    # Add normalized prices to the DataFrame and replace NaNs with 1
    df['normalized_price'] = pd.Series(normalized_prices).fillna(1)
    
    # Set 'time' as the index
    df['time'] = df['datetime'].dt.time
    df = df.set_index('time')
    
    # Plotting function
    plt.figure(figsize=(15, 8))  # Set the figure size
    colors = plt.cm.viridis(np.linspace(0, 1, len(df['date'].unique())))  # Generate colors for each unique date
    reference_date = datetime(2000, 1, 1)  # Arbitrary non-leap year date

    for i, (date, group) in enumerate(df.groupby(df['date'])):
        base_price = group[group.index == time(9, 30)]['normalized_price'].values
        if base_price.size > 0:
            normalized_prices = group['normalized_price'] / base_price[0]
            times = [mdates.date2num(datetime.combine(reference_date, t)) for t in group.index]
            plt.plot(times, normalized_prices, label=f'{date}', color=colors[i])

    # Formatting the plot
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Set major ticks to every hour
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Format x-axis labels as hours and minutes
    plt.gcf().autofmt_xdate()  # Rotate x-axis labels for better readability
    plt.title('Normalized Price Time Series for Each Day')  # Set plot title
    plt.xlabel('Time of Day')  # Set x-axis label
    plt.ylabel('Normalized Price')  # Set y-axis label
    plt.legend(title='Date')  # Add legend with title
    plt.show()  # Display the plot
    
    return df

# Function to create ListDataset
def create_list_datasets(df, freq='T', train_ratio=0.6, val_ratio=0.2):
    """
    Create ListDataset objects for training, validation, and testing.

    Args:
        df (pd.DataFrame): The input DataFrame containing the time series data.
        freq (str): The frequency of the time series data.
        train_ratio (float): The ratio of data to be used for training.
        val_ratio (float): The ratio of data to be used for validation.

    Returns:
        tuple: A tuple containing ListDataset objects for training, validation, and testing.
    """
    train_datasets = []  # List to hold training datasets
    val_datasets = []  # List to hold validation datasets
    test_datasets = []  # List to hold testing datasets

    # Group by the 'datetime' column's date part
    df['date'] = df['datetime'].dt.date
    unique_dates = df['date'].unique()  # Get unique dates

    # Determine the split indices
    train_split_index = int(len(unique_dates) * train_ratio)
    val_split_index = int(len(unique_dates) * (train_ratio + val_ratio))

    # Split the dates into training, validation, and testing sets
    train_dates = unique_dates[:train_split_index]
    val_dates = unique_dates[train_split_index:val_split_index]
    test_dates = unique_dates[val_split_index:]

    # Process training data
    for date in train_dates:
        group = df[df['date'] == date]
        group = group.sort_values(by='datetime')  # Sort by datetime

        # Create a dictionary for the training dataset
        train_ds = {'target': group['normalized_price'].values, 'start': str(group['datetime'].iloc[0]), 'item_id': str(date)}
        train_datasets.append(train_ds)

    # Process validation data
    for date in val_dates:
        group = df[df['date'] == date]
        group = group.sort_values(by='datetime')  # Sort by datetime

        # Create a dictionary for the validation dataset
        val_ds = {'target': group['normalized_price'].values, 'start': str(group['datetime'].iloc[0]), 'item_id': str(date)}
        val_datasets.append(val_ds)

    # Process testing data
    for date in test_dates:
        group = df[df['date'] == date]
        group = group.sort_values(by='datetime')  # Sort by datetime

        # Create a dictionary for the testing dataset
        test_ds = {'target': group['normalized_price'].values, 'start': str(group['datetime'].iloc[0]), 'item_id': str(date)}
        test_datasets.append(test_ds)

    # Report the number of unique days in train, val, and test datasets
    num_train_days = len(train_dates)
    num_val_days = len(val_dates)
    num_test_days = len(test_dates)
    print(f"Number of training days: {num_train_days}")
    print(f"Number of validation days: {num_val_days}")
    print(f"Number of testing days: {num_test_days}")

    return ListDataset(train_datasets, freq=freq), ListDataset(val_datasets, freq=freq), ListDataset(test_datasets, freq=freq)


# Function to convert dataset to DataFrame
def dataset_to_dataframe(dataset):
    df_list = []  # List to hold DataFrames
    for entry in dataset:
        # Explicitly convert start date from Period to Timestamp if necessary
        start_date = entry['start']
        if isinstance(start_date, pd.Period):
            start_date = start_date.to_timestamp()
        elif isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        
        # Create a DataFrame for the series
        series_df = pd.DataFrame({
            'date': pd.date_range(start=start_date, periods=len(entry['target']), freq='T'),
            'value': entry['target'],
            'series_id': entry['item_id']
        })
        df_list.append(series_df)
    return pd.concat(df_list, ignore_index=True)  # Concatenate all DataFrames

# Function to plot price time series for each day
def dataset_plot(dataset, ax, title):
    df = dataset_to_dataframe(dataset)  # Convert dataset to DataFrame
    unique_dates = df['date'].dt.date.unique()  # Get unique dates
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_dates)))  # Generate colors for each unique date

    # Create a reference date for all times to be plotted on the same x-axis
    reference_date = datetime(2000, 1, 1)  # Arbitrary non-leap year date

    for i, (date, group) in enumerate(df.groupby(df['date'].dt.date)):
        # Convert 'time' from datetime.time to matplotlib dates for plotting
        # Use a fixed date to align all times on the same x-axis
        times = [mdates.date2num(datetime.combine(reference_date, t.time())) for t in group['date']]
        
        # Plotting
        ax.plot(times, group['value'], label=f'{date}', color=colors[i % len(colors)])

    # Formatting the plot
    ax.set_xlim([mdates.date2num(datetime.combine(reference_date, datetime.min.time())) + 1/1440,  # 12:01 AM
                 mdates.date2num(datetime.combine(reference_date, datetime.min.time())) + 15/24])  # 3:00 PM
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Set major ticks to every hour
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Format x-axis labels as hours and minutes
    ax.set_title(title)  # Set plot title
    ax.set_xlabel('Time of Day')  # Set x-axis label
    ax.set_ylabel('Price')  # Set y-axis label
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")  # Rotate x-axis labels for better readability

# Function to create training datasets
def create_train_datasets(train_dataset, val_dataset, test_dataset, freq, prediction_length):
    """
    Create TrainDatasets object including training, validation, and testing datasets.

    Args:
        train_dataset (ListDataset): The training dataset.
        val_dataset (ListDataset): The validation dataset.
        test_dataset (ListDataset): The testing dataset.
        freq (str): The frequency of the time series data.
        prediction_length (int): The prediction length.

    Returns:
        TrainDatasets: The TrainDatasets object containing metadata and datasets.
    """
    metadata = MetaData(
        freq=freq,
        feat_static_cat=[CategoricalFeatureInfo(name='feat_static_cat_0', cardinality='1')],
        feat_static_real=[],
        feat_dynamic_real=[],
        feat_dynamic_cat=[],
        prediction_length=prediction_length
    )
    
    datasets = TrainDatasets(
        metadata=metadata,
        train=train_dataset,
        test=test_dataset,
        
    )
    
    return datasets

# Function to create metadata
def create_metadata():
    freq = "T"  # Frequency of the data
    path = Path("data")  # Path to save metadata
    
    # Ensure the directory exists
    path.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "freq": freq,
        "prediction_length": prediction_length,
        "feat_static_cat": [
            {
                "name": "feat_static_cat",
                "cardinality": "1"
            }
        ],
        "feat_static_real": [],
        "feat_dynamic_real": [],
        "feat_dynamic_cat": [],
        "train": "train",
        "test": "test"
    }
    
    # Save metadata to a JSON file
    with open(path / "metadata.json", "w") as f:
        json.dump(metadata, f)

# Function to convert JSON to DataFrame
def json_to_df(file_path):
    verbose = True  # Verbose flag for printing progress
    if verbose:
        print(f"Reading JSON file from: {file_path}")
    
    # List to hold all JSON objects
    data = []

    # Open the file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Ensure the line is not empty
                json_object = json.loads(line)
                data.append(json_object)
    
    if verbose:
        print(f"Loaded {len(data)} records from the JSON file.")

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Convert 'ts_event' to datetime and extract date and time
    df['ts_event'] = pd.to_datetime(df['hd'].apply(lambda x: int(x['ts_event']) / 1e9), unit='s')

    # Convert to Eastern Time Zone
    eastern = pytz.timezone('US/Eastern')
    df['ts_event'] = df['ts_event'].dt.tz_localize('UTC').dt.tz_convert(eastern)

    df['date'] = df['ts_event'].dt.date
    df['time'] = df['ts_event'].dt.time
    
    # Filter for symbols containing 'ES'
    df = df[df['symbol'].str.contains('ES')]

    if verbose:
        print(f"Filtered DataFrame to {len(df)} records containing 'ES' symbol.")

    # Convert 'open' and 'close' to numeric
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')

    # Calculate the average price from 'open' and 'close'
    df['price'] = (df['open'] + df['close']) / 2 

    # Initialize an empty DataFrame to hold the filtered rows
    filtered_df = pd.DataFrame()

    # Loop through each day
    total_days = df['date'].nunique()
    for i, (date, group) in enumerate(df.groupby('date'), 1):
        # Identify the first occurring symbol for the day
        first_symbol = group.sort_values(by='ts_event').iloc[0]['symbol']
        # Filter the group to keep only rows with the first occurring symbol
        filtered_group = group[group['symbol'] == first_symbol]
        # Append the filtered group to the filtered_df
        filtered_df = pd.concat([filtered_df, filtered_group])
        # Print progress
        print(f"Processed {i}/{total_days} days")
    
    df = filtered_df.reset_index(drop=True)

    if verbose:
        print(f"Filtered DataFrame to {len(df)} records after keeping only the first occurring symbol for each day.")

    # Identify the closing price at 4 PM EST for each day
    df['close_4pm'] = df[df['time'] == time(16, 0)]['price']

    # Forward fill the 4 PM closing price to the next day's rows
    df['prev_close_4pm'] = df['close_4pm'].shift(1).ffill()

    # Normalize prices based on the previous day's 4 PM closing price
    df['normalized_price'] = df['price'] / df['prev_close_4pm']

    # Replace NaNs in the 'normalized_price' with 1
    df['normalized_price'] = df['normalized_price'].fillna(1)
    # Normalize the 'normalized_price' column to 100x
    df['normalized_price'] = df['normalized_price'] 
    # Drop the 'hd' column as it's no longer needed
    df.drop(columns=['hd', 'close_4pm'], inplace=True)
    # Drop any group of dates that don't have a valid prev_close_4pm value
    df = df.dropna(subset=['prev_close_4pm'])

    # Drop any day groups that don't have at least 800 rows
    valid_dates = df.groupby('date').filter(lambda x: len(x) >= 800).date.unique()
    df = df[df['date'].isin(valid_dates)]

    if verbose:
        print(f"Final DataFrame contains {len(df)} records after dropping days with insufficient data.")
        print(f"The dataset contains data for {df['date'].nunique()} unique days.")

    return df

# Function to plot DataFrame
def df_plot(df):
    plt.figure(figsize=(15, 8))  # Set the figure size
    colors = plt.cm.viridis(np.linspace(0, 1, len(df['date'].unique())))  # Generate colors for each unique date

    print('starting to plot')
    # Create a reference date for all times to be plotted on the same x-axis
    reference_date = datetime(2000, 1, 1)  # Arbitrary non-leap year date
    max_days_to_plot = 10  # Define the maximum number of days to plot
    for i, (date, group) in enumerate(df.groupby('date')):
        if i >= max_days_to_plot:
            break
        # Convert 'time' from datetime.time to matplotlib dates for plotting
        # Use a fixed date to align all times on the same x-axis
        times = [mdates.date2num(datetime.combine(reference_date, t)) for t in group['time']]
        
        # Plotting the normalized_price column directly
        plt.plot(times, group['normalized_price'], label=f'{date}', color=colors[i])

    # Formatting the plot
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Set major ticks to every hour
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Format x-axis labels as hours and minutes
    plt.gcf().autofmt_xdate()  # Rotate x-axis labels for better readability
    plt.title('Normalized Price Time Series for Each Day')  # Set plot title
    plt.xlabel('Time of Day')  # Set x-axis label
    plt.ylabel('Normalized Price')  # Set y-axis label
    plt.show()  # Display the plot

# Function to plot predictions
def predict_plot(df):
    num_days_to_keep = 2  # Number of similar days to keep

    # Define the time range for comparison and full day plotting
    end_time = time(9, 45)
    full_day_end_time = time(17, 0)  # Define end of day as 5 PM

    # Normalize prices for each day within the specified time range up to 9:45 AM
    normalized_prices_by_day = {}
    full_day_prices_by_day = {}  # To store prices for the full day
    for date, group in df.groupby('date'):
        period = group[group['time'] <= end_time]
        full_day_period = group[group['time'] <= full_day_end_time]  # Full day data
        base_price = period[period['time'] == time(9, 30)]['price'].values
        if base_price.size == 0:
            # If no data is available at 9:30 for any day, skip that day
            continue
        normalized_prices = period['price'] / base_price[0]
        full_day_normalized_prices = full_day_period['price'] / base_price[0]  # Normalize full day prices
        normalized_prices_by_day[date] = normalized_prices.values
        full_day_prices_by_day[date] = full_day_normalized_prices.values  # Store full day normalized prices

    # Select the sample day's normalized prices using ts_event
    sample_date = pd.Timestamp('2024-03-19').date()

    if sample_date not in normalized_prices_by_day:
        # If no data is available at 9:30 for the sample day, handle the case
        print("No data available at 9:30 for sample day.")
        return
    sample_normalized_prices = normalized_prices_by_day[sample_date]
    sample_full_day_prices = full_day_prices_by_day[sample_date]  # Full day prices for sample day

    # Calculate similarity with other days based on the morning data
    similarity_scores = {}
    for date, normalized_prices in normalized_prices_by_day.items():
        if date != sample_date:
            # Calculate similarity score (e.g., using Euclidean distance)
            if len(normalized_prices) == len(sample_normalized_prices):
                distance = np.linalg.norm(sample_normalized_prices - normalized_prices)
                similarity_scores[date] = distance

    similar_days = sorted(similarity_scores, key=similarity_scores.get)[:num_days_to_keep]
    # Define a reference date for plotting
    reference_date = datetime(2000, 1, 1)  # Arbitrary fixed date for time alignment in plots

    # Set up the plot with proper time of the day x-axis
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # Plot the sample day for the full day
    sample_times = [mdates.date2num(datetime.combine(reference_date, t)) for t in df[df['date'] == sample_date]['time'] if t <= full_day_end_time]
    ax.plot(sample_times, sample_full_day_prices, label='Sample Day (04-18)', color='red', linewidth=2.5)

    # Plot the similar days with different dash types and transparency for the full day
    dash_styles = ['-', '--', '-.', ':']
    for i, date in enumerate(similar_days):
        day_data = df[df['date'] == date]
        full_day_period = day_data[day_data['time'] <= full_day_end_time]
        times = [mdates.date2num(datetime.combine(reference_date, t)) for t in full_day_period['time']]
        ax.plot(times, full_day_prices_by_day[date], label=f'Similar Day {date.strftime("%m-%d")}', color='gray', linewidth=1.5, linestyle=dash_styles[i % len(dash_styles)], alpha=0.5)

    # Add a vertical marker at 9:45 AM on the x-axis
    marker_time = mdates.date2num(datetime.combine(reference_date, time(9, 45)))
    ax.axvline(x=marker_time, color='blue', linestyle='--', linewidth=1, label='9:45 AM')

    ax.legend()
    plt.gcf().autofmt_xdate()  # Rotate date labels for better readability
    plt.show()


import numpy as np

from scipy import signal

def fake_wave_data():
    """
    Generate a DataFrame with various waveforms (sine, square, sawtooth) for a specific date range.

    Returns:
        pd.DataFrame: DataFrame containing datetime, date, time, and normalized_price columns.
    """
    # Generate a date range for the timestamps (150 days, 1-minute frequency)
    date_range = pd.date_range(start='2024-03-01 00:00', end='2024-06-28 23:59', freq='T')
    
    # Generate wave data with random amplitude, phase, and period alterations
    period = len(date_range)
    amplitude = 0.03 + 0.001 * np.random.randn()  # Randomly alter the amplitude by 0.001
    phase = np.random.uniform(0, 2 * np.pi)  # Random phase shift
    period_factor = np.random.uniform(0.5, 0.7)  # Random period alteration within 50% to 70%
    
    # Generate different waveforms
    t = np.arange(period)
    waveforms = {
        'sine': lambda t: 1.0 + amplitude * np.sin(2 * np.pi * (1/120) * period_factor * t + phase),
        'square': lambda t: 1.0 + amplitude * signal.square(2 * np.pi * (1/60) * period_factor * t + phase),
        'sawtooth': lambda t: 1.0 + amplitude * signal.sawtooth(2 * np.pi * (1/180) * period_factor * t + phase),
        'low_period_sine': lambda t: 1.0 + amplitude * np.sin(2 * np.pi * (1/1000) * t + phase)  # Very low period sine wave
    }
    
    # Create a DataFrame with the generated data
    df_filtered = pd.DataFrame({
        'datetime': date_range
    })
    
    # Add 'date' and 'time' columns
    df_filtered['date'] = df_filtered['datetime'].dt.date
    df_filtered['time'] = df_filtered['datetime'].dt.time
    
    # Assign a random waveform to each day
    unique_dates = df_filtered['date'].unique()
    normalized_prices = np.array([])
    
    for date in unique_dates:
        wave_type = np.random.choice(list(waveforms.keys()))
        day_indices = df_filtered[df_filtered['date'] == date].index
        day_wave = waveforms[wave_type](t[:len(day_indices)])
        normalized_prices = np.concatenate((normalized_prices, day_wave))
    
    # Inject 5 percent noise into the wave data
    noise = 0.0015 * np.random.randn(period)
    normalized_prices += noise
    
    df_filtered['normalized_price'] = normalized_prices
    
    print()
    return df_filtered

# Main execution
if __name__ == "__main__":
    matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting

    # Define the file path variable
    #json_file_path = 'stock_data/es-6month-1min.json'
    json_file_path = 'stock_data_ignored\es-10yr-1min.json'
    #json_file_path = 'stock_data/fake_waves.json'
    # Load or receive DataFrame from databento.py
    df = json_to_df(json_file_path)



    #df = fake_wave_data()  #unsupress for sine wave data a
    
    
    df_filtered = filter_prepare_and_plot_data(df)
   
    # Create ListDataset objects for training, validation, and testing
    train_datasets, val_datasets, test_datasets = create_list_datasets(df_filtered)

    # Create and save metadata
    create_metadata()

    # Convert datasets to DataFrame for reporting
    train_df = dataset_to_dataframe(train_datasets)
    val_df = dataset_to_dataframe(val_datasets)
    test_df = dataset_to_dataframe(test_datasets)

    # Report the number of unique days in train, val, and test datasets
    num_train_days = train_df['date'].dt.date.nunique()
    num_val_days = val_df['date'].dt.date.nunique()
    num_test_days = test_df['date'].dt.date.nunique()
    print(f"Number of training days: {num_train_days}")
    print(f"Number of validation days: {num_val_days}")
    print(f"Number of testing days: {num_test_days}")

    # Plot the train, val, and test series on a new figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 24))
    dataset_plot(train_datasets, ax1, title='Normalized Price Time Series for Training Data')
    dataset_plot(val_datasets, ax2, title='Normalized Price Time Series for Validation Data')
    dataset_plot(test_datasets, ax3, title='Normalized Price Time Series for Testing Data')
    plt.show()

    # Create training datasets
    datasets = create_train_datasets(train_datasets, val_datasets, test_datasets, freq="H", prediction_length=prediction_length)

    # Save the datasets to a pickle file
    os.makedirs('pickle', exist_ok=True)
    pickle_file_name = os.path.splitext(os.path.basename(json_file_path))[0] + '.pkl'
    pickle_file_path = f'pickle/{pickle_file_name}'
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(datasets, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Datasets have been saved to '{pickle_file_path}'")

    # Zip the pickle file
    zip_file_path = f'pickle/{os.path.splitext(pickle_file_name)[0]}.zip'
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(pickle_file_path, os.path.basename(pickle_file_path))
        print(f"Pickle file zipped to '{zip_file_path}'")