# Import necessary libraries
from gluonts.dataset.common import ListDataset
import json
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, time
from databento import json_to_df  # Assuming json_to_df is imported from databento.py
from gluonts.dataset.common import TrainDatasets, MetaData, CategoricalFeatureInfo


prediction_length = 32

# Function to filter, prepare data, and plot
def filter_prepare_and_plot_data(df):
    # Combine 'date' and 'time' into a single datetime column
    df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
    
    # Set 'datetime' as the index and filter to only include times from 12:01 AM to 5:00 PM
    df = df.set_index('datetime').between_time('00:01', '17:00').reset_index()
    
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
    plt.figure(figsize=(15, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(df['date'].unique())))
    reference_date = datetime(2000, 1, 1)  # Arbitrary non-leap year date

    for i, (date, group) in enumerate(df.groupby(df['date'])):
        base_price = group[group.index == time(9, 30)]['normalized_price'].values
        if base_price.size > 0:
            normalized_prices = group['normalized_price'] / base_price[0]
            times = [mdates.date2num(datetime.combine(reference_date, t)) for t in group.index]
            plt.plot(times, normalized_prices, label=f'{date}', color=colors[i])

    # Formatting the plot
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gcf().autofmt_xdate()  # Rotation
    plt.title('Normalized Price Time Series for Each Day')
    plt.xlabel('Time of Day')
    plt.ylabel('Normalized Price')
    plt.legend(title='Date')
    plt.show()
    
    return df

# Function to create ListDataset
def create_list_datasets(df, prediction_length, freq='T'):
    train_datasets = []
    test_datasets = []

    # Group by the 'datetime' column's date part
    df['date'] = df['datetime'].dt.date
    unique_dates = df['date'].unique()
    
    # Determine the split index
    split_index = int(len(unique_dates) * 0.7)
    
    # Split the dates into training and testing sets
    train_dates = unique_dates[:split_index]
    test_dates = unique_dates[split_index:]
    
    # Process training data
    for date in train_dates:
        group = df[df['date'] == date]
        group = group.sort_values(by='datetime')
        
        # Print out the series and the corresponding day
        #print(f"Training Series for {date}:")
        #print(group[['normalized_price']])
        
        train_ds = {'target': group['normalized_price'].values, 'start': str(group['datetime'].iloc[0]), 'item_id': str(date)}
        train_datasets.append(train_ds)
    
    # Process testing data
    for date in test_dates:
        group = df[df['date'] == date]
        group = group.sort_values(by='datetime')
        
        # Print out the series and the corresponding day
        #print(f"Testing Series for {date}:")
        #print(group[['normalized_price']])
        
        test_ds = {'target': group['normalized_price'].values, 'start': str(group['datetime'].iloc[0]), 'item_id': str(date)}
        test_datasets.append(test_ds)
    # Report the number of unique days in train and test datasets
    num_train_days = len(train_dates)
    num_test_days = len(test_dates)
    print(f"Number of training days: {num_train_days}")
    print(f"Number of testing days: {num_test_days}")

    return ListDataset(train_datasets, freq=freq), ListDataset(test_datasets, freq=freq)

# Function to generate data
def generate_data(df, prediction_length):
    df_prepared = filter_prepare_and_plot_data(df)
    return create_list_datasets(df_prepared, prediction_length)



# Function to convert dataset to DataFrame
def dataset_to_dataframe(dataset):
    df_list = []
    for entry in dataset:
        # Explicitly convert start date from Period to Timestamp if necessary
        start_date = entry['start']
        if isinstance(start_date, pd.Period):
            start_date = start_date.to_timestamp()
        elif isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        
        series_df = pd.DataFrame({
            'date': pd.date_range(start=start_date, periods=len(entry['target']), freq='T'),
            'value': entry['target'],
            'series_id': entry['item_id']
        })
        df_list.append(series_df)
    return pd.concat(df_list, ignore_index=True)


# Function to plot normalized price time series for each day
def plot_normalized(dataset, ax, title):
    df = dataset_to_dataframe(dataset)
    colors = plt.cm.viridis(np.linspace(0, 1, len(df['date'].unique())))

    # Create a reference date for all times to be plotted on the same x-axis
    reference_date = datetime(2000, 1, 1)  # Arbitrary non-leap year date

    for i, (date, group) in enumerate(df.groupby(df['date'].dt.date)):
        # Normalize prices based on the price at 9:30 AM
        base_price = group[group['date'].dt.time == time(9, 30)]['value'].values
        if base_price.size > 0:
            normalized_prices = group['value'] / base_price[0]
            # Convert 'time' from datetime.time to matplotlib dates for plotting
            # Use a fixed date to align all times on the same x-axis
            times = [mdates.date2num(datetime.combine(reference_date, t.time())) for t in group['date']]
            
            # Plotting
            ax.plot(times, normalized_prices, label=f'{date}', color=colors[i])

    # Formatting the plot
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_title(title)
    ax.set_xlabel('Time of Day')
    ax.set_ylabel('Normalized Price')
    ax.legend(title='Date')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

def create_train_datasets(train_dataset, test_dataset, freq, prediction_length):
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
        test=test_dataset
    )
    
    return datasets

# Function to create metadata
def create_metadata():
    freq = "T"
    path = Path("data")
    
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
    
    with open(path / "metadata.json", "w") as f:
        json.dump(metadata, f)

# Main execution
if __name__ == "__main__":
    # Load or receive DataFrame from databento.py
    df = json_to_df()  # Assuming json_to_df is imported from databento.py
    train_dataset, test_dataset = generate_data(df, prediction_length=32)

    # Create and save metadata
    create_metadata()

    # Convert datasets to DataFrame for reporting
    train_df = dataset_to_dataframe(train_dataset)
    test_df = dataset_to_dataframe(test_dataset)

    # Report the number of unique days in train and test datasets
    num_train_days = train_df['date'].dt.date.nunique()
    num_test_days = test_df['date'].dt.date.nunique()
    print(f"Number of training days: {num_train_days}")
    print(f"Number of testing days: {num_test_days}")

    # Plot the train and test series on a new figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16))

    # Plot train data
    plot_normalized(train_dataset, ax1, title='Normalized Price Time Series for Training Data')

    # Plot test data
    plot_normalized(test_dataset, ax2, title='Normalized Price Time Series for Testing Data')

    plt.tight_layout()
    plt.show()


    datasets = create_train_datasets(train_dataset, test_dataset, freq="H", prediction_length=300)