import pandas as pd
import json
from datetime import datetime, time
import pytz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy.spatial.distance import euclidean



def json_to_df(file_path):
    # Path to your JSON file
    #file_path = '/content/stock_data/es-6month-1min.json'   #for colab
    #file_path = '/content/stock_data/es-10yr-1min.json'   #for colab
    #file_path = 'stock_data/es-10yr-1min.json'   #for locla 
    #file_path = 'website/stock_data/spy-1yr-1min.json'
    
    # List to hold all JSON objects
    data = []

    # Open the file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Ensure the line is not empty
                json_object = json.loads(line)
                data.append(json_object)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Rescale the 'open' and 'close' prices by 100x
    

    # Convert 'ts_event' to datetime and extract date and time
    df['ts_event'] = pd.to_datetime(df['hd'].apply(lambda x: int(x['ts_event']) / 1e9), unit='s')

    # Convert to Eastern Time Zone
    eastern = pytz.timezone('US/Eastern')
    df['ts_event'] = df['ts_event'].dt.tz_localize('UTC').dt.tz_convert(eastern)

    df['date'] = df['ts_event'].dt.date
    df['time'] = df['ts_event'].dt.time
    
    # Filter for symbols containing 'ES'
    df = df[df['symbol'].str.contains('ES')]

    # Convert 'open' and 'close' to numeric
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')

    # Filter for a minimum open price of 2000
    #df = df[df['open'] >= 2000]
    
    print(df)
    # Calculate the average price from 'open' and 'close'
    df['price'] = (df['open'] + df['close']) / 2 
    

    # Initialize an empty DataFrame to hold the filtered rows
    filtered_df = pd.DataFrame()

    # Loop through each day
    for date, group in df.groupby('date'):
        # Identify the first occurring symbol for the day
        first_symbol = group.sort_values(by='ts_event').iloc[0]['symbol']
        # Filter the group to keep only rows with the first occurring symbol
        filtered_group = group[group['symbol'] == first_symbol]
        # Append the filtered group to the filtered_df
        filtered_df = pd.concat([filtered_df, filtered_group])

    df = filtered_df.reset_index(drop=True)

    # Normalize prices based on the price at 9:30 AM
    normalized_prices = []
    for date, group in df.groupby('date'):
        base_price = group[group['time'] == time(9, 30)]['price'].values
        if base_price.size > 0:
            normalized_price = group['price'] / base_price[0]
            normalized_prices.extend(normalized_price)
        else:
            print(f"No 9:30 AM price found for {date}. Using first available price for normalization.")
            first_price = group['price'].iloc[0]
            normalized_price = group['price'] / first_price
            normalized_prices.extend(normalized_price)

    # Add normalized prices to the DataFrame
    df['normalized_price'] = normalized_prices
    # Replace NaNs in the 'normalized_price' with 1
    df['normalized_price'] = df['normalized_price'].fillna(1)
    # Normalize the 'normalized_price' column to 100x
    df['normalized_price'] = df['normalized_price'] * 100
    # Drop the 'hd' column as it's no longer needed
    df.drop(columns=['hd'], inplace=True)
    ###  done for DF
    print('JSON to DF:')
    print(df)

    # Report the number of unique days in the DataFrame
    num_days = df['date'].nunique()
    print(f'The dataset contains data for {num_days} unique days.')
    return df
#########  PLotting    ######

def plot_every_day(df):
    plt.figure(figsize=(15, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(df['date'].unique())))

    print('starting to plot')
    # Create a reference date for all times to be plotted on the same x-axis
    reference_date = datetime(2000, 1, 1)  # Arbitrary non-leap year date
    max_days_to_plot = 5000  # Define the maximum number of days to plot
    for i, (date, group) in enumerate(df.groupby('date')):
        if i >= max_days_to_plot:
            break
        # Normalize prices based on the price at 9:30 AM
        base_price = group[group['time'] == time(9, 30)]['price'].values
        if base_price.size > 0:
            normalized_prices = group['price'] / base_price[0]
            # Convert 'time' from datetime.time to matplotlib dates for plotting
            # Use a fixed date to align all times on the same x-axis
            times = [mdates.date2num(datetime.combine(reference_date, t)) for t in group['time']]
            
            # Plotting
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



def predict_plot(df):
 

    num_days_to_keep = 2

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





if __name__ == "__main__":
    df = json_to_df('stock_data/es-6month-1min.json')  #takes the flat json from databento and converts to df 
    #print(df)
    
    #plot_every_day(df)
  
    #predict_plot(df)  #run the euclid prediction 
