import pandas as pd
import json
from datetime import datetime, time
import pytz
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy.spatial.distance import euclidean



if __name__ == "__main__":
    df = json_to_df('stock_data/es-6month-1min.json')  #takes the flat json from databento and converts to df 
    #print(df)
    
    plot_every_day(df)
  
    #predict_plot(df)  #run the euclid prediction 
