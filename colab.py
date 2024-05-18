# -*- coding: utf-8 -*-
import subprocess
import shutil
import os
import sys
from itertools import islice
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
import torch
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.pandas import PandasDataset
import pandas as pd
import pickle
from data_prep import *

# Add the cloned repository to the system path
sys.path.append(os.path.abspath('./lag-llama'))

# Import the LagLlamaEstimator after adding the repository to the path
from lag_llama.gluon.estimator import LagLlamaEstimator


context_length = 950  # 600 minutes (10 hours)
prediction_length = 360  # 360 minutes (6 hours)
num_samples = 1  # Number of sample paths to generate


def initialize():
    #git_executable = r"C:\Program Files\Git\cmd\git.exe"  # Update this path based on your installation
    subprocess.run([git_executable, "clone", "https://github.com/time-series-foundation-models/lag-llama/"])
    # Install requirements
    subprocess.run(["pip", "install", "-r", "lag-llama/requirements.txt"])
    sys.path.append(os.path.abspath('./lag-llama'))  # Add the cloned repository to the system path
    subprocess.run(["huggingface-cli", "download", "time-series-foundation-models/Lag-Llama", "lag-llama.ckpt", "--local-dir", "lag-llama"])  # Download the model checkpoint
    subprocess.run([git_executable, "config", "--global", "user.name", "Andrew McCalip"])
    subprocess.run([git_executable, "config", "--global", "user.email", "Andrew McCalip"])

# Ensure the repository is cloned and path is added before importing LagLlamaEstimator
#initialize()
sys.path.append(os.path.abspath('./lag-llama'))  # Ensure the path is added
from lag_llama.gluon.estimator import LagLlamaEstimator
 

def get_lag_llama_predictions(dataset, prediction_length, context_length, num_samples, device="cpu", batch_size=64, nonnegative_pred_samples=True):
    ckpt = torch.load("lag-llama/lag-llama.ckpt", map_location=torch.device('cpu'))
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    estimator = LagLlamaEstimator(
        ckpt_path="lag-llama/lag-llama.ckpt",
        prediction_length=prediction_length,
        context_length=context_length,

        # estimator args
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],

        nonnegative_pred_samples=nonnegative_pred_samples,

        # linear positional encoding scaling
        rope_scaling={
            "type": "linear",
            "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
        },

        batch_size=batch_size,
        num_parallel_samples=num_samples,
    )

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=num_samples
    )
    forecasts = list(tqdm(forecast_it, total=len(dataset), desc="Forecasting batches"))
    tss = list(tqdm(ts_it, total=len(dataset), desc="Ground truth"))

    return forecasts, tss

def prepare_df():
        # Load or receive DataFrame from databento.py
    df = json_to_df('stock_data\es-6month-1min.json')  # Assuming json_to_df is imported from databento.py
    df_filtered = filter_prepare_and_plot_data(df)
    train_datasets,test_datasets = create_list_datasets(df_filtered)

    # Create and save metadata
    create_metadata()

    # Convert datasets to DataFrame for reporting
    train_df = dataset_to_dataframe(train_datasets)
    test_df = dataset_to_dataframe(test_datasets)

    # Report the number of unique days in train and test datasets
    num_train_days = train_df['date'].dt.date.nunique()
    num_test_days = test_df['date'].dt.date.nunique()
    print(f"Number of training days: {num_train_days}")
    print(f"Number of testing days: {num_test_days}")

    # Plot the train and test series on a new figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16))

    # Plot train data
    plot_prices(train_datasets, ax1, title='Normalized Price Time Series for Training Data')

    # Plot test data
    plot_prices(test_datasets, ax2, title='Normalized Price Time Series for Testing Data')

    plt.tight_layout()
    plt.show()

    datasets = create_train_datasets(train_datasets, test_datasets, freq="H", prediction_length=360)


    

def forcast(datasets):
   
    device = "cuda"  # Use GPU if available
    #device = "CPU"  # Use GPU if available
    #TSS is the time series. 
    forecasts, tss = get_lag_llama_predictions(
        datasets.test,
        prediction_length=datasets.metadata.prediction_length,
        num_samples=num_samples,
        context_length=context_length,
        device=device
    )
    print('Finished forecast prediction')
    return forecasts, tss

def plot_forcast():
    plt.figure(figsize=(20, 15))
    date_formater = mdates.DateFormatter('%H:%M')  # Format to display hours and minutes
    plt.rcParams.update({'font.size': 15})

    # Iterate through the first 9 series, and plot the predicted samples
    for idx, (forecast, ts) in islice(enumerate(zip(forecasts, tss)), 9):
        ax = plt.subplot(3, 3, idx+1)

        # Plot the ground truth
        ground_truth = ts[-4 * prediction_length:].to_timestamp()
        plt.plot(ground_truth, label="target")

        # Plot the forecast
        forecast.plot(color='g')

        # Format x-axis
        plt.xticks(rotation=60)
        ax.xaxis.set_major_formatter(date_formater)
        ax.set_title(forecast.item_id)

        # Autoscale based on the ground truth
        ax.relim()  # Recompute the limits based on the data
        ax.autoscale_view()  # Autoscale the view to the new limits

    plt.gcf().tight_layout()
    plt.legend()
    plt.show()

def finetune(datasets):
    print('Starting fine tuning')
    # Fine-tuning
    device = "cuda"  # Use GPU if available
    ckpt = torch.load("lag-llama/lag-llama.ckpt", map_location=device)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    estimator = LagLlamaEstimator(
            ckpt_path="lag-llama/lag-llama.ckpt",
            prediction_length=prediction_length,
            context_length=context_length,

            # distr_output="neg_bin",
            scaling="mean",
            nonnegative_pred_samples=True,
            aug_prob=0,
            lr=5e-4,

            # estimator args
            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            time_feat=estimator_args["time_feat"],

            # rope_scaling={
            #     "type": "linear",
            #     "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
            # },

            batch_size=64,
            num_parallel_samples=num_samples,
            trainer_kwargs = {"max_epochs": 2,}, # <- lightning trainer arguments
        )

    predictor = estimator.train(datasets.train, cache_data=True, shuffle_buffer_length=7000)
    print('Done fine tuning')
    forecast_it, ts_it = make_evaluation_predictions(
            dataset=datasets.test,
            predictor=predictor,
            num_samples=num_samples
        )

    print('Startin forecasting')
    forecasts = list(tqdm(forecast_it, total=len(datasets), desc="Forecasting batches"))
    print('Done forecasting')
    tss = list(tqdm(ts_it, total=len(datasets), desc="Ground truth"))

   

    evaluator = Evaluator()
    agg_metrics, ts_metrics = evaluator(iter(tss), iter(forecasts))

    print(agg_metrics)
    return forecasts,tss



if __name__ == "__main__":
    #initialize()
   forecasts = None
   tss = None

   import zipfile

   # Path to the zip file
   zip_file_path = 'pickle/datasets_10yr_imported_stock_data.zip'
   extract_to_path = 'pickle/'

   # Unzip the file
   with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
       zip_ref.extractall(extract_to_path)
   print(f"Unzipped {zip_file_path} to {extract_to_path}")

   with open('pickle/datasets_10yr_imported_stock_data.pkl', 'rb') as f:
    datasets = pickle.load(f)
       
   

    # ####Zero shot
    # forecasts, tss = forcast(datasets)  #executes prediction lag-llama model - untrained
    # # Save the forecasts and tss to a pickle file in a folder called pickle
    # os.makedirs('pickle', exist_ok=True)
    # with open('pickle/zeroshot_forecasts_tss.pkl', 'wb') as f:
    #     pickle.dump({'forecasts': forecasts, 'tss': tss}, f)
    #     print("Zero shot forecasts have been saved to 'pickle/forecasts_tss.pkl'")
  



    forecasts,tss = finetune(datasets)
    with open('pickle/tuned_forecasts_tss.pkl', 'wb') as f:
            pickle.dump({'forecasts': forecasts, 'tss': tss}, f)
            print("Fine tuned forecasts  to 'pickle/tuned_forecasts_tss.pkl'")


  