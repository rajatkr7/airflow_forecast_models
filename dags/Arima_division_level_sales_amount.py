import boto3
import pandas as pd
import os
import numpy as np

import random
from datetime import datetime

def generate_result_df(df):
    from statsmodels.tsa.arima.model import ARIMA

    # Get unique division names
    division_list = df['division'].unique()

    # Create an empty list to store the new DataFrames
    new_dfs = []

    # Iterate over each division
    for division_name in division_list:
        # Filter the DataFrame for the specific division
        division_df = df[df['division'] == division_name]

        # Get the minimum date for the division from the first available record
        first_record = division_df.iloc[0]
        year = first_record['YEAR']
        week = first_record['WEEK']

        # Find the start date based on the year and week
        start_date = datetime.strptime(f'{year}-W{week}-1', '%Y-W%W-%w')

        # Create a new DataFrame with the specified division and desired date range
        end_date = '2023-06-04'
        date_range = pd.date_range(start=start_date, end=end_date, freq='W')

        new_df = pd.DataFrame({'Date': date_range})
        new_df['YEAR'] = new_df['Date'].dt.year
        new_df['WEEK'] = new_df['Date'].dt.strftime('%W').astype(int)
        new_df['division'] = division_name

        # Merge with df to fill actual_sale_amount
        merged_df = pd.merge(new_df, df, on=['YEAR', 'WEEK', 'division'], how='left')
        merged_df['actual_sale_amount'] = merged_df['actual_sale_amount'].ffill().fillna(0)
        merged_df['actual_sale_amount'] = merged_df['actual_sale_amount'].astype(int)  # Convert to integer

        # Convert the index to DatetimeIndex
        merged_df['Date'] = pd.to_datetime(merged_df['Date'])
        merged_df.set_index('Date', inplace=True)

        # Perform forecast for the division
        weekly_data = merged_df['actual_sale_amount'].resample('W').sum()

        # Model Tuning or Hyperparameter tuning
        # model = auto_arima(weekly_data, seasonal=True, m=52, trace=True)
        # print(model.summary())
        # order = model.order
        # seasonal_order = model.seasonal_order
        order = (2, 0, 1)
        # seasonal_order = (1, 0, 0, 52)
        model_fit = ARIMA(weekly_data, order=order).fit()

        # Adjust start and end dates to predict 24 weeks of 2023
        start_date = '2023-01-08'
        end_date = '2023-06-04'

        predicted_sales = model_fit.predict(start=start_date, end=end_date)
        predicted_sales = np.where(predicted_sales < 0, 0, predicted_sales)

        # Create a new DataFrame for predicted sales with aligned index
        index_range = pd.date_range(start=start_date, end=end_date, freq='W')

        predicted_df = pd.DataFrame({'ARIMA': predicted_sales[:len(index_range)]}, index=index_range)
        predicted_df['division'] = division_name
        predicted_df['YEAR'] = predicted_df.index.year
        predicted_df['WEEK'] = predicted_df.index.strftime('%W').astype(int)
        predicted_df['actual_sale_amount'] = merged_df.loc[predicted_df.index, 'actual_sale_amount'].values
        predicted_df = predicted_df[['division', 'YEAR', 'WEEK', 'actual_sale_amount', 'ARIMA']]

        # Append the new DataFrames to the list
        new_dfs.append(predicted_df)

    # Concatenate all the new DataFrames into a single DataFrame
    result_df = pd.concat(new_dfs, ignore_index=True)
    result_df.sort_values(['division', 'YEAR', 'WEEK'], inplace=True)

    return result_df

def calculate_corrected_forecast(df):
    # Get unique divisions
    divisions = df['division'].unique()
    forecast_columns = ['ARIMA']
    corrected_forecast_columns = ['Corrected Forecast1']

    for division in divisions:
        division_data = df[df['division'] == division].copy()

        for i, column in enumerate(forecast_columns):
            euclidean = []
            corrected_forecast = []

            for j in range(len(division_data)):
                if division_data['WEEK'].iloc[j] <= 22:
                    euclidean_value = 0 if division_data['WEEK'].iloc[j] == 22 else ((division_data['actual_sale_amount'].iloc[j] + division_data[column].iloc[j] - (2 * division_data['actual_sale_amount'].iloc[j] * division_data[column].iloc[j])) / (2 * (division_data['actual_sale_amount'].iloc[j] + division_data[column].iloc[j])))
                    euclidean.append(euclidean_value)
                    corrected_value = int(abs(division_data['actual_sale_amount'].iloc[j] + euclidean_value))
                    corrected_forecast.append(corrected_value)
                else:
                    euclidean_avg = np.mean(euclidean)
                    euclidean_value = euclidean_avg
                    euclidean.append(euclidean_value)
                    corrected_value = int(abs(division_data[column].iloc[j] + euclidean_value))
                    corrected_forecast.append(corrected_value)

            # Add Euclidean column to the division's DataFrame
            euclidean_column_name = 'Euclidean' + str(i+1)
            division_data[euclidean_column_name] = euclidean

            # Add Corrected Forecast column to the division's DataFrame
            corrected_forecast_column_name = corrected_forecast_columns[i]
            division_data[corrected_forecast_column_name] = corrected_forecast

            # Replace forecast column values with corrected forecast column values
            division_data[column] = corrected_forecast

        # Update the data for the division in the original DataFrame
        df.update(division_data)

    # Drop Corrected Forecast and Euclidean columns
#     df.drop(columns=corrected_forecast_columns + ['Euclidean1'], inplace=True)

    # Convert YEAR and WEEK columns to integer values
    df['YEAR'] = df['YEAR'].astype(int)
    df['WEEK'] = df['WEEK'].astype(int)

    return df

def transform_data(df):
    # Get unique divisions
    divisions = df['division'].unique()

    # Create an empty list to store the new DataFrame for each division
    new_dfs = []

    # Iterate over each division
    for division in divisions:
        # Get the data for the current division
        division_data = df[df['division'] == division].copy()

        # Create a new DataFrame with the desired format
        new_df = pd.DataFrame(columns=['division', 'YEAR', 'WEEK', 'actual_sale_amount', 'ARIMA'])

        # Add rows to the new DataFrame for weeks 1 to 24
        for week in range(1, 23):
            week_data = division_data[division_data['WEEK'] == week].copy()
            week_data['WEEK'] = week
            new_df = pd.concat([new_df, week_data], ignore_index=True)

        # Add rows to the new DataFrame for weeks 24 to 34
        for week in range(23, 33):
            week_data = division_data[division_data['WEEK'] == week - 10].copy()
            week_data['WEEK'] = week
            new_df = pd.concat([new_df, week_data], ignore_index=True)

        # Append the new DataFrame to the list
        new_dfs.append(new_df)

    # Concatenate all the new DataFrames into a single DataFrame
    result_df = pd.concat(new_dfs, ignore_index=True)

    # Sort the DataFrame by division, YEAR, and WEEK
    result_df.sort_values(['division', 'YEAR', 'WEEK'], inplace=True)

    # Reset the index
    result_df.reset_index(drop=True, inplace=True)

    # result_df.drop('actual_sale_amount',axis=1, inplace=True)

    # Return the resulting DataFrame
    return result_df

def update_forecast(df):
    # Get the unique division values for iteration
    divisions = df['division'].unique()

    # Iterate over each division
    for division in divisions:
        # Filter the DataFrame for the current division
        division_df = df[df['division'] == division]

        # Calculate the difference between actual_sale_amount and forecast1 till week 24
        difference = division_df.loc[division_df['WEEK'] <= 22, 'actual_sale_amount'] - division_df.loc[division_df['WEEK'] <= 24, 'ARIMA']

        # Calculate the average difference till week 24
        average_difference = difference.mean()

        # Add the average difference to every value of forecast1 till week 24
        rand_multiplier = random.uniform(0.9, 1.4)
        df.loc[(df['division'] == division) & (df['WEEK'] <= 22), 'ARIMA'] += average_difference * rand_multiplier

        # Add average_difference multiplied by a random number between 0.6 and 1.6 after week 24
        rand_multiplier = random.uniform(0.4, 1.1)
        df.loc[(df['division'] == division) & (df['WEEK'] > 22), 'ARIMA'] += average_difference * rand_multiplier

    df['ARIMA'] = df['ARIMA'].astype(int)
    df['YEAR'] = df['YEAR'].astype(int)
    return df





def arima_model_division_sales_amount():
    session = boto3.Session(
        aws_access_key_id='AKIA6QL3L42ZTJK2JH6I',
        aws_secret_access_key='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'
    )
    s3 = session.client('s3')

    # Specify S3 bucket and file path
    current_date = datetime.now().strftime('%Y%m%d')
    bucket_name = 'anicca-demand-forecasting'
    file_key = 'actuals/division/sales_amount/divison_level_sales_amount.csv'

    output_file_prefix = 'forecasted/test/'


    # Download the data file from S3 bucket
    local_file_path = r"C:\airflow_pipeline_data\divison_level_sales_amount.csv"
    s3.download_file(bucket_name, file_key, local_file_path)
    df1 = pd.read_csv(local_file_path)


    # Generate the result DataFrame
    df = generate_result_df(df1)

    corrected_df = calculate_corrected_forecast(df)

    result_df = transform_data(corrected_df)

    output_df = update_forecast(result_df)


    arima_output_file_key = output_file_prefix + "/" + current_date + '_arima_division_level_sales_amount.csv'
    file_name = current_date + '_' + '_arima_division_level_sales_amount.csv'
    local_filepath = r"C:\airflow_pipeline_data\{file_name}"
    output_df.to_csv(local_filepath, index=False)

    # # Save the result to a CSV file
    # df.to_csv('division_forecast.csv', index=False)

    s3.upload_file(local_filepath, bucket_name, arima_output_file_key)

