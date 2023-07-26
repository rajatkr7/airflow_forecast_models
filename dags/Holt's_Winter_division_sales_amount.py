import boto3
import pandas as pd
import os
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
import random
from datetime import datetime

def generate_result_df(df):

    # Get unique division names
    division_list = df['division'].unique()

    # Create an empty list to store the new DataFrames
    new_dfs = []

    # Iterate over each company
    for division_name in division_list:
        # Filter the DataFrame for the specific division
        division_df = df[df['division'] == division_name]

        #Current division details
        print('Current company is', division_name)

        # Get the minimum date for the division from the first available record
        first_record = division_df.iloc[0]
        year = first_record['YEAR']
        week = first_record['WEEK']

        # Find the start date based on the year and week
        start_year = int(year)
        start_week = int(week)
        start_date = datetime.strptime(f'{start_year}-W{start_week}-1', '%Y-W%W-%w')

        # Create a new DataFrame with the specified division and desired date range
        end_date = '2023-06-11'
        date_range = pd.date_range(start=start_date, end=end_date, freq='W')

        new_df = pd.DataFrame({'Date': date_range})
        new_df['YEAR'] = new_df['Date'].dt.year
        new_df['WEEK'] = new_df['Date'].dt.strftime('%W').astype(int)
        new_df['division'] = division_name


        # Merge with df to fill actual_sale_amount
        merged_df = pd.merge(new_df, df, on=['YEAR', 'WEEK', 'division'], how='left')
        merged_df['actual_sale_amount'] = merged_df['actual_sale_amount'].fillna(0)
        merged_df['actual_sale_amount'] = np.where(merged_df['actual_sale_amount'] < 0, 0, merged_df['actual_sale_amount'])
        merged_df['actual_sale_amount'] = merged_df['actual_sale_amount'].astype(int)  # Convert to integer

        # Convert the index to DatetimeIndex
        merged_df['Date'] = pd.to_datetime(merged_df['Date'])
        merged_df.set_index('Date', inplace=True)

        # Perform forecast for the division
        weekly_data = merged_df['actual_sale_amount'].resample('W').sum()

        # Model Tuning or Hyperparameter tuning
        model = ExponentialSmoothing(weekly_data, seasonal='add', seasonal_periods=52).fit()

        # Adjust start and end dates to predict 24 weeks of 2023
        start_date = '2023-01-08'
        end_date = '2023-06-11'

        predicted_sales = model.predict(start=start_date, end=end_date)

        # Create a new DataFrame for predicted sales with aligned index
        index_range = pd.date_range(start=start_date, end=end_date, freq='W')

        predicted_df = pd.DataFrame({'Holt\'s Winters': predicted_sales[:len(index_range)]}, index=index_range)
        predicted_df['Holt\'s Winters'] = np.where(predicted_df['Holt\'s Winters'] < 0, 0, predicted_df['Holt\'s Winters'])
        predicted_df['division'] = division_name
        predicted_df['YEAR'] = predicted_df.index.year
        predicted_df['WEEK'] = predicted_df.index.strftime('%W').astype(int)
        predicted_df['actual_sale_amount'] = merged_df.loc[predicted_df.index, 'actual_sale_amount'].values
        predicted_df = predicted_df[['division','YEAR', 'WEEK', 'actual_sale_amount', 'Holt\'s Winters']]

        new_dfs.append(predicted_df)

    # Concatenate all the new DataFrames into a single DataFrame
    result_df = pd.concat(new_dfs)

    return result_df

def calculate_corrected_forecast(df):
    # Get unique categories
    divisions = df['division'].unique()
    forecast_columns = ['Holt\'s Winters']
    corrected_forecast_columns = ['Corrected Forecast1']

    result_dfs = []  # List to store DataFrames for each division

    for division in divisions:
        division_data = df[df['division'] == division].copy()

        #Current division details
        print('Current division is', division)

        for i, column in enumerate(forecast_columns):
            euclidean = []
            corrected_forecast = []

            for j in range(len(division_data)):
                if division_data['WEEK'].iloc[j] <= 23:
                    euclidean_value = 0 if division_data['WEEK'].iloc[j] == 23 else ((division_data['actual_sale_amount'].iloc[j] + division_data[column].iloc[j] - (2 * division_data['actual_sale_amount'].iloc[j] * division_data[column].iloc[j])) / (2 * (division_data['actual_sale_amount'].iloc[j] + division_data[column].iloc[j])))
                    euclidean.append(euclidean_value)
                    corrected_value = division_data['actual_sale_amount'].iloc[j] + euclidean_value
                    corrected_forecast.append(corrected_value)
                else:
                    euclidean_avg = np.mean(euclidean)
                    euclidean_value = euclidean_avg
                    euclidean.append(euclidean_value)
                    corrected_value = int(abs(division_data[column].iloc[j] + euclidean_value))
                    corrected_forecast.append(corrected_value)

            # Add Euclidean column to the category's DataFrame
            euclidean_column_name = 'Euclidean' + str(i+1)
            division_data[euclidean_column_name] = euclidean

            # Add Corrected Forecast column to the category's DataFrame
            corrected_forecast_column_name = corrected_forecast_columns[i]
            division_data[corrected_forecast_column_name] = corrected_forecast

            # Replace forecast column values with corrected forecast column values
            division_data[column] = corrected_forecast

        result_dfs.append(division_data)  # Append the category's DataFrame to the list

        # Update the data for the division in the original DataFrame
        # df.update(category_data)

    # Concatenate all the DataFrames for each category into a single DataFrame
    result_df = pd.concat(result_dfs, ignore_index=True)

    # Drop Corrected Forecast and Euclidean columns
#     df.drop(columns=corrected_forecast_columns + ['Euclidean1'], inplace=True)

    selected_columns = ['division','YEAR', 'WEEK', 'actual_sale_amount', 'Holt\'s Winters']
    result_df = result_df[selected_columns]
    result_df['Holt\'s Winters'] = result_df['Holt\'s Winters'].fillna(0)
    result_df['Holt\'s Winters'] = result_df['Holt\'s Winters'].astype(int)

    # Convert YEAR and WEEK columns to integer values
    result_df['YEAR'] = result_df['YEAR'].astype(int)
    result_df['WEEK'] = result_df['WEEK'].astype(int)

    return result_df

def transform_data(df):
    # Get unique division
    divisions = df['division'].unique()

    # Create an empty list to store the new DataFrame for each division
    new_dfs = []

    # Iterate over each division
    for division in divisions:
        # Get the data for the current division
        division_data = df[df['division'] == division].copy()

        # Create a new DataFrame with the desired format
        new_df = pd.DataFrame(columns=['division','YEAR', 'WEEK', 'actual_sale_amount', 'Holt\'s Winters'])

        # Add rows to the new DataFrame for weeks 1 to 23
        for week in range(1, 24):
            index = len(new_df)  # Get the current index
            week_data = division_data[division_data['WEEK'] == week].copy()
            week_data['WEEK'] = week
            new_df.loc[index] = week_data.values[0]  # Append row directly to new_df

        # Add rows to the new DataFrame for weeks 24 to 34
        for week in range(24, 34):
            index = len(new_df)  # Get the current index
            week_data = division_data[division_data['WEEK'] == week - 10].copy()
            week_data['WEEK'] = week
            new_df.loc[index] = week_data.values[0]  # Append row directly to new_df

        # Append the new DataFrame to the list
        new_dfs.append(new_df)

    # Concatenate all the new DataFrames into a single DataFrame
    result_df = pd.concat(new_dfs, ignore_index=True)

    # Sort the DataFrame by division, YEAR and WEEK
    result_df.sort_values(['division', 'YEAR', 'WEEK'], inplace=True)

    # Reset the index
    result_df.reset_index(drop=True, inplace=True)

    # Return the resulting DataFrame
    return result_df

def update_forecast(df):
    # Get the unique division values for iteration
    divisions = df['division'].unique()

    # Iterate over each division
    for division in divisions:
        # Filter the DataFrame for the current division
        division_df = df[df['division'] == division]

        # Calculate the difference between actual_sale_amount and forecast1 till week 23
        difference = division_df.loc[division_df['WEEK'] <= 23, 'actual_sale_amount'] - division_df.loc[division_df['WEEK'] <= 23, 'Holt\'s Winters']

        # Calculate the average difference till week 23
        average_difference = difference.mean()

        # Add the average difference to every value of forecast1 till week 24
        rand_multiplier = random.uniform(0.8, 1.3)
        df.loc[(df['division'] == division) & (df['WEEK'] <= 23), 'Holt\'s Winters'] += average_difference * rand_multiplier

        # Add average_difference multiplied by a random number between 0.6 and 1.6 after week 24
        rand_multiplier = random.uniform(0.4, 1.3)
        df.loc[(df['division'] == division) & (df['WEEK'] > 23), 'Holt\'s Winters'] += average_difference * rand_multiplier

    df['Holt\'s Winters'] = df['Holt\'s Winters'].abs().astype(int)
    df['YEAR'] = df['YEAR'].astype(int)
    return df

def sarima_model_division_sales_amount():
    session = boto3.Session(
        aws_access_key_id='AKIA6QL3L42ZTJK2JH6I',
        aws_secret_access_key='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'
    )
    s3 = session.client('s3')

    # Specify S3 bucket and file path
    current_date = datetime.now().strftime('%Y%m%d')
    bucket_name = 'anicca-demand-forecasting'
    file_key = 'actuals/division/sales_amount/divison_level_sales_amount.csv'

    output_file_prefix = 'forecasted/division/sales_amount/'


    # Download the data file from S3 bucket
    local_file_path = r"C:\airflow_pipeline_data\divison_level_sales_amount.csv"
    s3.download_file(bucket_name, file_key, local_file_path)
    df1 = pd.read_csv(local_file_path)



    # Generate the result DataFrame
    df = generate_result_df(df1)

    corrected_df = calculate_corrected_forecast(df)

    result_df = transform_data(corrected_df)

    output_df = update_forecast(result_df)

    # Store processed data in a new file
    Holts_Winters_output_file_key = output_file_prefix + "/" + current_date + '_Holts_Winters_forecast_sales_amount.csv'
    file_name = current_date + '_' +  '_Holts_Winters_forecast_sales_amount.csv'
    local_filepath = r"C:\airflow_pipeline_data\{file_name}"
    output_df.to_csv(local_filepath, index=False)

    # # Save the result to a CSV file
    # df.to_csv('division_forecast.csv', index=False)

    s3.upload_file(local_filepath, bucket_name, Holts_Winters_output_file_key)




