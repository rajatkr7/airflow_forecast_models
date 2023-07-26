import boto3
import pandas as pd
import numpy as np
from datetime import datetime

import random



def generate_result_df(df):
    from sklearn.ensemble import RandomForestRegressor

    # Get unique division names
    division_list = df['division'].unique()

    # Create an empty list to store the new DataFrames
    new_dfs = []

    # Iterate over each division
    for division_name in division_list:
        # Filter the DataFrame for the specific division
        division_df = df[df['division'] == division_name]
        print('Current Division is', division_name)

        # Get the minimum date for the division from the first available record
        first_record = division_df.iloc[0]
        year = first_record['YEAR']
        week = first_record['WEEK']

        # Find the start date based on the year and week
        start_date = datetime.strptime(f'{year}-W{week}-1', '%Y-W%W-%w')

        # Create a new DataFrame with the specified division and desired date range
        end_date = '2023-06-04'  # Updated end date to 24th week
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

        # Hyperparameter tuning for XGBoost
        # param_grid = {
        #     'n_estimators': [100, 200, 300],
        #     'max_features': ['auto', 'sqrt', 'log2'],
        #     'max_depth': [None, 3, 5, 7]
        # }

        # rf_model = RandomForestRegressor(random_state=0)
        # grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5)
        # grid_search.fit(np.arange(len(weekly_data)).reshape(-1, 1), weekly_data)

        best_params = {'max_depth': 3, 'max_features': 'auto', 'n_estimators': 300}

        # Best parameters from grid search
        # best_params = grid_search.best_params_
        # print('Best Parameters:', best_params)

        # Fit Random Forest model with best parameters
        rf_model = RandomForestRegressor(random_state=0, **best_params)
        rf_model.fit(np.arange(len(weekly_data)).reshape(-1, 1), weekly_data)

        # Adjust start and end dates to align with the weekly index
        start_date = '2023-01-08'
        end_date = weekly_data.index[21].strftime('%Y-%m-%d')  # Use the 24th index instead of the last index

        predicted_sales = rf_model.predict(np.arange(len(weekly_data), len(weekly_data) + 22).reshape(-1, 1))
        predicted_sales = np.where(predicted_sales < 0, 0, predicted_sales)

        # Create a new DataFrame for predicted sales with aligned index
        index_range = pd.date_range(start=start_date, periods=22, freq='W')

        predicted_df = pd.DataFrame({'Random_Forest': predicted_sales}, index=index_range)
        predicted_df['division'] = division_name
        predicted_df['YEAR'] = predicted_df.index.year.astype(int)
        predicted_df['WEEK'] = predicted_df.index.strftime('%W').astype(int)
        predicted_df['Random_Forest'] = predicted_df['Random_Forest'].astype(int)

        # Merge with merged_df to fill actual_sale_amount for predicted weeks
        predicted_df = pd.merge(predicted_df, merged_df, on=['YEAR', 'WEEK', 'division'], how='left')
        predicted_df['actual_sale_amount'] = predicted_df['actual_sale_amount'].ffill().fillna(0)
        predicted_df['actual_sale_amount'] = predicted_df['actual_sale_amount'].astype(int)  # Convert to integer
        predicted_df = predicted_df[['division', 'YEAR', 'WEEK', 'actual_sale_amount', 'Random_Forest']]

        # Append the new DataFrames to the list
        new_dfs.append(predicted_df)

    # Concatenate all the new DataFrames into a single DataFrame
    result_df = pd.concat(new_dfs, ignore_index=True)
    result_df.sort_values(['division', 'YEAR', 'WEEK'], inplace=True)

    return result_df

def calculate_corrected_forecast(df):
    # Get unique divisions
    divisions = df['division'].unique()
    forecast_columns = ['Random_Forest']
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
            division_data[column] = division_data[column].astype(int)

        # Update the data for the division in the original DataFrame
        df.update(division_data)

    # Drop Corrected Forecast columns for weeks beyond 20
    # df = df[df['WEEK'] <= 20]
    df.reset_index(drop=True, inplace=True)

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
        new_df = pd.DataFrame(columns=['division', 'YEAR', 'WEEK', 'actual_sale_amount', 'Random_Forest'])

        # Add rows to the new DataFrame for weeks 1 to 20
        for week in range(1, 23):
            week_data = division_data[division_data['WEEK'] == week].copy()
            week_data['WEEK'] = week
            new_df = pd.concat([new_df, week_data], ignore_index=True)

        # Add rows to the new DataFrame for weeks 21 to 30
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
        difference = division_df.loc[division_df['WEEK'] <= 22, 'actual_sale_amount'] - division_df.loc[division_df['WEEK'] <= 24, 'Random_Forest']

        # Calculate the average difference till week 24
        average_difference = difference.mean()

        # Add the average difference to every value of forecast1 till week 24
        rand_multiplier = random.uniform(0.7, 1.4)
        df.loc[(df['division'] == division) & (df['WEEK'] <= 22), 'Random_Forest'] += average_difference * rand_multiplier

        # Add average_difference multiplied by a random number between 0.6 and 1.6 after week 24
        rand_multiplier = random.uniform(0.4, 1.1)
        df.loc[(df['division'] == division) & (df['WEEK'] > 22), 'Random_Forest'] += average_difference * rand_multiplier

    df['Random_Forest'] = df['Random_Forest'].astype(int)
    df['YEAR'] = df['YEAR'].astype(int)
    return df





def random_forest_model_division_sales_amount():
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

    result_df = calculate_corrected_forecast(df)

    corrected_df = transform_data(result_df)

    output_df = update_forecast(corrected_df)



    randomforest_output_file_key = output_file_prefix + "/" + current_date + '_randomforest_division_level_sales_amount.csv'
    file_name = current_date + '_' + '_randomforest_division_level_sales_amount.csv'
    local_filepath = r"C:\airflow_pipeline_data\{file_name}"
    output_df.to_csv(local_filepath, index=False)

    # # Save the result to a CSV file
    # df.to_csv('division_forecast.csv', index=False)

    s3.upload_file(local_filepath, bucket_name, randomforest_output_file_key)

