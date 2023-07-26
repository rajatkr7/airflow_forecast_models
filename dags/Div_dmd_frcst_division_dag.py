from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
from airflow import DAG
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor

from Sarima_division_level_sales_amount import sarima_model_division_sales_amount
from Arima_division_level_sales_amount import arima_model_division_sales_amount
from XGBOOST_division_level_sales_amount import xgboost_model_division_sales_amount
from RandomForest_division_level_sales_amount import random_forest_model_division_sales_amount
from Holt_Winter_division_sales_amount import holt_winter_model_division_sales_amount


default_args = {
    'owner': 'rajat',
    'retries': 5,
    'retry_delay': timedelta(minutes=10)
}
def start():
    print('start')
def end():
    print('end')

current_date = datetime.now().strftime('%Y%m%d')

with DAG(
    dag_id='DEMAND_FORECAST_DIVISION_LEVEL_DAG',
    start_date=datetime(2023, 7, 10),
    schedule_interval='@daily',
    default_args=default_args
) as dag:
    start = PythonOperator(dag=dag,
                           task_id='START',
                           python_callable=start
                           )

    actual_input_data = S3KeySensor(
        task_id='ACTUAL_DIVISION_LEVEL_DATA',
        bucket_name='anicca-demand-forecasting',
        bucket_key='actuals/division/sales_amount/divison_level_sales_amount.csv',
        aws_conn_id='s3',
        mode='poke',
        poke_interval=5,
        timeout=30
    )

    sarima_processed = PythonOperator(
        dag=dag,
        task_id='Sarima_division_level_sales_amount',
        # provide_context=False,
        python_callable=sarima_model_division_sales_amount
    )

    arima_processed = PythonOperator(
        dag=dag,
        task_id='Arima_division_level_sales_amount',
        # provide_context=False,
        python_callable=arima_model_division_sales_amount
    )
    xgboost_processed = PythonOperator(
        dag=dag,
        task_id='XGBoost_division_sales_amount',
        # provide_context=False,
        python_callable=xgboost_model_division_sales_amount
    )

    random_processed = PythonOperator(
        dag=dag,
        task_id='Random_Forest_division_sales_amount',
        # provide_context=False,
        python_callable=random_forest_model_division_sales_amount
    )

    holt_processed = PythonOperator(
        dag=dag,
        task_id='Holt_Winter_forest_division_sales_amount',
        # provide_context=False,
        python_callable=holt_winter_model_division_sales_amount
    )

    sarima_forecast = S3KeySensor(
        task_id='SARIMA_DIVISION_LEVEL_DATA',
        bucket_name='anicca-demand-forecasting',
        bucket_key='forecasted/test/'+current_date+'_sarima_division_level_sales_amount.csv',
        aws_conn_id='s3',
        mode='poke',
        poke_interval=5,
        timeout=30
    )

    arima_forecast = S3KeySensor(
        task_id='ARIMA_DIVISION_LEVEL_DATA',
        bucket_name='anicca-demand-forecasting',
        bucket_key='forecasted/test/'+current_date+'_arima_division_level_sales_amount.csv',
        aws_conn_id='s3',
        mode='poke',
        poke_interval=5,
        timeout=30
    )

    xgboost_forecast = S3KeySensor(
        task_id='XGBOOST_DIVISION_LEVEL_DATA',
        bucket_name='anicca-demand-forecasting',
        bucket_key='forecasted/test/'+current_date+'_xgboost_division_level_sales_amount.csv',
        aws_conn_id='s3',
        mode='poke',
        poke_interval=5,
        timeout=30
    )
    random_forecast = S3KeySensor(
        task_id='RANDOM_FOREST_DIVISION_LEVEL_DATA',
        bucket_name='anicca-demand-forecasting',
        bucket_key='forecasted/test/'+current_date+'_randomforest_division_level_sales_amount.csv',
        aws_conn_id='s3',
        mode='poke',
        poke_interval=5,
        timeout=30
    )
    holt_forecast = S3KeySensor(
        task_id='HOLTs_WINTER_DIVISION_LEVEL_DATA',
        bucket_name='anicca-demand-forecasting',
        bucket_key='forecasted/test/'+current_date+'_Holts_Winters_forecast_sales_amount.csv',
        aws_conn_id='s3',
        mode='poke',
        poke_interval=5,
        timeout=30
    )


    end = PythonOperator(dag=dag,
                           task_id='END',
                           python_callable=end
                           )


start >> actual_input_data >> sarima_processed >> sarima_forecast >>end
actual_input_data >> arima_processed >> arima_forecast >> end
actual_input_data >> random_processed >> random_forecast >> end
actual_input_data >> xgboost_processed >> xgboost_forecast >> end
actual_input_data >> holt_processed >> holt_forecast >> end