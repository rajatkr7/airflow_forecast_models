from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
from airflow import DAG
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from sample import process_test1
default_args = {
    'owner': 'rajat',
    'retries': 5,
    'retry_delay': timedelta(minutes=10)
}
def start():
    print('start')
def end():
    print('end')
with DAG(
    dag_id='Data_Processing_DAG',
    start_date=datetime(2023, 7, 25),
    schedule_interval='@daily',
    default_args=default_args
) as dag:
    start = PythonOperator(dag=dag,
                           task_id='start',
                           python_callable=start
                           )
    task2 = PythonOperator(
        dag=dag,
        task_id='test_process',
        #provide_context=False,
        python_callable=process_test1
        )
    task1 = S3KeySensor(
        task_id='read_file',
        bucket_name='anicca-executive-summary',
        bucket_key='test/abc.csv',
        aws_conn_id='s3',
        mode='poke',
        poke_interval=5,
        timeout=30
    )
    task3 = S3KeySensor(
        task_id='write_file',
        bucket_name='anicca-executive-summary',
        bucket_key='test/cdef.csv',
        aws_conn_id='s3',
        mode='poke',
        poke_interval=5,
        timeout=30
    )
    end = PythonOperator(dag=dag,
                           task_id='end',
                           python_callable=end
                           )

start >> task1 >> task2 >> task3 >> end