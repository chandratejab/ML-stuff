"""
Code that goes along with the Airflow located at:
http://airflow.readthedocs.org/en/latest/tutorial.html
"""
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2018, 7, 9),
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    # 'schedule_interval': '@daily',
    # 'queue': 'bash_queue',
    #'pool': 'backfill',
    # 'priority_weight': 10,
    'end_date': datetime(2018,12,31),
}

dag = DAG(
    'MyFW', default_args=default_args)

# t1, t2 and t3 are examples of tasks created by instantiating operators#
# execute = BashOperator(
#     task_id='execute',
#     bash_command='date',
#     #bash_command='cd /opt/projects/DigitalServiceReporting && git fetch --all && git reset --hard origin/dev',
#     dag=dag)


data_pull = BashOperator(
    task_id='data_pull',
    bash_command='python /usr/src/app/src/data/data_pull.py',
    dag=dag)

data_preprocess = BashOperator(
    task_id='data_preprocess',
    bash_command='python /usr/src/app/src/data/data_preprocess.py',
    dag=dag)

train_model = BashOperator(
    task_id='train_model',
    bash_command='python /usr/src/app/src/process/train_model.py',
    dag=dag)

test_model = BashOperator(
    task_id='test_model',
    bash_command='python /usr/src/app/src/tests/test_model.py',
    dag=dag)
# import pdb;pdb.set_trace()
# data_pull.set_upstream(execute)
data_preprocess.set_upstream(data_pull)
train_model.set_upstream(data_preprocess)
test_model.set_upstream(train_model)
print("lol")
