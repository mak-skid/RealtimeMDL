import json
import time
import pandas
from kafka import KafkaProducer

def transmitter():
    """
    Function to send test data to Kafka.

    Author: Makoto Ono
    """

    producer = KafkaProducer(bootstrap_servers='localhost:9092')

    pdf = pandas.read_csv('dataset/test_data.csv', header=0)

    elapsed_time = 1118849209200
    print("Started sending test data")
    while elapsed_time <= 1118849752200:
        start = time.time()
        
        datapoints = pdf.loc[pdf['Global_Time'] == elapsed_time]
        datapoints_json = datapoints.to_json(orient='records')
        
        data = datapoints_json.encode('utf-8')

        producer.send('us101', value=data)
        producer.flush()

        end = time.time()
        duration = 0.1-(end-start) if end-start < 0.1 else 0
        time.sleep(duration)
        elapsed_time += 100

    print("All test data sent")
    producer.close()

transmitter()