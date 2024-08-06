import json
import time
import pandas
from kafka import KafkaProducer

producer = KafkaProducer(bootstrap_servers='localhost:9092')

pdf = pandas.read_csv('predict/test_data.csv', header=0)

elapsed_time = 1118849209200
print("Started sending test data")
while elapsed_time <= 1118849752200:
    start = time.time()
    
    datapoints = pdf.loc[pdf['Global_Time'] == elapsed_time]
    datapoints_json = datapoints.to_json(orient='records')

    key = str(elapsed_time).encode('utf-8')
    data = datapoints_json.encode('utf-8')

    producer.send('us101', key=key, value=data)
    producer.flush()

    end = time.time()
    duration = 0.1-(end-start) if end-start < 0.1 else 0
    time.sleep(duration)
    elapsed_time += 100

print("All test data sent")
producer.close()