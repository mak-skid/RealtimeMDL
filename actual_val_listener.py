import time
from kafka import KafkaConsumer
import json

def json_creator():
    consumer = KafkaConsumer('us101_a_agg', bootstrap_servers='localhost:9092')
    
    ls = []
    start = time.time()
    i = 0
    for message in consumer:
        if time.time() - start > 1000:
            with open("act_val.json", "w") as f:
                json.dump(ls, f)
                break
        json_k = message.key.decode('utf-8')
        global_time = json.loads(json_k)["Global_Time"]
        timewindow = json.loads(json_k)["col1"]
        start_timewindow = timewindow["timewindow"]["start"]
        end_timewindow = timewindow["timewindow"]["end"]
        json_v = message.value.decode('utf-8')
        ls.append({"timewindow": {"start": start_timewindow, "end": end_timewindow}, "Global_Time": global_time, "actual": json_v})
        print(ls[i])
        i += 1

json_creator()