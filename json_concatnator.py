import os
import json

path = os.path.join(os.getcwd(), 'realtime_pred_res')
files = os.listdir(path)
files = [f for f in files if f.endswith('.json')]

jsons = []

for f in files:
    with open(os.path.join(path, f), 'r+') as file:
        try:
            content = json.load(file)
            global_time = content['Global_Time']
            start = content["based_timewindow"]["start"]
            end = content["based_timewindow"]["end"]
            pred = content["prediction"]
            jsons.append({"timewindow": {"start": start, "end": end}, "Global_Time": global_time, "prediction": pred})

        except:
            continue
        
with open(os.path.join(path, 'concatnated.json'), 'w') as file:
    json.dump(jsons, file)