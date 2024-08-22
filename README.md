1. Look for other paper that use NGSIM dataset to see how other models perform on it

2. distribute the training 

### How to run real-time prediction ###
1. run docker `docker run -p 9092:9092 apache/kafka:3.8.0`
2. run transmitter file: `python3 predict/transmitter.py`
3. in another terminal, run `python3 predict_main_1.py`
4. in another terminal, run `python3 predict_main_2.py`


experiment idea
different model complexity: 1 vs 2 features = 2 variants
different history length: 20, 40, 60 and skip 10, 20 = 6 variants
with or without ramp: 2 variants

for real-time analysis, try timewindow 10 sec, speed feature only, skip 1 history 6 = 1min,  

write the abstract part by the next meeting 