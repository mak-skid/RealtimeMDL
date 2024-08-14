1. Look for other paper that use NGSIM dataset to see how other models perform on it

2. distribute the training 

### How to run real-time prediction ###
1. run docker `docker run -p 9092:9092 apache/kafka:3.8.0`
2. run transmitter file: `python3 predict/transmitter.py`
3. in another terminal, run `python3 predict_main_1.py`
4. in another terminal, run `python3 predict_main_2.py`
