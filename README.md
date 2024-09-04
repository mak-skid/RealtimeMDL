This is the code for my MSc project "Real-time prediction of lane-level traffic speeds using Mixed Deep Learning model with Apache Spark and Kafka".
The code is written by Makoto Ono, unless specified.

### How to run real-time prediction ###
1. run docker `docker run -p 9092:9092 apache/kafka:3.8.0`
2. run transmitter file: `python3 transmitter.py`
3. in another terminal, run `python3 realtime_aggregator_1.py`
4. in another terminal, run `python3 realtime_aggregator_2.py`