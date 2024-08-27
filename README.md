1. Look for other paper that use NGSIM dataset to see how other models perform on it

2. distribute the training 

### How to run real-time prediction ###
1. run docker `docker run -p 9092:9092 apache/kafka:3.8.0`
2. run transmitter file: `python3 predict/transmitter.py`
3. in another terminal, run `python3 predict_main_1.py`
4. in another terminal, run `python3 predict_main_2.py`

"""
experiment idea
different model complexity: 1 vs 2 features = 2 variants
different history length: 20, 40, 60 and skip 10, 20 = 6 variants
with or without ramp: 2 variants
"""

for real-time analysis, try timewindow 10 sec, speed feature only, skip 1 history 6 = 1min,  

write the abstract part by the next meeting 

skip = 20sec_1: 159044 + 99319 + 97778 + 98032 + 97655 + 79762
skip = 10sec_2: 160019 + 106916 + 96316 + 97978 + 7155

in literature review, include macroscopic (explain what it is) as well as microscopic
Create my own fig for Figure 3.6: The comparison between (a) RNN and (b) LSTM structures
Figure 3.7: Inner structure of ConvLSTM.
Figure 3.8: The structure of the orginal MDL model. From Lu et al (for this one, create my own or focus on differences between downsized model)
Remove altogether Figure 3.10: Throughput result comparison and scalability of Spark Structured Streaming
on the Yahoo! benchmark. From Armbrust et al.