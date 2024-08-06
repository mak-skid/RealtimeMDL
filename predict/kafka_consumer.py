from kafka import KafkaConsumer

consumer = KafkaConsumer(bootstrap_servers='localhost:9092', group_id=None, auto_offset_reset='earliest')
consumer.subscribe(['test'])
for message in consumer:
    print("received")
    msg = message.value.decode('utf-8')
    print(msg)
consumer.close()
