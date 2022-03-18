from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS
from datetime import datetime
import argparse
import json
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins='*')
import random
import time
from threading import Thread, Event, RLock
from stock_metadata import company_and_ticks
from kafka import KafkaConsumer


x = 0
consumer = None
stop_flag = Event()


def server_side_event():
    """ Function to publish server side event """
    while True:
        if consumer is None:
            time.sleep(20)
        for msg in consumer:
            msg_json = json.loads(msg.value.decode('utf-8'))
            print(msg_json)
            portfolio_value= {}
            transaction_values = []
            for key, value in msg_json.items():
                if key == "portfolio":
                    portfolio_value["y"] = value
                elif key == "timestamp":
                    portfolio_value["x"] = datetime.fromtimestamp(value).strftime("%m/%d/%Y, %H:%M:%S")
                else:
                    for key_stock, value_stock in msg_json['stock_deltas'].items():
                        if value_stock == 0:
                            continue
                        transaction = dict()
                        transaction["tick"] = key_stock
                        if value_stock < 0:
                            transaction["operation"] = "SELL"
                        else:
                            transaction["operation"] = "BUY"
                        transaction["qty"] = abs(value_stock)
                        transaction_values.append(transaction)
            socketio.emit('portfolio', portfolio_value)
            socketio.emit('transaction', transaction_values)


@socketio.on('connect')
def continuous_thread():
    global stop_flag


@socketio.on('disconnect')
def stop_thread():
    global stop_flag
    stop_flag.set()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream tweets to stdout or kafka topic')
    parser.add_argument('topic', metavar='<topic_name>', help='Kafka topic name')
    parser.add_argument('hosts', nargs='+', metavar='<hosts>', help='Space separated list of Hostname:port of bootstrap servers')
    args = parser.parse_args()
    thread_start = Thread(target=server_side_event)
    thread_start.start()
    if args.hosts and args.topic:
        topic = args.topic
        consumer = KafkaConsumer(topic, auto_offset_reset='latest', \
                bootstrap_servers=args.hosts, api_version=(0, 10), consumer_timeout_ms=1000)
    socketio.run(app,host='0.0.0.0', port=5000)
