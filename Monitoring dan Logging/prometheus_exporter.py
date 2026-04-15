from prometheus_client import start_http_server, Summary, Counter
import time
import random

REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
REQUEST_COUNT = Counter('request_count', 'Total request count')

@REQUEST_TIME.time()
def process_request():
    time.sleep(random.random())

if __name__ == "__main__":
    start_http_server(8000)
    print("Prometheus exporter running on port 8000")

    while True:
        process_request()
        REQUEST_COUNT.inc()
        time.sleep(2)