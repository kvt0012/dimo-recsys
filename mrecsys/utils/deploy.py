from configparser import ConfigParser
import logging
import socket
import traceback
import json
import os
import sys
import time
from threading import Thread
from mrecsys.utils.model_selection import select_latest_model, select_model_by_time
from mrecsys.utils.dataset import load_interactions_by_time


def request_update(service_ip, service_port, service_name, service_token):
    # the return will be in bytes, so decode
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.connect((service_ip, service_port))

    data = {
        "service_name": service_name,
        "service_token": service_token,
        "data": {
            "request": "update_latest",
        }
    }
    soc.send(str(data).encode("utf8"))
    result_bytes = soc.recv(4096)
    result_string = result_bytes.decode("utf8")
    return result_string


class InferenceUnit(object):
    def __init__(self, model_dirpath, model_type, time_code=None):
        self.model_dirpath = model_dirpath
        self.model_type = model_type
        if time_code is None:
            self.model, self.time_code = select_latest_model(model_dirpath, model_type)
        else:
            self.time_code = time_code
            self.model = select_model_by_time(self.time_code, model_dirpath, model_type)
        self.interactions, self.user_indexer, self.item_indexer = load_interactions_by_time(self.time_code)

    def update_latest(self):
        self.model, self.time_code = select_latest_model(self.model_dirpath, self.model_type)
        self.interactions, self.user_indexer, self.item_indexer = load_interactions_by_time(self.time_code)


class SocketServer(object):
    def __init__(self,
                 config_path=None):
        self.request_handler = None
        self.config_path = config_path
        config = ConfigParser()
        config.read(config_path)

        self.log_path = str(config.get('main', 'LOG_PATH'))
        self.ip = str(config.get('main', 'SERVICE_IP'))
        self.port = int(config.get('main', 'SERVICE_PORT'))
        self.name = str(config.get('main', 'SERVICE_NAME'))
        self.token = str(config.get('main', 'SERVICE_TOKEN'))

        logging.basicConfig(filename=os.path.join(self.log_path, str(time.time()) + ".log"), filemode="w",
                            level=logging.DEBUG,
                            format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        console = logging.StreamHandler()
        console.setLevel(logging.ERROR)
        logging.getLogger("").addHandler(console)
        self.logger = logging.getLogger(__name__)
        self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def _client_thread(self, conn, ip, port, MAX_BUFFER_SIZE=4096):
        revc_bytes = conn.recv(MAX_BUFFER_SIZE)
        size = sys.getsizeof(revc_bytes)
        if size >= MAX_BUFFER_SIZE:
            print("The length of input is probably too long: {}".format(size))

        try:
            revc_string = revc_bytes.decode("utf8").rstrip() \
                .replace("'", '"').replace('(', '"(').replace(')', ')"')
            revc_json = json.loads(revc_string)
            res = str(self.request_handler(revc_json))
            print("Result of processing {} is:\n{}".format(revc_string, res))
            vysl = res.encode("utf8")
            conn.sendall(vysl)
        except Exception as e:
            self.logger.error(str(e))
            self.logger.error(str(traceback.print_exc()))
        conn.close()
        print('Connection ' + ip + ':' + port + " ended")

    def start(self, request_handler):
        print("SERVICE_IP:", self.ip)
        print("SERVICE_PORT:", self.port)
        print("SERVICE_NAME:", self.name)
        print("SERVICE_TOKEN:", self.token)

        self.soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        print('Socket created')

        try:
            self.soc.bind((self.ip, self.port))
            print('Socket bind complete')
        except socket.error as e:
            self.logger.error(str(e))
            self.logger.error(str(traceback.print_exc()))
            sys.exit()
        self.request_handler = request_handler
        self.soc.listen(10)
        print('{} service is ready'.format(self.name))

        while True:
            conn, addr = self.soc.accept()
            ip, port = str(addr[0]), str(addr[1])
            print('Accepting connection from ' + ip + ':' + port)
            try:
                Thread(target=self._client_thread, args=(conn, ip, port)).start()
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(str(e))
                self.logger.error(str(traceback.print_exc()))
        self.soc.close()
