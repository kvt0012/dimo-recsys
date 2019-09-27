from mrecsys.utils.deploy import SocketServer
import mrecsys.reloader_service
import numpy as np
import os

from mrecsys.reloader_service import refresher
import threading

os.chdir(os.path.dirname(__file__))
service = SocketServer(mrecsys.reloader_service.__config_path__)


def handler(revc_json):
    service_name = revc_json['service_name']
    service_token = revc_json['service_token']
    response_json = {
        "status": "Unsupported Request",
        "code": 415
    }
    if service_name == service.name and service_token == service.token:
        data = revc_json['data']
        request = data['request']

        if request == 'update_latest':
            refresher.run()
            response_json = {
                "status": "ok",
                "code": 200,
            }
        return response_json


def run():
    service.start(request_handler=handler)


if __name__ == '__main__':
    run()
