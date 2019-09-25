from mrecsys.utils.deploy import SocketServer
from mrecsys.utils.deploy import InferenceUnit
import mrecsys.sequence
from mrecsys.sequence import service
import numpy as np
import os

os.chdir(os.path.dirname(__file__))

service = SocketServer(service.__config_path__)
inference_unit = InferenceUnit(mrecsys.sequence.__model_path__, 'cnn')


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
        if request == 'rank_items':
            inputs = data['inputs']
            sequence = inputs['sequence']
            selected_items = inputs['selected_items']
            try:
                filter_items = inputs['filter_items']
                selected_items = np.setdiff1d(selected_items, filter_items)
            except:
                pass
            selected_items = np.setdiff1d(selected_items, sequence)

            sequence = inference_unit.item_indexer.multi_index(sequence)
            selected_items = inference_unit.item_indexer.multi_index(selected_items)
            sequence = sequence[sequence != -1]
            selected_items = selected_items[selected_items != -1]

            if sequence.shape[0] < 1 or selected_items.shape[0] < 1:
                response_json = {
                    "status": "Invalid Item ID",
                    "code": 415
                }
                return response_json

            result = inference_unit.model.predict(sequences=sequence,
                                                  item_ids=selected_items.reshape(-1, 1)).argsort()[::-1]
            selected_items = inference_unit.item_indexer.multi_deindex(selected_items)
            result = selected_items[result]
            response_json = {
                "status": "ok",
                "code": 200,
                "predicts": list(result.astype(str)),
            }

        elif request == 'update_latest':
            inference_unit.update_latest()
            response_json = {
                "status": "ok",
                "code": 200,
            }
        return response_json


def run():
    service.start(request_handler=handler)


if __name__ == '__main__':
    run()
