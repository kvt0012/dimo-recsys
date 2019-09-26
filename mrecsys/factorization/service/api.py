from mrecsys.utils.deploy import SocketServer
from mrecsys.utils.deploy import InferenceUnit
import mrecsys.factorization.service
import numpy as np
import os

os.chdir(os.path.dirname(__file__))
service = SocketServer(mrecsys.factorization.service.__config_path__)
inference_unit = InferenceUnit(mrecsys.factorization.__model_path__, 'bpr')


def handle_factorization_result(result, selected_items=None):
    result = np.array(result)
    result = np.apply_along_axis(lambda x: x.astype(int)[0], axis=1, arr=result)
    if selected_items is not None:
        result = np.intersect1d(result, selected_items)
    return inference_unit.item_indexer.multi_deindex(result)


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
            inference_unit.update_latest()
            response_json = {
                "status": "ok",
                "code": 200,
            }
            return response_json

        inputs = data['inputs']
        selected_items = inputs['selected_items']

        if request == 'similar_items':
            item_id = inputs['item_id']
            item_id = inference_unit.item_indexer.index(item_id)
            selected_items = inference_unit.item_indexer.multi_index(selected_items)
            selected_items = selected_items[selected_items != -1]
            if item_id == -1 or selected_items.shape[0] < 1:
                response_json = {
                    "status": "Invalid Item ID",
                    "code": 415
                }
                return response_json

            result = inference_unit.model.similar_items(item_id, inference_unit.interactions.tocsr())
            result = handle_factorization_result(result)

            response_json = {
                "status": "ok",
                "code": 200,
                "predicts": list(result.astype(str)),
            }
            return response_json

        user_id = inputs['user_id']
        user_id = inference_unit.user_indexer.index(user_id)
        if user_id == -1:
            response_json = {
                "status": "Invalid User ID",
                "code": 415,
            }
            return response_json

        selected_items = inference_unit.item_indexer.multi_index(selected_items)
        try:
            filter_items = inputs['filter_items']
            filter_items = inference_unit.item_indexer.multi_index(filter_items)
            selected_items = np.setdiff1d(selected_items, filter_items)
        except:
            pass
        selected_items = selected_items[selected_items != -1]
        if selected_items.shape[0] < 1:
            response_json = {
                "status": "Invalid Item ID",
                "code": 415,
            }
            return response_json

        if request == 'rank_items':
            result = inference_unit.model.rank_items(userid=user_id,
                                                     user_items=inference_unit.interactions.tocsr(),
                                                     selected_items=list(selected_items))
        if request == 'recommend':
            result = inference_unit.model.recommend(userid=user_id,
                                                    user_items=inference_unit.interactions.tocsr(),
                                                    filter_items=list(filter_items))
        result = handle_factorization_result(np.array(result))
        response_json = {
            "status": "ok",
            "code": 200,
            "predicts": list(result.astype(str)),
        }
    return response_json


def run():
    service.start(request_handler=handler)


if __name__ == '__main__':
    run()
