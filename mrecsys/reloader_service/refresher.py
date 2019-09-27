from mrecsys.utils.dataset import reload_data
from mrecsys.factorization import trainer as f_trainer
from mrecsys.sequence import trainer as s_trainer
from mrecsys.utils.deploy import request_update
from mrecsys import reloader_service


def run():
    reloader_service.__busy__ = True
    reload_data(item_col='service_id')
    f_trainer.run()
    s_trainer.run()
    request_update("localhost", 6000)
    request_update("localhost", 5000)
    reloader_service.__busy__ = False

if __name__ == '__main__':
    run()
