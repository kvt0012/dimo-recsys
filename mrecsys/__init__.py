import os
__dataset_path__ = os.path.join(os.path.dirname(__file__), 'dataset')
__sequence_api_path__ = os.path.join(os.path.dirname(__file__), 'sequence/service/api.py')
__factorization_api_path__ = os.path.join(os.path.dirname(__file__), 'factorization/service/api.py')

from mrecsys.sequence.service import api as sequence_api
from mrecsys.factorization.service import api as factorization_api
import threading


class MyThread(threading.Thread):

    # Thread class with a _stop() method.
    # The thread itself has to check
    # regularly for the stopped() condition.

    def __init__(self, *args, **kwargs):
        super(MyThread, self).__init__(*args, **kwargs)
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()


sequence_thread = MyThread(target=sequence_api.run)
factorization_thread = MyThread(target=factorization_api.run)


def run():
    sequence_thread.daemon = True
    sequence_thread.start()

    factorization_thread.daemon = True
    factorization_thread.start()
