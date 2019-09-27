import json
import numpy as np


class Indexer:
    def __init__(self,
                 values: np.ndarray = None,
                 dumped_filepath: str = None):
        if values is not None:
            self._val_to_idx = {}
            self._idx_to_val = {}
            for idx, val in enumerate(values):
                self._val_to_idx[str(val)] = str(idx + 1)
                self._idx_to_val[str(idx + 1)] = str(val)

        elif dumped_filepath is not None:
            with open(dumped_filepath, 'r') as fp:
                self._val_to_idx = json.load(fp)
                self._idx_to_val = dict([(value, key) for key, value in self._val_to_idx.items()])

        else:
            raise RuntimeError("No entry values")

    def index(self, val):
        return np.int32(self._val_to_idx.get(str(val), "-1"))

    def deindex(self, idx):
        return self._idx_to_val.get(str(idx), "-1")

    def multi_index(self, vals):
        func = np.vectorize(lambda x: self.index(x))
        return func(vals)

    def multi_deindex(self, vals):
        func = np.vectorize(lambda x: self.deindex(x))
        return func(vals)

    def dumps(self, PATH):
        with open(PATH, 'w') as fp:
            json.dump(self._val_to_idx, fp)

    def print(self):
        print(self._val_to_idx)
        print(self._idx_to_val)