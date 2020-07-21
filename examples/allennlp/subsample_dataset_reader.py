import random

import allennlp
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from sklearn.model_selection import train_test_split


@DatasetReader.register("subsample")
class SubsampleDatasetReader(allennlp.data.dataset_readers.TextClassificationJsonReader):
    def __init__(self, train_data_size, validation_data_size, **kwargs):
        super().__init__(**kwargs)
        self.train_data_size = train_data_size
        self.valid_data_size = valid_data_size

    def _read(self, datapath):
        data = list(super()._read(datapath))
        labels = list(map(lambda instance: instance.get("label").label, data))

        if datapath.endswith("train.jsonl"):
            print("train file")
            train_size = self.train_data_size
        else:
            print("valid file")
            train_size = self.valid_data_size

        _data, _ = train_test_split(data, stratify=labels, train_size=train_size)
        random.shuffle(_data)
        yield from _data
