import itertools

import allennlp
from allennlp.data.dataset_readers.dataset_reader import DatasetReader


@DatasetReader.register("subsample")
class SubsampleDatasetReader(allennlp.data.dataset_readers.TextClassificationJsonReader):
    def __init__(self, train_data_size, validation_data_size, **kwargs):
        super().__init__(**kwargs)
        self.train_data_size = train_data_size
        self.validation_data_size = validation_data_size

    def _read(self, datapath):
        if datapath.endswith("train.jsonl"):
            data_size = self.train_data_size
        else:
            data_size = self.validation_data_size

        yield from itertools.islice(super()._read(datapath), data_size)
