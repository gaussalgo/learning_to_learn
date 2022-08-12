import abc
import os
from enum import Enum
from typing import Tuple, Iterable, Sequence, List
from urllib.request import urlopen


class Metric(Enum):
    ACCURACY: int = 1
    FSCORE: int = 1


class Task(abc.ABC):

    label: str
    metric_type: Metric
    url: str
    data_file: str
    data: List[Tuple[str, str, str]] = []  # input, label, category

    def __init__(self, cache_dir: str = ".") -> None:
        self.cache_dir = cache_dir
        self.data_file = self._maybe_download()

    def _maybe_download(self) -> str:
        fname = self.url.split("/")[-1]
        target_fpath = os.path.join(self.cache_dir, fname)
        if os.path.exists(target_fpath):
            return target_fpath
        else:
            print("Downloading %s" % self.url)
            resp = urlopen(self.url)
            with open(target_fpath, "wb") as data_f:
                data_f.write(resp.read())

            assert os.path.exists(target_fpath)

            return target_fpath

    @abc.abstractmethod
    def verbalize(self, input_texts: Sequence[str], label: str) -> str:
        pass

    @abc.abstractmethod
    def read_input_label_pairs(self) -> None:
        pass

    def get_random_example(self) -> Tuple[str, str]:
        # TODO: here
        pass

    def get_random_example_of_category(self, category: str) -> Tuple[str, str]:
        # TODO: here
        pass

    def get_all_examples(self) -> Sequence[Tuple[str, str]]:
        pass
