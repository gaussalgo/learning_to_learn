# https://super.gluebenchmark.com/tasks

import json
from typing import List, Sequence
from zipfile import ZipFile

from evaluation.tasks.task import Task


class Broadcoverage(Task):

    def verbalize(self, input_texts: Sequence[str], label: str) -> str:
        # TODO
        pass

    def read_input_label_pairs(self) -> None:

        with ZipFile(open(self.data_file)) as zipfile:
            with zipfile.open("AX-b.jsonl") as f:
                for l in f.readlines():
                    entry = json.loads(l)

                    label = entry["label"]
                    input_str = self.verbalize((entry["sentence1"], entry["sentence2"]), label)
                    cat = entry["logic"]

                    self.data.append((input_str, label, cat))

