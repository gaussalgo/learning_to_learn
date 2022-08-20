# https://super.gluebenchmark.com/tasks
import abc
import json
from typing import List, Sequence, Tuple, Optional, Dict
from zipfile import ZipFile

from promptsource.templates import DatasetTemplates

from evaluation.tasks.task import Task, Metric


class SuperGLUE(Task, abc.ABC):

    promptsource_id: str

    def __init__(self, prompts_template: str = "GPT-3 style"):
        super().__init__()
        template = DatasetTemplates(self.promptsource_id)
        self.prompt = template[prompts_template]


class Broadcoverage(SuperGLUE):

    label: str = "broadcoverage"
    url: str = "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/AX-b.zip"
    promptsource_id: str = "super_glue/axb"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._read_input_label_pairs()
        self.metric_type = Metric.ACCURACY

    def verbalize(self, input_texts: Sequence[str], label: str) -> Tuple[str, str]:
        # format from `load_dataset("super_glue", "cb")["validation"][1]`
        example = {"sentence1": input_texts[0],
                   "sentence2": input_texts[1],
                   "label": 0 if "not" in label else 1}
        # TODO: check: this might return crap
        return self.prompt.apply(example)

    def _read_input_label_pairs(self) -> None:

        with ZipFile(self.data_file) as zipfile:
            with zipfile.open("AX-b/AX-b.jsonl") as f:
                for l in f.readlines():
                    entry = json.loads(l)

                    input_str, label = self.verbalize((entry["sentence1"], entry["sentence2"]), entry["label"])
                    cat = entry["logic"] if "logic" in entry else ""

                    self.data.append((input_str, label, cat))


class BoolQ(SuperGLUE):

    label: str = "boolq"
    url: str = "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip"
    promptsource_id: str = "super_glue/boolq"

    def __init__(self, split: str = "train", prompts_template: str = "GPT-3 Style"):
        super().__init__(prompts_template)
        self.metric_type = Metric.ACCURACY

        from datasets import load_dataset
        dataset = load_dataset("super_glue", "boolq")[split]

        # self.data = [(*("Context: %s. %s" % (pair[0], pair[1]) for pair in self.prompt.apply(example)),  # type:ignore
        #              self._example_cat(example)) for example in dataset]  # List[(input, target, category) tuples]
        self.data = [(*self.prompt.apply(example), self._example_cat(example)) for example in dataset]  # type: ignore
        self.data = [(("Context: %s" % sample[0]).replace("\nAnswer", "? Yes, or No? \nAnswer").replace("\n", " "), sample[1], sample[2])
                     for sample in self.data]

    def _example_cat(self, example: Dict[str, str]) -> str:
        return example["question"].split()[0]


class CommitmentBank(SuperGLUE):

    label: str = "cb"
    promptsource_id = "super_glue/cb"

    def __init__(self, split: str = "train", prompts_template: str = "GPT-3 Style"):
        super().__init__(prompts_template)
        self.metric_type = Metric.ACCURACY

        from datasets import load_dataset
        dataset = load_dataset("super_glue", "cb")[split]

        # no explicit categories either in HF dataset or in the original SGlue sources
        raise NotImplementedError()
        # TODO: where to find categories?
        # self.data = [(*self.prompt.apply(example), self._example_cat(example)) for example in dataset]  # type:ignore

    def _example_cat(self, example: Dict[str, str]) -> str:
        return example["question"].split()[0]


class WinogradSchema(SuperGLUE):

    label: str = "winograd"
    promptsource_id: str = 'super_glue/wsc.fixed'

    def __init__(self, split: str = "train", prompts_template: str = "GPT-3 Style"):
        super().__init__(prompts_template)
        self.metric_type = Metric.ACCURACY

        from datasets import load_dataset
        dataset = load_dataset("super_glue", "wsc")[split]

        self.data = [(*self.prompt.apply(example), self._example_cat(example)) for example in dataset]  # type:ignore

    def _example_cat(self, example: Dict[str, str]) -> str:
        # as category, we use the desambiguated pronouns, i.e. we use "he" examples as demonstrations
        return example["span2_text"].split()[0]
