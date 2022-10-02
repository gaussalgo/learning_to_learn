from typing import Dict, Any

from datasets import Dataset

from evaluation.tasks.task import Task
from training.priming_objective import priming_formats


class PrimedQATask(Task):

    def _construct_qa_prompt(self, question: str, context: str) -> str:
        return priming_formats["QA"][self.lang_id] % (question, context)

    @staticmethod
    def _informative_factor(sample: Dict[str, Any]) -> str:
        return (sample["question"].split()[0]
                if not sample["question"].startswith("To")
                else " ".join(sample["question"].split()[:2]))

    def __init__(self, dataset: Dataset, lang_id: str):
        super().__init__()
        self.lang_id = lang_id
        # dataset = load_dataset(*hf_dataset_identifiers)[hf_dataset_split]

        self.data = [(self._construct_qa_prompt(sample["question"], sample["context"]),
                      sample["answers"]["text"][0],
                      self._informative_factor(sample)) for sample in dataset]
