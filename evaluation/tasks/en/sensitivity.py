from typing import Optional

import torch
from adaptor.evaluators.generative import ROUGE
from datasets import Dataset
from transformers import PreTrainedTokenizer

from evaluation.evaluator import Evaluator
from evaluation.tasks.en.qa import PrimedQATask


class QADemonstrationsSensitivityROUGE(ROUGE):

    def __init__(self, dataset: Dataset, lang: str,
                 num_demonstrations: int = 3, firstn: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.task = PrimedQATask(dataset, lang)
        self.num_demonstrations = num_demonstrations
        self.firstn = firstn

    def __call__(self, model: torch.nn.Module, tokenizer: PreTrainedTokenizer, _) -> float:
        expected, actual_random = Evaluator.collect_predictions(model, tokenizer,
                                                                self.task, self.num_demonstrations, self.firstn)
        random_performance = self.evaluate_str(expected, actual_random)
        print("Model's performance in random selection: %s" % random_performance)

        expected, actual_informative = Evaluator.collect_predictions(model, tokenizer,
                                                                     self.task, self.num_demonstrations, self.firstn)
        informative_performance = self.evaluate_str(expected, actual_informative)

        print("Model's performance in informative selection: %s" % informative_performance)

        return informative_performance - random_performance

    def __str__(self):
        return "%s_%s" % (self.task, super().__str__())
