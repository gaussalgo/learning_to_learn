import itertools
from typing import Iterable, Dict, List

from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizer

from tasks.task import Task, Metric


class Evaluator:

    def __init__(self,
                 demo_selection_strategy: str = "random",
                 batch_size: int = 16):
        self.demo_selection_strategy = demo_selection_strategy
        self.batch_size = batch_size

    def evaluate(self,
                 model: AutoModelForSeq2SeqLM,
                 tokenizer: PreTrainedTokenizer,
                 tasks: Iterable[Task]) -> Dict[str, float]:
        evaluations = {}

        for task in tasks:
            evaluations[task.label] = self.evaluate_task(model, tokenizer, task)

        return evaluations

    def evaluate_task(self,
                      model: AutoModelForSeq2SeqLM,
                      tokenizer: PreTrainedTokenizer,
                      task: Task) -> float:

        pairs = task.get_all_examples()

        expected_texts = []
        predicted_texts = []

        for batch_offset in range(0, len(pairs), self.batch_size):
            pairs_batch = pairs[batch_offset: batch_offset + self.batch_size]
            input_texts = [pair[0] for pair in pairs_batch]
            expected_texts.extend(input_texts)

            encodings = tokenizer(input_texts)
            predictions = model.generate(**encodings)
            pred_batch = tokenizer.batch_decode(predictions)
            predicted_texts.extend(pred_batch)

        return self._evaluate_results_for_metric(expected_texts, predicted_texts, task.metric_type)

    def _evaluate_results_for_metric(self, expected: List[str], actual: List[str], metric: Metric) -> float:
        assert expected == actual, "Different size of expected and actual predictions :("

        if metric == Metric.ACCURACY:
            return sum(e == a for e, a in zip(expected, actual)) / len(expected)
        elif metric == Metric.FSCORE:
            # token-level F1-score, averaged over all samples:
            fscores = []
            for expected_one, actual_one in zip(expected, actual):

                expected_answers_set = set(itertools.chain(*[a.split() for a in expected_one]))
                actual_answer_set = actual_one.split()

                true_positives = sum(a_word in expected_answers_set for a_word in actual_answer_set)
                false_positives = sum(a_word not in expected_answers_set for a_word in actual_answer_set)
                false_negatives = sum(e_word not in actual_answer_set for e_word in expected_answers_set)

                fscores.append(true_positives / (true_positives + 0.5 * (false_positives + false_negatives)))

            return sum(fscores) / len(fscores)





