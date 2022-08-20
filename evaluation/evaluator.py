import itertools
from typing import Iterable, Dict, List, Optional, Tuple, Union

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
                 tasks: Iterable[Task],
                 firstn: Optional[int] = None) -> Dict[str, float]:
        evaluations = {}

        for task in tasks:
            task_eval = self.evaluate_task(model, tokenizer, task, firstn)
            print("Task %s eval: %s" % (task, task_eval))

            evaluations[task.label] = task_eval

        return evaluations

    def selection_criterion(self,
                            predicted_example: Tuple[str, str, str],
                            candidate_demonstration: Tuple[str, str, str]) -> bool:
        if self.demo_selection_strategy == "random":
            # any sample is fine for demonstration, with the random selection strategy
            return predicted_example[2] == candidate_demonstration[2]
        else:
            raise ValueError("Demo selection strategy %s unknown." % self.demo_selection_strategy)

    def _construct_sample(self,
                          demonstrations: List[Tuple[str, str, str]],
                          predicted_sample: Tuple[str, str, str]) -> str:
        return "\n".join(["%s %s" % demo[:2] for demo in demonstrations] + [predicted_sample[0]])

    def evaluate_task(self,
                      model: AutoModelForSeq2SeqLM,
                      tokenizer: PreTrainedTokenizer,
                      task: Task,
                      firstn: Optional[int] = None,
                      num_demonstrations: int = 3) -> float:

        expected_texts = []
        predicted_texts = []

        num_samples = firstn if firstn is not None else len(task.data)
        skipped = 0

        for batch_offset in range(0, num_samples, self.batch_size):
            tuples_batch = task.data[batch_offset: batch_offset + self.batch_size]
            input_texts = []
            targets = []

            for sample in tuples_batch:
                demonstrations = []
                while len(demonstrations) < num_demonstrations:
                    try:
                        demonstrations.append(next(demo for demo in task.data
                                                   if demo[0] != sample[0] and demo not in demonstrations
                                                   and self.selection_criterion(sample, demo)))
                    except StopIteration:
                        break
                if not demonstrations:
                    skipped += 1
                    continue
                input_texts.append(self._construct_sample(demonstrations, sample))
                targets.append(sample[1])

            encodings = tokenizer(input_texts, return_tensors="pt", padding=True)

            predictions = model.generate(**encodings)
            pred_batch = tokenizer.batch_decode(predictions, skip_special_tokens=True)

            expected_texts.extend(targets)
            predicted_texts.extend(pred_batch)

        print("Skipped samples: %s out of total: %s" % (skipped, num_samples))
        # BoolQ responds consistently in Czech
        # print(self._evaluate_results_for_metric(["ano" if "es" in e else "ne" for e in expected_texts], predicted_texts,
        #                                         task.metric_type, ignore_casing=True))

        return self._evaluate_results_for_metric(expected_texts, predicted_texts, task.metric_type, ignore_casing=True)

    def _evaluate_results_for_metric(self,
                                     expected: List[str],
                                     actual: List[str],
                                     metric: Union[Metric, int],
                                     ignore_casing: bool) -> float:
        assert len(expected) == len(actual), "Different size of expected and actual predictions :("
        if ignore_casing:
            expected = [e.lower() for e in expected]
            actual = [a.lower() for a in actual]

        if metric.value == Metric.ACCURACY.value:
            return sum(e == a for e, a in zip(expected, actual)) / len(expected)
        elif metric.value == Metric.FSCORE.value:
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
        else:
            raise ValueError("Not implemented metric: %s" % metric)
