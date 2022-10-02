from typing import Tuple, List


def selection_criterion(predicted_example: Tuple[str, str, str],
                        candidate_demonstration: Tuple[str, str, str],
                        demo_selection_strategy: str) -> bool:
    if demo_selection_strategy == "random":
        return True
    elif demo_selection_strategy == "cluster-random":
        # any sample is fine for demonstration, with the random selection strategy
        return predicted_example[2] == candidate_demonstration[2]
    else:
        raise ValueError("Demo selection strategy %s unknown." % demo_selection_strategy)


def construct_sample(demonstrations: List[Tuple[str, str, str]],
                     predicted_sample: Tuple[str, str, str]) -> str:
    demonstrations = "\n".join(["Input: %s Prediction: %s" % demo[:2] for demo in demonstrations])
    return demonstrations + "Input: %s Prediction:" % predicted_sample[0]
