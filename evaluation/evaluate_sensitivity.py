import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from evaluation.tasks.en.sensitivity import QADemonstrationsSensitivityROUGE

# TODO: aren't SQuAD examples more informative?
# dataset = load_dataset("squad")
dataset = load_dataset("adversarial_qa", "adversarialQA")
dataset_eval = dataset["validation"].filter(lambda entry: len(entry["context"]) < 2000)
dataset_eval = dataset_eval.select(range(50))

evaluator = QADemonstrationsSensitivityROUGE(dataset_eval, lang="en")

model_path = "gaussalgo/mt5-base-priming-QA_en-cs"

model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_path)

evaluation = evaluator(model, tokenizer, None)
print("%s performance difference: %s" % (model_path, evaluation))
