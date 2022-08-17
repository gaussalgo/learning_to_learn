from promptsource.templates import DatasetTemplates
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from evaluation.tasks.en.superglue import Broadcoverage, BoolQ, WinogradSchema
from evaluator import Evaluator

model_path = "gaussalgo/mt5-base-priming-QA_en-cs"


model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

templates = DatasetTemplates("super_glue/axb")

tasks = [
    # Broadcoverage(),
    BoolQ(),
    # WinogradSchema()
]


evaluator = Evaluator()
evaluations = evaluator.evaluate(model, tokenizer, tasks, firstn=50)

print("Evaluation done: %s" % evaluations)
print()
