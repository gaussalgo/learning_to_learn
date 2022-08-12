from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from evaluator import Evaluator

model_path = "gaussalgo/mt5-base-priming-QA_en-cs"


model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

tasks = []


evaluator = Evaluator()
evaluator.evaluate(model, tokenizer, )
