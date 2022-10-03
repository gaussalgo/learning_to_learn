# Learning to Learn
Training and evaluation testbed for few-shot learners on unseen tasks

## Training

To reproduce the training of our published models, clone this repository and run the following scripts:

```shell
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pip install -r train/requirements.txt
pip install -r evaluation/requirements.txt

python train/train_mt5_qa_en+cs_hard_priming.py
```
Don't forget to prepend the execution of the training scripts with GPU configuration (`CUDA_VISIBLE_DEVICES`) or logging configuration allowing you to track the experiment (we use comet.ml, setting `COMET_API_KEY`).

## Evaluation

To run the evaluation over selected SuperGLUE tasks, you can use `run_evaluator.py` script:

```shell
cd evaluation
python run_evaluator.py
```
or from python interpreter (working dir `./evaluation`):
```python
from promptsource.templates import DatasetTemplates
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from evaluation.tasks.en.superglue import Broadcoverage, BoolQ, WinogradSchema
from evaluator import Evaluator

model_path = "gaussalgo/mt5-base-priming-QA_en-cs"


model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

templates = DatasetTemplates("super_glue/axb")

tasks = [
    Broadcoverage(),
    BoolQ(),
    WinogradSchema()
]


evaluator = Evaluator()
evaluations = evaluator.evaluate(model, tokenizer, tasks, firstn=50)

print("Evaluation done: %s" % evaluations)
```
