from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

load_dataset("imdb", split="test")
load_dataset("emotion", split="test")
load_dataset("ag_news", split="test")

load_dataset("glue", "mnli_mismatched", split="validation")
load_dataset("glue", "mnli_matched", split="validation")
load_dataset("glue", "mrpc", split="validation")

for pair in ['cs-en', 'kk-en', 'fi-en', 'gu-en', 'de-en', 'kk-en', 'lt-en', 'ru-en', 'zh-en', 'fr-en']:
    print(pair)
    load_dataset("wmt19", pair, split="validation")["translation"]

load_dataset("glue", "rte", split="validation")

for MODEL_NAME in ['t5-small', 't5-base', 't5-large', 't5-3b', 'bigscience/T0_3B', 'bigscience/T0pp',
                   'bigscience/T0p', 'bigscience/T0 gpt', 'gpt2 distilgpt2', 'EleutherAI/gpt-neo-125M',
                   'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-j-6B', 'EleutherAI/gpt-neo-2.7B']:
    print(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModel.from_pretrained(MODEL_NAME)
