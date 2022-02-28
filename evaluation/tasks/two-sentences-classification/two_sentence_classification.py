from datasets import load_dataset
from jinja2 import Template
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import difflib
import random

from evaluation.tasks.auto_task import AutoTask
from evaluation.utils.log import get_logger

template_mnli = """Sentence 1: {{sent1}} Sentence 2: {{sent2}} Do Sentence 1 and Sentence 2 convey the same meaning? Yes or No?"""
TEMPLATE_MNLI = Template(template_mnli)

template_rte = """Sentence 1: {{sent1}} Sentence 2: {{sent2}} Do Sentence 1 and Sentence 2 convey the same meaning? Yes or No?"""
TEMPLATE_RTE = Template(template_rte)

template_mrpc = """Sentence 1: {{sent1}} Sentence 2: {{sent2}} Do Sentence 1 and Sentence 2 convey the same meaning? Yes or No?"""
TEMPLATE_MRPC = Template(template_mrpc)

template_wmt = """Sentence 1: {{sent1}} Sentence 2: {{sent2}} Is Sentence 1 a valid translation of Sentence 2 ? Yes or No?"""
TEMPLATE_WMT = Template(template_wmt)


class WMTEnglishDataset(Dataset):
    def __init__(self, tokenizer, pair="kk-en"):
        super().__init__()

        self.languages = ['cs-en', 'kk-en', 'fi-en']  # , 'gu-en','de-en', 'kk-en', 'lt-en', 'ru-en', 'zh-en', 'fr-en']
        self.filter = 150

        assert "en" in pair, f"Expected `pair` to contain English, but got {pair} instead"
        wmt_ds = dict()
        for pair in self.languages:
            print('Loading', pair)
            wmt_ds[pair] = load_dataset("wmt19", pair, split="validation")["translation"]

        self.items = []
        self.labels2id = {
            "Yes": 0, 'No': 1
        }
        self.id2labels = {v: k for k, v in self.labels2id.items()}
        for key, wmt in wmt_ds.items():
            key_1 = key.split('-')[0]
            key_2 = key.split('-')[1]
            for index, sample in enumerate(wmt):
                if index == self.filter:
                    break
                prompt = TEMPLATE_WMT.render(
                    sent1=sample[key_1],
                    sent2=sample[key_2],
                ).strip()
                # Tokenize and construct this sample
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                )
                self.items.append(
                    {"sentence1": sample[key_1],
                     "sentence2": sample[key_2],
                     "prompt": prompt,
                     "pair": key,
                     "input_ids": inputs["input_ids"],
                     "attention_mask": inputs["attention_mask"],
                     "input_len": inputs["attention_mask"].shape[1],
                     "label": "Yes",  # TODO voir les details
                     }
                )
                # select language + sentence
                negative_language = random.choice(self.languages)
                while negative_language == key:
                    negative_language = random.choice(self.languages)
                key_s = negative_language.split('-')[0]
                sentence = random.choice(wmt_ds[negative_language])[key_s]
                self.items.append(
                    {"sentence1": sentence,
                     "sentence2": sample[key_2],
                     "prompt": prompt,
                     "pair": negative_language,
                     "input_ids": inputs["input_ids"],
                     "attention_mask": inputs["attention_mask"],
                     "input_len": inputs["attention_mask"].shape[1],
                     "label": "No",
                     }
                )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.items[index]


class MNLIDataset(Dataset):
    def __init__(self, tokenizer):
        super().__init__()
        mnli_mismatched = load_dataset("glue", "mnli_mismatched", split="validation")
        mnli_matched = load_dataset("glue", "mnli_matched", split="validation")
        self.items = []
        self.labels2id = {
            "Yes": 0, 'No': 1
        }
        self.id2labels = {v: k for k, v in self.labels2id.items()}
        for index, ds in enumerate([mnli_mismatched, mnli_matched]):
            for sample in ds:
                prompt = TEMPLATE_MNLI.render(
                    sent1=sample["premise"],
                    sent2=sample["hypothesis"],
                ).strip()

                # Tokenize and construct this sample
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                )
                self.items.append(
                    {"sentence1": sample["premise"],
                     "sentence2": sample["hypothesis"],
                     "prompt": prompt,
                     "input_ids": inputs["input_ids"],
                     "attention_mask": inputs["attention_mask"],
                     "input_len": inputs["attention_mask"].shape[1],
                     "label": ["Yes", "No"][index],
                     }
                )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class RTEDataset(Dataset):
    def __init__(self, tokenizer):
        super().__init__()
        rte = load_dataset("glue", "rte", split="validation")
        self.items = []
        self.items = []
        self.labels2id = {
            "Yes": 0, 'No': 1
        }
        self.id2labels = {v: k for k, v in self.labels2id.items()}

        for sample in rte:
            prompt = TEMPLATE_RTE.render(
                sent1=sample["sentence1"],
                sent2=sample["sentence2"],
            ).strip()

            # Tokenize and construct this sample
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
            )
            self.items.append(
                {"sentence1": sample["sentence1"],
                 "sentence2": sample["sentence2"],
                 "prompt": prompt,
                 "input_ids": inputs["input_ids"],
                 "attention_mask": inputs["attention_mask"],
                 "input_len": inputs["attention_mask"].shape[1],
                 "label": ["Yes", "No"][sample["label"]],
                 }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class MRPCDataset(Dataset):
    def __init__(self, tokenizer):
        super().__init__()
        mrpc = load_dataset("glue", "mrpc", split="validation")
        self.items = []
        self.labels2id = {
            "Yes": 0, 'No': 1
        }
        self.id2labels = {v: k for k, v in self.labels2id.items()}
        for sample in mrpc:
            prompt = TEMPLATE_MRPC.render(
                sent1=sample["sentence1"],
                sent2=sample["sentence2"],
            ).strip()

            # Tokenize and construct this sample
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
            )
            self.items.append(
                {"sentence1": sample["sentence1"],
                 "sentence2": sample["sentence2"],
                 "prompt": prompt,
                 "input_ids": inputs["input_ids"],
                 "attention_mask": inputs["attention_mask"],
                 "input_len": inputs["attention_mask"].shape[1],
                 "label": ["Yes", "No"][sample["label"]],
                 }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class TwoSentenceClassificationTask(AutoTask):
    @staticmethod
    def get_display_name() -> str:
        return "two-sentences-classification"

    def evaluate(self, dataset_name='mnli', seed=42, number_of_shots=5) -> None:
        if dataset_name == 'mnli':
            dataset = MNLIDataset(self.tokenizer)
        elif dataset_name == 'rte':
            dataset = RTEDataset(self.tokenizer)
        elif dataset_name == 'mrpc':
            dataset = MRPCDataset(self.tokenizer)
        elif dataset_name == 'wmt':
            dataset = WMTEnglishDataset(self.tokenizer)
        else:
            raise NotImplementedError

        LABELS_LIST = dict()
        for k, v in dataset.labels2id.items():
            assert len(self.tokenizer.tokenize(k)) == 1, "Thinks of changing the label {}".format(
                self.tokenizer.tokenize(k))
            LABELS_LIST[k] = self.tokenizer.vocab[self.tokenizer.tokenize(k)[0]]

        is_first = True

        logger = get_logger()
        prompts, sentences, soft_labels, y_predicted, y_label, stry_label = [], [], [], [], [], []
        count = 0

        def get_output(sample, label_list_ids):
            with torch.no_grad():
                if ('t5' in self.model.name_or_path.lower()) or ('t0' in self.model.name_or_path.lower()):
                    output = self.model(labels=sample["input_ids"].to(self.device),
                                        input_ids=sample["input_ids"].to(self.device),
                                        attention_mask=sample["attention_mask"].to(self.device))

                elif ('gpt' in self.model.name_or_path.lower()):
                    output = self.model(input_ids=sample["input_ids"].to(self.device),
                                        attention_mask=sample["attention_mask"].to(self.device))
                else:
                    raise NotImplementedError
                logits = output['logits']
            sofmax_results = torch.nn.Softmax()(
                torch.tensor([logits[:, -1, label_id] for label_id in list(label_list_ids.values())])).tolist()
            return sofmax_results

        template = {
            'mnli': TEMPLATE_MNLI, 'rte': TEMPLATE_RTE, 'mrpc': TEMPLATE_MRPC, 'wmt': TEMPLATE_WMT
        }
        for sample in tqdm(dataset, desc=f"Evaluating {self.get_display_name()}"):
            count += 1
            if count == 7:
                break
            # use the output to confirm
            prompt = template[dataset_name].render(
                sent1=sample["sentence1"],
                sent2=sample["sentence2"]
            )
            # Tokenize and construct this sample
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
            )

            sample_confirmation = {
                "prompt": prompt,
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "input_len": inputs["attention_mask"].shape[1],
            }
            soft_confirmations = get_output(sample_confirmation, LABELS_LIST)
            if is_first:
                is_first = False
                log_msg = "Evaluation example for MRPC-Negative\nLabels\t{}\n".format(LABELS_LIST)

                log_msg += "\n\nprompt:\n" + sample_confirmation["prompt"]
                log_msg += "\nsoft model output:\n" + str(soft_confirmations)
                log_msg += "\ngolden:\n" + str(sample["label"])
                log_msg += "\ngolden:\n" + str(dataset.labels2id[sample["label"]])
                logger.info(log_msg)

            # log the prompts and the outputs

            prompts.append(prompt)
            sentences.append(sample["sentence1"])
            y_label.append(dataset.labels2id[sample["label"]])
            stry_label.append(sample["label"])
            soft_labels.append(soft_confirmations)

        self.metrics = {
            "prompts": prompts,
            "sentences": sentences,
            "soft_labels": soft_labels,
            "stry_label": stry_label,
            "y_label": y_label,
        }
        logger.info("Metrics : {}".format(self.metrics))
