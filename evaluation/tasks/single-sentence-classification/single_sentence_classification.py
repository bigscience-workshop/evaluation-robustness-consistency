from datasets import load_dataset
from jinja2 import Template
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import difflib
from evaluation.tasks.auto_task import AutoTask
from evaluation.utils.log import get_logger

from jinja2 import Template

template_imdb = """Sentence 1: {{sent1}} """ + """Is Sentence 1 {} or {} ? """.format("'neg", "positive")
TEMPLATE_IMDB = Template(template_imdb)

template_agnews = """Sentence 1: {{sent1}} """ + """ Is the theme of Sentence 1 {} or {} or {} or {} ? """.format(
    "World", "Sports", "Business", "Tech")
TEMPLATE_AGNEWS = Template(template_agnews)

template_emotion = """Sentence 1: {{sent1}} """ + """Is the emotion expressed in Sentence 1 {}, {}, {}, {}, {} or {} ? """.format(
    "sadness", "joy", "love", "anger", "fear", "surprise")
TEMPLATE_EMOTION = Template(template_emotion)


# TODO: 1, 2 shotsx

class AgNewsDataset(Dataset):
    def __init__(self, tokenizer, seed, number_of_shots):
        super().__init__()
        dataset_agnew = load_dataset("ag_news", split="test")
        self.items = []
        self.labels2id = {
            "World": 0, 'Sports': 1, "Business": 2, 'Tech': 3
        }
        self.id2labels = {v: k for k, v in self.labels2id.items()}

        for sample in dataset_agnew:
            prompt = TEMPLATE_AGNEWS.render(
                sent1=sample["text"],
            )

            # Tokenize and construct this sample
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
            )
            self.items.append(
                {
                    "prompt": prompt,
                    "sentence1": sample["text"],
                    "label": self.id2labels[sample["label"]],
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "input_len": inputs["attention_mask"].shape[1],
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class IMDBDataset(Dataset):
    def __init__(self, tokenizer, seed, number_of_shots):
        super().__init__()
        dataset_imdb = load_dataset("imdb", split="test")
        self.items = []
        self.labels2id = {'neg': 0, 'positive': 1}
        self.id2labels = {v: k for k, v in self.labels2id.items()}

        for sample in dataset_imdb:
            prompt = TEMPLATE_IMDB.render(
                sent1=sample["text"],
            )

            # Tokenize and construct this sample
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
            )
            self.items.append(
                {
                    "prompt": prompt,
                    "sentence1": sample["text"],
                    "label": self.id2labels[sample["label"]],
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "input_len": inputs["attention_mask"].shape[1],
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class EmotionDataset(Dataset):
    def __init__(self, tokenizer, seed, number_of_shots):
        super().__init__()
        dataset_emotion = load_dataset("emotion", split="test")
        self.items = []
        self.labels2id = {"sadness": 0, "joy": 1, "love": 2, "anger": 3, "fear": 4, "surprise": 5}
        self.id2labels = {v: k for k, v in self.labels2id.items()}

        for sample in dataset_emotion:
            prompt = TEMPLATE_EMOTION.render(
                sent1=sample["text"],
            )

            # Tokenize and construct this sample
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
            )
            self.items.append(
                {
                    "prompt": prompt,
                    "sentence1": sample["text"],
                    "label": self.id2labels[sample["label"]],
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "input_len": inputs["attention_mask"].shape[1],
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class Classification_Task(AutoTask):
    @staticmethod
    def get_display_name() -> str:
        return "single-sentence-classification"

    def evaluate(self, dataset_name='imdb', seed=42, number_of_shots=5) -> None:
        if dataset_name == 'imdb':
            dataset = IMDBDataset(self.tokenizer, seed=42, number_of_shots=5)
        elif dataset_name == 'ag-news':
            dataset = AgNewsDataset(self.tokenizer, seed=42, number_of_shots=5)
        elif dataset_name == 'emotion':
            dataset = EmotionDataset(self.tokenizer, seed=42, number_of_shots=5)
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
            'imdb': TEMPLATE_IMDB, 'emotion': TEMPLATE_EMOTION, 'ag-news': TEMPLATE_AGNEWS
        }
        for sample in tqdm(dataset, desc=f"Evaluating {self.get_display_name()}"):
            count += 1
            if count == 7:
                break
            # use the output to confirm
            prompt = template[dataset_name].render(
                sent1=sample["sentence1"],
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
