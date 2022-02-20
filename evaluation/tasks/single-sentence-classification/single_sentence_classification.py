from datasets import load_dataset
from jinja2 import Template
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import difflib
from evaluation.tasks.auto_task import AutoTask
from evaluation.utils.log import get_logger

TEMPLATE_CONFIRMATION = Template(
    """Sentence 1: {{sent1}}
Is Sentence 1 positive or negative ? 
    """
)


class AgNewsDataset(Dataset):
    def __init__(self, tokenizer):
        super().__init__()
        dataset_imdb = load_dataset("ag_news", split="test")
        self.items = []
        self.labels2id = {
            "World": 0, 'Sports': 1, "Business": 2, 'Tech': 3
        }

        for sample in dataset_imdb:
            prompt = TEMPLATE_CONFIRMATION.render(
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
                    "label": sample["label"],
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
    def __init__(self, tokenizer):
        super().__init__()
        dataset_imdb = load_dataset("imdb", split="test")
        self.items = []
        self.labels2id = {'neg': 0, 'pos': 1}

        for sample in dataset_imdb:
            prompt = TEMPLATE_CONFIRMATION.render(
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
                    "sentence1": sample["sentence1"],
                    "label": sample["label"],
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
    def __init__(self, tokenizer):
        super().__init__()
        dataset_imdb = load_dataset("emotion", split="test")
        self.items = []
        self.labels2id = {"sadness": 0, "joy": 1, "love": 2, "anger": 3, "fear": 4.}

        for sample in dataset_imdb:
            prompt = TEMPLATE_CONFIRMATION.render(
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
                    "sentence1": sample["sentence1"],
                    "label": sample["label"],
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
        return "classification-task"

    def evaluate(self, dataset_name='imdb') -> None:
        if dataset_name == 'imdb':
            dataset = IMDBDataset(self.tokenizer)
        elif dataset_name == 'ag-news':
            dataset = IMDBDataset(self.tokenizer)
        else:
            dataset = EmotionDataset(self.tokenizer)

        LABELS_LIST = dataset.labels2id  # TODO check if this is correct or need to be adapted

        is_first = True

        def extract_label_list_id():
            label_ids = []
            matched_labels = {}
            for label in LABELS_LIST:
                matched_label = difflib.get_close_matches(label, list(self.tokenizer.vocab.keys()))[
                    0]  # take the most likely match
                assert len(matched_label) > 0
                label_ids.append(self.tokenizer.vocab[matched_label])
                matched_labels[label] = matched_label
            return label_ids, matched_labels

        label_ids, matched_labels = extract_label_list_id()

        logger = get_logger()
        prompts, sentences, soft_labels, y_predicted, y_label = [], [], [], [], []
        count = 0
        for sample in tqdm(dataset, desc=f"Evaluating {self.get_display_name()}"):
            count += 1

            def get_output(sample, label_list_ids=None):
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
                    torch.tensor([logits[:, -1, label_id] for label_id in label_list_ids])).tolist()
                return sofmax_results

            # use the output to confirm
            prompt = TEMPLATE_CONFIRMATION.render(
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
            soft_confirmations = get_output(sample_confirmation, label_ids)
            confirmation_output = LABELS_LIST[soft_confirmations.index(max(soft_confirmations))]
            if is_first:
                is_first = False
                log_msg = "Evaluation example for MRPC-Negative\nLabels\t{}\n".format(LABELS_LIST)

                log_msg += "\nprompt#1 (Standard):\n" + sample["prompt"]
                log_msg += "\nmodel output:\n" + paraphrase

                log_msg += "\n\nprompt#2 (Negative):\n" + sample_confirmation["prompt"]
                log_msg += "\nsoft model output:\n" + str(soft_confirmations)
                log_msg += "\npredicted model output:\n" + confirmation_output
                logger.info(log_msg)

            # log the prompts and the outputs

            prompts.append(prompt)
            sentences.append(sample["sentence1"])
            y_label.append(sample["label"])
            soft_labels.append(soft_confirmations)
            y_predicted.append(confirmation_output)

    self.metrics = {
        "prompts": prompts,
        "sentences": sentences,
        "soft_labels": soft_labels,
        "y_predicted": y_predicted,
        "y_label": y_label
    }
    logger.info("Metrics : {}".format(self.metrics))
