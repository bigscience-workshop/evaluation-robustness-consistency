from datasets import load_dataset
from jinja2 import Template
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import difflib

from evaluation.tasks.auto_task import AutoTask
from evaluation.utils.log import get_logger

TEMPLATE_STD = Template(
    """
Sentence 1: {{sent1}}
Sentence 2: {{sent2}}
Do Sentence 1 and Sentence 2 convey the same meaning? Yes or No?
    """
)

TEMPLATE_NEG = Template(
    """
Sentence 1: {{sent1}}
Sentence 2: {{sent2}}
Do Sentence 1 and Sentence 2 express different meanings? Yes or No?
    """
)


class MRPCDataset(Dataset):
    def __init__(self, tokenizer, TEMPLATE):
        super().__init__()
        mrpc = load_dataset("glue", "mrpc", split="validation")
        self.items = []

        for sample in mrpc:
            prompt = TEMPLATE.render(
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
                {
                    "prompt": prompt,
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "input_len": inputs["attention_mask"].shape[1],
                    "label": ["Yes", "No"][1 - sample["label"]],
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

def get_output(task, sample, label_list_ids):
    # unbatched function
    with torch.no_grad():
        if ('t5' in task.model.name_or_path.lower()) or ('t0' in task.model.name_or_path.lower()):
            output = task.model(labels=sample["input_ids"].to(task.device),
                                input_ids=sample["input_ids"].to(task.device),
                                attention_mask=sample["attention_mask"].to(task.device))

        elif ('gpt' in task.model.name_or_path.lower()):
            output = task.model(
                input_ids=sample["input_ids"].to(task.device),
                attention_mask=sample["attention_mask"].to(task.device))
        else:
            raise NotImplementedError

        logits = output['logits']
    sofmax_results = torch.nn.Softmax()(
        torch.tensor([logits[:, -1, label_id] for label_id in label_list_ids])).tolist()
    return sofmax_results

def extract_label_list_id(task, dataset):
    labels = list(set([sample['label'] for sample in dataset]))
    label_ids = []
    matched_labels = {}
    for label in labels:
        matched_label = difflib.get_close_matches(label, list(task.tokenizer.vocab.keys()))[
            0]  # take the most likely match
        assert len(matched_label) > 0
        label_ids.append(task.tokenizer.vocab[matched_label])
        matched_labels[label] = matched_label
    return label_ids, matched_labels, labels


class MRPCNegativeTask(AutoTask):
    @staticmethod
    def get_display_name() -> str:
        return "mrpc_negative"

    def evaluate(self) -> None:
        dataset_std = MRPCDataset(self.tokenizer, TEMPLATE_STD)
        dataset_neg = MRPCDataset(self.tokenizer, TEMPLATE_NEG)

        accuracy = 0
        consistency = 0
        logs = []

        label_ids_std, matched_labels_std, labels_std = extract_label_list_id(self, dataset_std)
        label_ids_neg, matched_labels_neg, labels_neg = extract_label_list_id(self, dataset_neg)
        
        logger = get_logger()
        
        logger.info("Labels for std are {}".format(matched_labels_std))
        logger.info("Labels for neg are {}".format(label_ids_neg))
        count = 0
        for sample_std, sample_neg in tqdm(zip(dataset_std, dataset_neg), desc=f"Evaluating {self.get_display_name()}"):
            count += 1

            soft_predicted_answer_std = get_output(self, sample_std, label_ids_std)
            soft_predicted_answer_neg = get_output(self, sample_neg, label_ids_neg)

            predicted_answer_neg = labels_neg[soft_predicted_answer_neg.index(max(soft_predicted_answer_neg))]
            predicted_answer_std = labels_std[soft_predicted_answer_std.index(max(soft_predicted_answer_std))]

            label = matched_labels_std[sample_std["label"]]
            label_match = int(label == predicted_answer_std)
            accuracy += label_match
            # consistent if their answers are different
            consistency += int(predicted_answer_std.lower() != predicted_answer_neg.lower())

            logs.append({
                "standard prompt": sample_std["prompt"],
                "standard answer": predicted_answer_std,
                "negative prompt": sample_neg["prompt"],
                "negative answer": predicted_answer_neg,
                "gold label": sample_std["label"]
                })

            if len(logs) == 1:
                logger.info(logs[0])
        
        self.metrics = {
            "0_accuracy": accuracy / len(dataset_std) * 100,
            "1_consistency": consistency / len(dataset_std) * 100,
            "2_output log": logs,
        }
