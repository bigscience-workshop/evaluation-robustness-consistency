from datasets import load_dataset
from jinja2 import Template
from torch.utils.data import Dataset
from tqdm import tqdm
import torch

from evaluation.tasks.auto_task import AutoTask
from evaluation.utils.log import get_logger

TEMPLATE_STD = Template(
    """
Sentence 1: {{sent1}}
Sentence 2: {{sent2}}
Do these two sentences express the same meaning? Yes or no?
    """
)

TEMPLATE_NEG = Template(
    """
Sentence 1: {{sent1}}
Sentence 2: {{sent2}}
Do these two sentences express different meanings? Yes or no?
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

def get_output(task, sample):
    with torch.no_grad():
        output = task.model.generate(
            input_ids=sample["input_ids"].to(task.device),
            attention_mask=sample["attention_mask"].to(task.device),
            max_length=min(sample["input_len"] * 2, 1024),
            # hard-coded to 1024 since each model has diferent naming for max length
        )
        decoded_output = task.tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded_output


class MRPCNegativeTask(AutoTask):
    @staticmethod
    def get_display_name() -> str:
        return "mrpc-negative"

    def evaluate(self) -> None:
        dataset_std = MRPCDataset(self.tokenizer, TEMPLATE_STD)
        dataset_neg = MRPCDataset(self.tokenizer, TEMPLATE_NEG)

        accuracy = 0
        consistency = 0
        logs = []

        logger = get_logger()
        for sample_std, sample_neg in tqdm(zip(dataset_std, dataset_neg), desc=f"Evaluating {self.get_display_name()}"):
            predicted_answer_std = get_output(self, sample_std)
            predicted_answer_neg = get_output(self, sample_neg)

            # compute the performance and log the prompts and the outputs
            label = sample_std["label"]
            label_match = int(label.lower().strip() == predicted_answer_std.lower().strip())

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
