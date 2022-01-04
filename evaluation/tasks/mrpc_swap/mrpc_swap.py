from datasets import load_dataset
from jinja2 import Template
from torch.utils.data import Dataset
from tqdm import tqdm
import torch

from evaluation.tasks.auto_task import AutoTask
from evaluation.utils.log import get_logger

from evaluation.tasks.mrpc_negative.mrpc_negative import MRPCDataset, get_output

TEMPLATE_STD = Template(
    """
Sentence 1: {{sent1}}
Sentence 2: {{sent2}}
Do these two sentences express the same meaning? Yes or no?
    """
)

TEMPLATE_NEG = Template(
    """
Sentence 1: {{sent2}}
Sentence 2: {{sent1}}
Do these two sentences express the same meaning? Yes or no?
    """
)



class MRPCSwapTask(AutoTask):
    @staticmethod
    def get_display_name() -> str:
        return "mrpc_swap"

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
            # consistent if their answers are the same
            consistency += int(predicted_answer_std.lower() == predicted_answer_neg.lower())

            logs.append({
                "standard prompt": sample_std["prompt"],
                "standard answer": predicted_answer_std,
                "swap prompt": sample_neg["prompt"],
                "swap answer": predicted_answer_neg,
                "gold label": sample_std["label"]
                })

            if len(logs) == 1:
                logger.info(logs[0])
        
        self.metrics = {
            "0_accuracy": accuracy / len(dataset_std) * 100,
            "1_consistency": consistency / len(dataset_std) * 100,
            "2_output log": logs,
        }
