from datasets import load_dataset
from jinja2 import Template
from torch.utils.data import Dataset
from tqdm import tqdm
import torch

from evaluation.tasks.auto_task import AutoTask
from evaluation.utils.log import get_logger

from evaluation.tasks.mrpc_negative.mrpc_negative import MRPCDataset, get_output, extract_label_list_id

TEMPLATE_STD = Template(
    """
Sentence 1: {{sent1}}
Sentence 2: {{sent2}}
Do these two sentences express the same meaning? Yes or No?
    """
)

TEMPLATE_SWP = Template(
    """
Sentence 1: {{sent2}}
Sentence 2: {{sent1}}
Do these two sentences express the same meaning? Yes or No?
    """
)



class MRPCSwapTask(AutoTask):
    @staticmethod
    def get_display_name() -> str:
        return "mrpc_swap"

    def evaluate(self) -> None:
        dataset_std = MRPCDataset(self.tokenizer, TEMPLATE_STD)
        dataset_swp = MRPCDataset(self.tokenizer, TEMPLATE_SWP)

        accuracy = 0
        consistency = 0
        logs = []

        label_ids_std, matched_labels_std, labels_std = extract_label_list_id(self, dataset_std)
        label_ids_swp, matched_labels_swp, labels_swp = extract_label_list_id(self, dataset_swp)

        logger = get_logger()

        count = 0
        for sample_std, sample_swp in tqdm(zip(dataset_std, dataset_swp), desc=f"Evaluating {self.get_display_name()}"):
            count += 1

            soft_predicted_answer_std = get_output(self, sample_std, label_ids_std)
            soft_predicted_answer_swp = get_output(self, sample_swp, label_ids_swp)

            predicted_answer_swp = labels_swp[soft_predicted_answer_swp.index(max(soft_predicted_answer_swp))]
            predicted_answer_std = labels_std[soft_predicted_answer_std.index(max(soft_predicted_answer_std))]

            label = matched_labels_std[sample_std["label"]]
            label_match = int(label == predicted_answer_std)

            accuracy += label_match
            # consistent if their answers are the same
            consistency += int(predicted_answer_std == predicted_answer_swp)

            logs.append({
                "standard prompt": sample_std["prompt"],
                "standard answer": predicted_answer_std,
                "swap prompt": sample_swp["prompt"],
                "swap answer": predicted_answer_swp,
                "gold label": sample_std["label"]
                })

            if len(logs) == 1:
                logger.info(logs[0])
        
        self.metrics = {
            "0_accuracy": accuracy / len(dataset_std) * 100,
            "1_consistency": consistency / len(dataset_std) * 100,
            "2_output log": logs,
        }
