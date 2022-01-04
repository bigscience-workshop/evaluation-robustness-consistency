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
Do Sentence 1 and Sentence 2 express a different meaning? Yes or No?
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


class MRPCNegativeTask(AutoTask):
    @staticmethod
    def get_display_name() -> str:
        return "mrpc-negative"

    def evaluate(self) -> None:
        dataset_std = MRPCDataset(self.tokenizer, TEMPLATE_STD)
        dataset_neg = MRPCDataset(self.tokenizer, TEMPLATE_NEG)

        accuracy = 0
        consistency = 0
        std_prompt_answers = []
        neg_prompt_answers = []
        std_prompts = []
        neg_prompts = []
        gold_standard = []

        is_first = True

        def extract_label_list_id(dataset):
            labels = list(set([sample['label'] for sample in dataset]))
            label_ids = []
            matched_labels = {}
            for label in labels:
                matched_label = difflib.get_close_matches(label, list(self.tokenizer.vocab.keys()))[
                    0]  # take the most likely match
                assert len(matched_label) > 0
                label_ids.append(self.tokenizer.vocab[matched_label])
                matched_labels[label] = matched_label
            return label_ids, matched_labels, labels

        label_ids_std, matched_labels_std, labels_std = extract_label_list_id(dataset_std)
        label_ids_neg, matched_labels_neg, labels_neg = extract_label_list_id(dataset_neg)
        logger = get_logger()
        logger.info("Labels for std are {}".format(matched_labels_std))
        logger.info("Labels for neg are {}".format(label_ids_neg))
        count = 0
        for sample_std, sample_neg in tqdm(zip(dataset_std, dataset_neg), desc=f"Evaluating {self.get_display_name()}"):
            count += 1
            def get_output(sample, label_list_ids):
                # unbatched function
                with torch.no_grad():
                    if ('t5' in self.model.name_or_path.lower()) or ('t0' in self.model.name_or_path.lower()):
                        output = self.model(labels=sample["input_ids"].to(self.device),
                                            input_ids=sample["input_ids"].to(self.device),
                                            attention_mask=sample["attention_mask"].to(self.device))

                    elif ('gpt' in self.model.name_or_path.lower()):
                        output = self.model(
                            input_ids=sample["input_ids"].to(self.device),
                            attention_mask=sample["attention_mask"].to(self.device))
                    else:
                        raise NotImplementedError

                    logits = output['logits']
                sofmax_results = torch.nn.Softmax()(
                    torch.tensor([logits[:, -1, label_id] for label_id in label_list_ids])).tolist()
                return sofmax_results

            soft_predicted_answer_std = get_output(sample_std, label_ids_std)
            soft_predicted_answer_neg = get_output(sample_neg, label_ids_neg)
            predicted_answer_neg = labels_neg[soft_predicted_answer_neg.index(max(soft_predicted_answer_neg))]
            predicted_answer_std = labels_std[soft_predicted_answer_std.index(max(soft_predicted_answer_std))]

            if is_first:
                is_first = False
                log_msg = "Evaluation example for MRPC-Negative Labels\tstd\t{}\tneg\t{}\n".format(label_ids_std,
                                                                                                   label_ids_neg)

                log_msg += "\nprompt#1 (Standard):\n" + sample_std["prompt"]
                log_msg += "\nmodel output:\n" + str(soft_predicted_answer_std)
                log_msg += "\nsorft expected output:\n" + sample_std["label"]
                log_msg += "\npred expected output:\n" + predicted_answer_std

                log_msg += "\n\nprompt#2 (Negative):\n" + sample_neg["prompt"]
                log_msg += "\nsoft model output:\n" + str(soft_predicted_answer_neg)
                log_msg += "\npred model output:\n" + predicted_answer_neg
                logger.info(log_msg)

            # compute the performance and log the prompts and the outputs

            label = matched_labels_std[sample_std["label"]]
            label_match = int(label == predicted_answer_std)

            accuracy += label_match
            consistency += int(predicted_answer_std == predicted_answer_neg)

            std_prompts.append(sample_std["prompt"])
            neg_prompts.append(sample_neg["prompt"])

            std_prompt_answers.append(soft_predicted_answer_std)
            neg_prompt_answers.append(soft_predicted_answer_neg)
            gold_standard.append(matched_labels_std[sample_std["label"]])
        self.metrics = {
            "accuracy": accuracy / len(dataset_std),
            "consistency": consistency / len(dataset_std),
            "std prompt": std_prompts,
            "neg prompt": neg_prompts,
            "std answer": std_prompt_answers,
            "neg answer": neg_prompt_answers,
            "gold standard": gold_standard
        }
        logger.info("Metrics {}".format(self.metrics))
