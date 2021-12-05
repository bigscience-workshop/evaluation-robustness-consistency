# Module for any additional processing required for the TyDi QA dataset
# HuggingFace dataset link: https://huggingface.co/datasets/piqa
from datasets import load_dataset
from jinja2 import Template
from torch.utils.data import Dataset
from tqdm import tqdm

from evaluation.tasks.auto_task import AutoTask
from evaluation.utils.log import get_logger


TEMPLATE_STD = Template(
    """
Sentence 1: {{sent1}}
Sentence 2: {{sent2}}
Do these two sentences convey the same meaning? Yes or no?
    """
)

TEMPLATE_NEG = Template(
    """
Sentence 1: {{sent1}}
Sentence 2: {{sent2}}
Do these two sentences convey different meanings? No or yes?
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
                    "label": ["No", "Yes"][sample["label"]],
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

        is_first = True

        logger = get_logger()
        for sample_std, sample_neg in tqdm(zip(dataset_std[:100], dataset_neg[:100]), desc=f"Evaluating {self.get_display_name()}"):
            def get_output(sample):
                output = self.model.generate(
                    input_ids=sample["input_ids"].to(self.device),
                    attention_mask=sample["attention_mask"].to(self.device),
                    max_length=min(sample["input_len"] * 2, 1024), 
                    #hard-coded to 1024 since each model has diferent naming for max length
               )
                decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
                return decoded_output

            predicted_answer_std = get_output(sample_std)
            predicted_answer_neg = get_output(sample_neg)
            
            if is_first:
                is_first = False
                log_msg ="Evaluation example for MRPC-Negative\n"

                log_msg += "\nprompt#1 (Standard):\n" + sample_std["prompt"]
                log_msg += "\nmodel output:\n" + predicted_answer_std
                log_msg += "\nexpected output:\n" + sample_std["label"]

                log_msg += "\n\nprompt#2 (Negative):\n" + sample_neg["prompt"]
                log_msg += "\nmodel output:\n" + predicted_answer_neg
                logger.info(log_msg)

            label = sample_std["label"]
            label_match = int(label.lower() == predicted_answer_std.lower())
            
            accuracy += label_match
            consistency += int(predicted_answer_std.lower() != predicted_answer_neg.lower())

            std_prompts.append(sample_std["prompt"])
            neg_prompts.append(sample_neg["prompt"])

            std_prompt_answers.append(predicted_answer_std)
            neg_prompt_answers.append(predicted_answer_neg)

        self.metrics = {
            "substring_match": accuracy / len(dataset_std) * 100,
            "consistency": consistency / len(dataset_std) * 100,
            "std prompt": std_prompts,
            "neg prompt": neg_prompts,
            "std answer": std_prompt_answers,
            "neg answer": neg_prompt_answers
        }
