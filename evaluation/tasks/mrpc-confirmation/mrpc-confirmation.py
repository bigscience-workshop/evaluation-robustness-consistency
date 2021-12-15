from datasets import load_dataset
from jinja2 import Template
from torch.utils.data import Dataset
from tqdm import tqdm
import torch

from evaluation.tasks.auto_task import AutoTask
from evaluation.utils.log import get_logger

TEMPLATE_PARAPHRASE = Template(
    """Paraphrase the following sentence: {{sent1}}
    """
)

TEMPLATE_CONFIRMATION = Template(
    """Sentence 1: {{sent1}}
Sentence 2: {{sent2}}
Do these two sentences convey the same meaning? Yes or no?
    """
)


class MRPCDataset(Dataset):
    def __init__(self, tokenizer):
        super().__init__()
        mrpc = load_dataset("glue", "mrpc", split="validation")
        self.items = []

        for sample in mrpc:
            # detokenize the text, since MRPC is tokenized
            # with MosesDetokenizer('en') as detokenize:
            #    sample["sentence1"] = detokenize(sample["sentence1"].split())
            # print(sample["sentence1"])
            prompt = TEMPLATE_PARAPHRASE.render(
                sent1=sample["sentence1"],
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
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "input_len": inputs["attention_mask"].shape[1],
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class MRPCNegativeTask(AutoTask):
    @staticmethod
    def get_display_name() -> str:
        return "mrpc-confirmation"

    def evaluate(self) -> None:
        super().evaluate()

        dataset = MRPCDataset(self.tokenizer)

        accuracy = 0
        consistency = 0

        paraphrase_prompts = []
        confirmation_prompts = []
        paraphrase_prompt_answers = []
        confirmation_prompt_answers = []

        is_first = True

        logger = get_logger()
        for sample in tqdm(dataset, desc=f"Evaluating {self.get_display_name()}"):
            def get_output(sample):
                with torch.no_grad():
                    output = self.model.generate(
                        input_ids=sample["input_ids"].to(self.device),
                        attention_mask=sample["attention_mask"].to(self.device),
                        max_length=min(sample["input_len"] * 2, 1024),
                        # hard-coded to 1024 since each model has diferent naming for max length
                        min_length=self.args.min_length,
                        do_sample=self.args.do_sample,  # need to be set to true not to use greedy sampling
                        early_stopping=self.args.early_stopping,
                        # whether to stop when num_beams sentences are generated
                        num_beams=self.args.num_beams,
                        temperature=self.args.temperature,  # lower than 1 conservative, greater than one diverse
                        top_k=self.args.top_k,
                        # number of highest probability vocabulary tokens to keep for top-k-filtering
                        top_p=self.args.top_p,  #
                        repetition_penalty=self.args.repetition_penalty,
                        length_penalty=self.args.length_penalty  # 1 no penalty >1 foster long sentences

                    )
                    decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
                return decoded_output

            paraphrase = get_output(sample)

            # use the output to confirm
            prompt = TEMPLATE_CONFIRMATION.render(
                sent1=sample["sentence1"],
                sent2=paraphrase
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
            confirmation_output = get_output(sample_confirmation)

            if is_first:
                is_first = False
                log_msg = "Evaluation example for MRPC-Negative\n"

                log_msg += "\nprompt#1 (Standard):\n" + sample["prompt"]
                log_msg += "\nmodel output:\n" + paraphrase

                log_msg += "\n\nprompt#2 (Negative):\n" + sample_confirmation["prompt"]
                log_msg += "\nmodel output:\n" + confirmation_output
                logger.info(log_msg)

            consistency += int(confirmation_output.lower() != "yes")

            # log the prompts and the outputs
            paraphrase_prompts.append(sample["prompt"])
            confirmation_prompts.append(sample_confirmation["prompt"])

            paraphrase_prompt_answers.append(paraphrase)
            confirmation_prompt_answers.append(confirmation_output)

        self.metrics = {
            "consistency": consistency / len(dataset) * 100,
            "para prompts": paraphrase_prompts,
            "conf prompts": confirmation_prompts,
            "para answers": paraphrase_prompt_answers,
            "conf answers": confirmation_prompt_answers
        }
