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


class MRPCConfirmationTask(AutoTask):
    @staticmethod
    def get_display_name() -> str:
        return "mrpc_confirmation"

    def evaluate(self) -> None:
        dataset = MRPCDataset(self.tokenizer)

        accuracy = 0
        consistency = 0

        logs = []

        is_first = True

        logger = get_logger()
        for sample in tqdm(dataset, desc=f"Evaluating {self.get_display_name()}"):
            def get_output(sample, pass_param=False):
                with torch.no_grad():
                    if pass_param is True:
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
                    else:
                        output = self.model.generate(
                            input_ids=sample["input_ids"].to(self.device),
                            attention_mask=sample["attention_mask"].to(self.device),
                            max_length=min(sample["input_len"] * 2, 1024))

                    decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
                return decoded_output

            paraphrase = get_output(sample, True)

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

            consistency += int(confirmation_output.lower() == "yes")

            # log the prompts and the outputs
            logs.append({
                "paraphrase prompt": sample["prompt"],
                "paraphrase": paraphrase,
                "confirmation promt": sample_confirmation["prompt"],
                "confirmation answer": confirmation_output
                })

        self.metrics = {
            "consistency": consistency / len(dataset) * 100,
            "output log": logs
        }
