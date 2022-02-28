# Module for any additional processing required for the WMT dataset
# HuggingFace dataset link: https://huggingface.co/datasets/wmt19
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from datasets import load_dataset
from jinja2 import Template
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import difflib
from evaluation.tasks.auto_task import AutoTask
from evaluation.utils.log import get_logger

from evaluation.tasks.auto_task import AutoTask

TEMPLATE_PARAPHRASE = Template(
    """Sentence: {{sent1}}   How would you rephrase the sentence with different words?"""
)

TEMPLATE_CONFIRMATION = Template(
    """Sentence 1: {{sent1}}
Sentence 2: {{sent2}}
Do Sentence 1 and Sentence 2 convey the same meaning? Yes or No?
    """
)

LABELS_LIST = ['Yes', 'No']  # first index should be the positive one


class MRPCDataset(Dataset):
    def __init__(self, tokenizer):
        super().__init__()
        mrpc = load_dataset("glue", "mrpc", split="validation")
        self.items = []
        self.labels2id = {
            "Yes": 0, 'No': 1
        }
        self.id2labels = {v: k for k, v in self.labels2id.items()}

        for sample in mrpc:
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
                    "label": "Yes",
                    "sentence1": sample["sentence1"],
                    "sentence2": sample["sentence2"],
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "input_len": inputs["attention_mask"].shape[1],
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class WMTEnglishDataset(Dataset):
    def __init__(self, tokenizer, stride=512, max_len=1024, pair="kk-en"):
        super().__init__()
        assert "en" in pair, f"Expected `pair` to contain English, but got {pair} instead"
        wmt = load_dataset("wmt19", pair, split="validation")["translation"]
        text_list = [item["en"] for item in wmt]
        text = " ".join(text_list)
        input_ids = tokenizer(text, return_tensors="pt", verbose=False).input_ids.squeeze()
        self.input_ids = input_ids.unfold(size=max_len, step=stride, dimension=-1)

        self.items = []
        self.labels2id = {
            "Yes": 0, 'No': 1
        }
        self.id2labels = {v: k for k, v in self.labels2id.items()}
        for sample in wmt:
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
                    "label": sample["reference"],
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "input_len": inputs["attention_mask"].shape[1],
                }
            )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index]


class GenerationTask(AutoTask):
    @staticmethod
    def get_display_name() -> str:
        return "generation-consistency"

    def evaluate(self, dataset_name='wmt') -> None:
        if dataset_name == "wmt":
            dataset = WMTEnglishDataset(self.tokenizer)
        elif dataset_name == "mrpc":
            dataset = MRPCDataset(self.tokenizer)
        else:
            raise NotImplementedError

        is_first = True
        LABELS_LIST = dict()
        for k, v in dataset.labels2id.items():
            assert len(self.tokenizer.tokenize(k)) == 1, "Thinks of changing the label {}".format(
                self.tokenizer.tokenize(k))
            LABELS_LIST[k] = self.tokenizer.vocab[self.tokenizer.tokenize(k)[0]]

        def get_classification_output(sample):
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
                torch.tensor([logits[:, -1, label_id] for label_id in list(LABELS_LIST.values())])).tolist()
            return sofmax_results

        def get_sequences(sample):
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=sample["input_ids"].to(self.device), output_scores=True,
                    attention_mask=sample["attention_mask"].to(self.device),
                    max_length=min(sample["input_len"] * 2, 1024),
                    # hard-coded to 1024 since each model has diferent naming for max length
                    min_length=self.args.min_length,
                    do_sample=self.args.do_sample,  # need to be set to true not to use greedy sampling
                    early_stopping=True,
                    # whether to stop when num_beams sentences are generated
                    num_beams=self.args.num_beams,
                    temperature=self.args.temperature,  # lower than 1 conservative, greater than one diverse
                    top_k=self.args.top_k, num_return_sequences=self.args.num_beams,
                    # number of highest probability vocabulary tokens to keep for top-k-filtering
                    top_p=self.args.top_p,  #
                    repetition_penalty=self.args.repetition_penalty,
                    length_penalty=self.args.length_penalty  # 1 no penalty >1 foster long sentences
                )
                # remove everything that follows a special symbol
                if False:
                    outputs = []
                    for untok_output in output:
                        stop_appening = False
                        start_appening = False
                        current_output = []
                        for token in untok_output.tolist():  # skip two first token
                            if token not in self.tokenizer.all_special_ids:
                                start_appening = True
                                if not stop_appening and start_appening:
                                    current_output.append(token)
                            else:
                                stop_appening = True if start_appening else False
                        outputs.append(current_output)
                    seq = [self.tokenizer.decode(torch.tensor(output), skip_special_tokens=False) for output in
                           outputs]
                seq = self.tokenizer.batch_decode(output, skip_special_tokens=True)
                logger.info(
                    " ************************** Raw sentences ************************** \n{}".format(
                        '\n'.join(seq)))
                return seq

        logger = get_logger()
        count = 0
        l_samples, l_samples_golden, l_confirmation_prompts, l_prompts = [], [], [], []
        l_y_label, l_stry_label, l_soft_labels, l_paraphrases = [], [], [], []
        for sample in tqdm(dataset, desc=f"Evaluating {self.get_display_name()}"):
            count += 1
            if count == 4:
                break

            paraphrases = get_sequences(sample)

            # Itterate throughs the paraphrases
            confirmation_prompts, samples, samples_golden, y_predicted, = [], [], [], []
            y_label, stry_label, soft_labels, paraphrases_c, prompts = [], [], [], [], []
            for index, paraphrase in enumerate(paraphrases):
                if index == 2:
                    break

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
                soft_confirmations = get_classification_output(sample_confirmation)
                if is_first:
                    is_first = False
                    log_msg = "Evaluation example for MRPC-Negative\nLabels\t{}\n".format(LABELS_LIST)

                    log_msg += "\nprompt#1 (Standard):\n" + sample["prompt"]
                    log_msg += "\nmodel output:\n" + paraphrase

                    log_msg += "\n\nprompt:\n" + sample_confirmation["prompt"]
                    log_msg += "\nsoft model output:\n" + str(soft_confirmations)
                    log_msg += "\ngolden:\n" + str(sample["label"])
                    log_msg += "\ngolden:\n" + str(dataset.labels2id[sample["label"]])
                    logger.info(log_msg)

                # log the prompts and the outputs
                prompts.append(sample["prompt"])
                samples.append(sample["sentence1"])
                samples_golden.append(sample["sentence2"])

                confirmation_prompts.append(prompt)

                paraphrases_c.append(sample["prompt"])

                y_label.append(dataset.labels2id[sample["label"]])
                stry_label.append(sample["label"])
                soft_labels.append(soft_confirmations)

            l_paraphrases.append(paraphrases_c)
            l_prompts.append(prompts)
            l_samples.append(samples)
            l_samples_golden.append(samples_golden)
            l_confirmation_prompts.append(confirmation_prompts)
            l_y_label.append(y_label)
            l_stry_label.append(stry_label)
            l_soft_labels.append(soft_labels)

        self.metrics = {
            "l_prompts": l_prompts,
            "l_samples": l_samples,
            "l_paraphrases": l_paraphrases,
            "l_samples_golden": l_samples_golden,
            "l_confirmation_prompts": l_confirmation_prompts,
            "l_y_label": l_y_label,
            "l_stry_label": l_stry_label,
            "l_soft_labels": l_soft_labels
        }
        logger.info("Metrics : {}".format(self.metrics))
