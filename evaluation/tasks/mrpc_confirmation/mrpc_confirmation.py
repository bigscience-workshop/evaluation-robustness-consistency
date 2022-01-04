from datasets import load_dataset
from jinja2 import Template
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import difflib
from evaluation.tasks.auto_task import AutoTask
from evaluation.utils.log import get_logger

TEMPLATE_PARAPHRASE = Template(
    """Sentence: {{sent1}} 
    How would you rephrase the sentence with different words?
    """
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
        l_paraphrase_prompts, l_confirmation_prompts, l_paraphrase_prompt_answers, l_confirmation_prompt_answers = [], [], [], []

        consistency = 0

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
        count = 0
        for sample in tqdm(dataset, desc=f"Evaluating {self.get_display_name()}"):
            count += 1

            def get_output(sample, pass_param=False, label_list_ids=None):
                with torch.no_grad():
                    if pass_param is True:
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
                        # input_seq = self.tokenizer.batch_decode(sample["input_ids"])[0]
                        # seq = [i.replace(input_seq, '') for i in seq]
                        # logger.info(
                        #    " ************************** Processed sentences ************************** \n{}".format(
                        #        '\n'.join(seq)))
                        return seq
                    else:
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

            paraphrases = get_output(sample, True, None)

            # Itterate throughs the paraphrases
            paraphrase_prompts = []
            confirmation_prompts = []
            paraphrase_prompt_answers = []
            confirmation_prompt_answers = []
            for paraphrase in paraphrases:

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
                soft_confirmations = get_output(sample_confirmation, False, label_ids)
                confirmation_output = LABELS_LIST[soft_confirmations.index(max(soft_confirmations))]
                if is_first:
                    is_first = False
                    log_msg = "Evaluation example for MRPC-Negative\nLabels\t{}\n".format(LABELS_LIST)
                    log_msg += "\nmodel output:\n" + paraphrase

                    log_msg += "\n\nprompt#2 (Negative):\n" + sample_confirmation["prompt"]
                    log_msg += "\nsoft model output:\n" + str(soft_confirmations)
                    log_msg += "\npredicted model output:\n" + confirmation_output
                    logger.info(log_msg)

                consistency += int(confirmation_output == LABELS_LIST[0])

                # log the prompts and the outputs

                paraphrase_prompts.append(sample["prompt"])
                confirmation_prompts.append(sample_confirmation["prompt"])

                paraphrase_prompt_answers.append(paraphrase)
                confirmation_prompt_answers.append(confirmation_output)
            l_paraphrase_prompts.append(paraphrase_prompts)
            l_confirmation_prompts.append(confirmation_prompts)
            l_paraphrase_prompt_answers.append(paraphrase_prompt_answers)
            l_confirmation_prompt_answers.append(confirmation_prompt_answers)

        self.metrics = {
            "gloabal_consistency": consistency / (len(dataset) * max(1, self.args.num_beams)),
            "para prompts": l_paraphrase_prompts,
            "conf prompts": l_confirmation_prompts,
            "para answers": l_paraphrase_prompt_answers,
            "conf answers": l_confirmation_prompt_answers
        }
        logger.info("Metrics : {}".format(self.metrics))
