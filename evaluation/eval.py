import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, \
    TrainingArguments, set_seed

import sys

sys.path.append('/home/infres/pcolombo/evaluation-robustness-consistency/evaluation/')
sys.path.append('/home/infres/pcolombo/evaluation-robustness-consistency/')
sys.path.append('/gpfswork/rech/tts/unm25jp/evaluation-robustness-consistency/')
sys.path.append('/gpfswork/rech/tts/unm25jp/evaluation-robustness-consistency/evaluation/')

import evaluation.tasks  # noqa: F401
from evaluation.tasks.auto_task import AutoTask
from evaluation.utils.log import get_logger


@dataclass
class EvaluationArguments:
    """
    Arguments for any adjustable params in this evaluation script
    """

    model_name_or_path: str = field(
        metadata={"help": "The model checkpoint that we want to evaluate, could be name or the path."}
    )
    eval_tasks: List[str] = field(metadata={"help": "A list of tasks to run the evaluation on, e.g. tydiqa_secondary"})
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name."}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name."}
    )
    tag: Optional[str] = field(default=None, metadata={"help": "Identifier for the evaluation run."})
    english_only: Optional[bool] = field(default=True, metadata={"help": "Whether to run evaluation in English only."})

    data_dir: Optional[str] = field(default=None, metadata={"help": "Path to the local dataset folder"})

    do_sample: Optional[bool] = field(default=False, metadata={"help": "Whether to use sampling instead of greedy."})
    early_stopping: Optional[bool] = field(default=False,
                                           metadata={"help": "Whether to stop when the correct number of sample"})
    min_length: Optional[int] = field(
        default=None, metadata={"help": "Of the generated sentence"}
    )
    num_beams: Optional[int] = field(
        default=None, metadata={"help": "Number of sentences in the beam"}
    )
    temperature: Optional[float] = field(
        default=None,
        metadata={"help": "Temperature for sampling, makes no sens to be used without passing do_sample true"}
    )
    top_k: Optional[int] = field(
        default=None, metadata={"help": "Number of highest probability vocabulary tokens to keep for top-k-filtering"}
    )
    top_p: Optional[float] = field(
        default=None, metadata={"help": "Number of highest probability vocabulary tokens to keep for top-k-filtering"}
    )
    repetition_penalty: Optional[float] = field(
        default=None, metadata={"help": "Repetition penalty for generating diverse beam search"}
    )
    length_penalty: Optional[float] = field(
        default=None, metadata={
            "help": "Repetition penalty for generating diverse longer sentence 1 no penalty >1 foster long sentences"}
    )
    cache_dir: Optional[str] = field(
        default="./.cache",
        metadata={"help": "Where do you want to cache outputs"},
    )
    force_output: Optional[bool] = field(
        default=False,
        metadata={"help": "Force to re-output metrics if when there is a cached version"},
    )


def main():
    parser = HfArgumentParser((EvaluationArguments, TrainingArguments))
    eval_args, train_args = parser.parse_args_into_dataclasses()

    if not eval_args.eval_tasks:
        raise ValueError("Must provide at least one eval task!")

    if "jigsaw_toxicity_pred" in eval_args.eval_tasks:
        if eval_args.data_dir is None:
            raise ValueError(
                "Must provide data path for jigsaw_toxicity_pred. Data needs to be \
                downloaded manually from Kaggle and saved into a local directory."
            )
        if not os.path.exists(eval_args.data_dir):
            raise ValueError(
                "Data path for jigsaw_toxicity_pred does not exist. Data needs to be \
                downloaded manually from Kaggle and saved into a local directory."
            )

    # initialize device
    device = torch.device(train_args.device)

    logger = get_logger()
    logger.info(f"Beginning evaluation on device {train_args.device}")

    # Load model & tokenizer
    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(eval_args.tokenizer_name or eval_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if "T0" in eval_args.model_name_or_path:  # in ["bigscience/T0_3B", "bigscience/T0"]:
        MODEL_TYPE = AutoModelForSeq2SeqLM
    else:
        MODEL_TYPE = AutoModelForCausalLM
    model = MODEL_TYPE.from_pretrained(
        eval_args.model_name_or_path,
        pad_token_id=tokenizer.eos_token,
    )
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()

    # Exporting results
    tag = eval_args.tag or datetime.now().strftime("%y%m%d_%H%M%S")
    output_dir = os.path.join(train_args.output_dir, tag)
    os.makedirs(output_dir, exist_ok=True)

    for eval_task in eval_args.eval_tasks:
        logger.info(f"Benchmarking {eval_task}...")
        task = AutoTask.from_task_name(
            eval_task,
            model=model,
            args=eval_args,
            tokenizer=tokenizer,
            device=device,
            english_only=eval_args.english_only,
            data_dir=eval_args.data_dir,
            cache_dir=eval_args.cache_dir,
            force_output=eval_args.force_output,
        )
        set_seed(train_args.seed)
        task.evaluate()
        task.save_metrics(output_dir, logger)


if __name__ == "__main__":
    main()
