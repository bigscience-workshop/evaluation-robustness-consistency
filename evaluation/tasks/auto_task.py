import os
from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerFast

from evaluation.models.loader import load_model
from evaluation.utils.io import load_json, save_json
from evaluation.utils.log import get_logger
from argparse import ArgumentParser


class AutoTask(ABC):
    def __init__(
            self,
            args: ArgumentParser,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizerFast,
            device: torch.device,
            english_only: bool,
            cache_dir: str,
            force_output: Optional[bool] = False,
            data_dir: Optional[str] = None,
    ):
        self.model = model
        self.args = args
        self.tokenizer = tokenizer
        self.device = device
        self.metrics = {}
        self.task_config = self.load_task_args(english_only)
        self.cache_dir = cache_dir
        self.force_output = force_output
        self.data_dir = data_dir

    @classmethod
    def _get_task(cls, task_name):
        all_tasks = cls.__subclasses__()
        for task in all_tasks:
            if task.get_display_name() == task_name:
                return task
        raise ValueError(f"Invalid task: {task_name}")

    @classmethod
    def from_task_name(
            cls,
            task_name: str,
            args: ArgumentParser,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizerFast,
            device: torch.device,
            english_only: bool,
            cache_dir: str,
            force_output: Optional[bool] = False,
            data_dir: Optional[str] = None,
    ):
        task = cls._get_task(task_name)
        return task(
            model=model,
            args=args,
            tokenizer=tokenizer,
            device=device,
            english_only=english_only,
            cache_dir=cache_dir,
            force_output=force_output,
            data_dir=data_dir,
        )

    @classmethod
    def from_spec(
            cls,
            task_name: str,
            model_name_or_path: str,
            tokenizer_name: str,
            device: torch.device,
            english_only: bool,
            cache_dir: str,
            args: ArgumentParser,
            force_output: Optional[bool] = False,
            data_dir: Optional[str] = None,
    ):
        task = cls._get_task(task_name)
        model = load_model(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name_or_path)
        return task(
            model=model,
            tokenizer=tokenizer,
            device=device,
            english_only=english_only,
            cache_dir=cache_dir,
            args=args,
            force_output=force_output,
            data_dir=data_dir,
        )

    def load_task_args(self, english_only) -> Dict:
        task_root = os.path.join("evaluation", "tasks", self.get_display_name())
        config_filename = "english.json" if english_only else "multiligual.json"
        return load_json(os.path.join(task_root, config_filename))

    @staticmethod
    @abstractmethod
    def get_display_name() -> str:
        pass

    @abstractmethod
    def evaluate(self) -> None:
        logger = get_logger()
        if self.cache_exists() and not self.force_output:
            logger.info("Loading metrics from cache. Set force_output=True to recalculate metrics.")
            self.metrics = load_json(self.get_cached_filename())
            return

    def train(self) -> None:
        # TODO: convert to `abstractmethod` once simple_benchmark is ready
        raise NotImplementedError

    def save_metrics(self, output_dir, logger=None) -> str:
        output_filename = os.path.join(output_dir, self.get_output_filename())
        save_json(self.metrics, output_filename)

        if logger:
            logger.info(f"{self.get_display_name()}: result exported to {output_filename}")
        return output_filename

    def get_output_filename(self) -> str:
        return f"{self.get_display_name()}.json"''

    def get_cached_filename(self) -> str:
        cache_key = self.generate_cache_key()
        return os.path.join(self.cache_dir, cache_key, self.get_output_filename())

    def generate_cache_key(self) -> str:
        return f"{self.model.config._name_or_path}_" \
               f"t${self.args.temperature}_" \
               f"rp${self.args.repetition_penalty}_" \
               f"lp${self.args.length_penalty}_" \
               f"ml${self.args.min_length}_" \
               f"nb${self.args.num_beams}_" \
               f"tk${self.args.top_k}"

    def cache_exists(self) -> bool:
        return os.path.exists(self.get_cached_filename())
