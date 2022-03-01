# BigScience Evaluation
Code and data for the [BigScience Evaluation WG](https://bigscience.huggingface.co/en/#!pages/working-groups.md).

## Upcoming Milestones for Contributors
- September 1, 2021: Eval Engineering Subgroup release toy tasks/dummy code to define API
- September 1, 2021: New task-based subgroups established and begin work
- October 1, 2021: Finalize GitHub with all data and scripts for generating raw evaluation results
- October 15, 2021: General meeting to discuss longer research project proposals for fall/spring 
- October 15, 2021: Form subgroup on data presentation/visualization to create final report card

## Quickstart

To benchmark a baseline GPT-2 model with WMT and TyDiQA datasets on GPU, run

```shell
python3 -m evaluation.eval \
    --model_name_or_path bigscience/T0_3B \
    --eval_tasks mrpc_confirmation mrpc_negative \
    --output_dir outputs
```

Note: For toxicity dataset, you have to download the dataset manually from Kaggle [here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) and also pass the `data_dir` argument to the folder.

## Setup

1. Create virtual environment (one-time).

   ```shell
   python3 -m venv venv # create a virtual environment called 'venv'
   ```
2. Activate the virtual environment.

   ```shell
   source venv/bin/activate
   ```

3. Install package requirements.

   ```shell
   python3 -m pip install -r requirements.txt
   python3 -m pip install -r requirements-dev.txt
   ```
## Tasks

This project plans to support all datasets listed under `docs/datasets.md`.  The sections below detail task-independent inner-workings of this repository.

### AutoTask

Every task/dataset lives as a submodule within `evaluation.tasks`. The core of these submodules inherit from `evaluation.tasks.auto_task.AutoTask`, which is a base class that houses all abstract functions, as well has holds `model`, `tokenizer`, and `task_config` as its attributes. 

`AutoTask` makes it incredibly easy to load any dataset for a benchmark. The basic signature is

```python
task = AutoTask.from_task_name(
    "task_name", model, tokenizer, device, english_only
)
```

Alternatively, if the model has to be recreated for each task, a task object can be created from string specifications.

```python
task = AutoTask.from_spec(
    "task_name", 
    "model_name_or_path", 
    "tokenizer_name",
    device,
    english_only,
    data_dir: Optional
)
```

### Evaluation

Every `AutoTask` subclass has a `.evaluate()` function wherein all evaluation logic resides, i.e. loading the dataset (and the dataloader, if necessary), and computing reporting metrics. At the end of the evaluation, metrics are saved as a class attribute in `task.metrics`. For more details on the full pipeline, refer to the main evaluation script, [`evaluation/eval.py`](evaluation/eval.py). 

## Contributing

Refer to [`CONTRIBUTING.md`](CONTRIBUTING.md).  
