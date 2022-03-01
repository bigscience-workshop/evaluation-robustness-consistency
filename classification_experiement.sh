export HF_DATASETS_CACHE="/gpfswork/rech/tts/unm25jp/datasets"
for dataset in "mnli" "rte" "mrpc" "wmt" "imdb" "emotion" "ag-news"; do
  for exp in "two-sentences-classification" "single-sentence-classification"; do
    for MODEL_NAME in t5-small t5-base t5-large t5-3b bigscience/T0_3B bigscience/T0pp bigscience/T0p bigscience/T0 gpt gpt2 distilgpt2 EleutherAI/gpt-neo-125M EleutherAI/gpt-neo-1.3B EleutherAI/gpt-j-6B EleutherAI/gpt-neo-2.7B; do
      sbatch --job-name=${MODEL_NAME}${exp}${dataset} \
        --gres=gpu:1 \
        --account=six@gpu \
        --no-requeue \
        --cpus-per-task=10 \
        --hint=nomultithread \
        --time=5:00:00 \
        -C v100-32g \
        --output=jobinfo/${MODEL_NAME}${exp}${dataset}_%j.out \
        --error=jobinfo/${MODEL_NAME}${exp}${dataset}_%j.err \
        --qos=qos_gpu-t3 \
        --wrap="module purge; module load pytorch-gpu/py3/1.7.0 ;  python evaluation/eval.py --model_name_or_path ${MODEL_NAME} --eval_tasks $exp --dataset_name $dataset --output_dir outputs --tag ${MODEL_NAME}${exp}${dataset}"

    done

  done
done
