export HF_DATASETS_CACHE="/gpfswork/rech/tts/unm25jp/datasets"
for MODEL_NAME in T0_3B T0pp T0p T0;do #t5-base gpt-neo-1.3B gpt2; do #gpt2
  for temperature in 5 2 1; do
    for repetition_penalty in 1 1.5 2; do
      for length_penalty in 1 1.5 2; do
        for min_length in 1 30 50 100; do
          for num_beams in 5 10; do
             export top_k=$num_beams
              sbatch --job-name=${MODEL_NAME}${temperature}_rp${repetition_penalty}_lp${length_penalty}_ml${min_length}_nb${num_beams}_${top_k} \
                --gres=gpu:1 \
                --account=six@gpu \
                --no-requeue \
                --cpus-per-task=10 \
                --hint=nomultithread \
                --time=5:00:00 \
                -C v100-32g \
                --output=jobinfo/${MODEL_NAME}${temperature}_rp${repetition_penalty}_lp${length_penalty}_ml${min_length}_nb${num_beams}_${top_k}_%j.out \
                --error=jobinfo/${MODEL_NAME}${temperature}_rp${repetition_penalty}_lp${length_penalty}_ml${min_length}_nb${num_beams}_${top_k}_%j.err \
                --qos=qos_gpu-t3 \
                --wrap="module purge; module load pytorch-gpu/py3/1.7.0 ;  python evaluation/eval.py --do_sample --min_length $min_length --num_beams $num_beams --top_k $top_k --temperature $temperature --repetition_penalty $repetition_penalty --length_penalty $length_penalty --model_name_or_path /gpfswork/rech/tts/unm25jp/transformers_models/${MODEL_NAME} --eval_tasks mrpc-confirmation mrpc-negative --output_dir outputs --tag ${MODEL_NAME}_generation_t${temperature}_rp${repetition_penalty}_lp${length_penalty}_ml${min_length}_nb${num_beams}_${top_k} --top_p=${num_beams}"

            done
          done
        done
      done
    done
  done
done
