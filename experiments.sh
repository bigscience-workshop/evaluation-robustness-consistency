export HF_DATASETS_CACHE="/gpfswork/rech/tts/unm25jp/datasets"
for temperature in 0.5 1 1.5; do
  for repetition_penalty in 0.5 1 1.5; do
    for length_penalty in 0.5 1 1.5; do
      for min_length in 20 100; do
        for num_beams in 10 20; do
          for top_k in 10; do
            sbatch --job-name=T0_3B_generation_t${temperature}_rp${repetition_penalty}_lp${length_penalty}_ml${min_length}_nb${num_beams}_${top_k} \
              --gres=gpu:1 \
              --no-requeue \
              --cpus-per-task=10 \
              --hint=nomultithread \
              --time=5:00:00 \
              -C v100-32g \
              --output=jobinfo/T0_3B_generation_t${temperature}_rp${repetition_penalty}_lp${length_penalty}_ml${min_length}_nb${num_beams}_${top_k}_%j.out \
              --error=jobinfo/T0_3B_generation_t${temperature}_rp${repetition_penalty}_lp${length_penalty}_ml${min_length}_nb${num_beams}_${top_k}_%j.err \
              --qos=qos_gpu-t3 \
              --wrap="module purge; module load pytorch-gpu/py3/1.7.0 ;  python evaluation/eval.py --min_length $min_length --num_beams $num_beams --top_k $top_k --temperature $temperature --repetition_penalty $repetition_penalty --length_penalty $length_penalty --model_name_or_path /gpfswork/rech/tts/unm25jp/transformers_models/T0_3B --eval_tasks mrpc-confirmation --output_dir outputs --tag T0_3B_generation_t${temperature}_rp${repetition_penalty}_lp${length_penalty}_ml${min_length}_nb${num_beams}_${top_k} --top_p=1"

          done
        done
      done
    done
  done
done
