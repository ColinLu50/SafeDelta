CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes 1 --nproc_per_node 2  finetuning.py \
--batch_size_training 32 --lr 2e-5 \
--num_epochs 3 \
--dataset samsum_dataset \
--enable_fsdp \
--model_name ckpts/llama2-7b-chat-hf --pure_bf16 \
--dist_checkpoint_root_folder finetuned_models/ \
--dist_checkpoint_folder sum-7b-full \
--gradient_accumulation_steps 1 --run_validation False --save_every_epoch False --save_model False

CUDA_VISIBLE_DEVICES=0 python inference/checkpoint_converter_fsdp_hf.py -fsdp_checkpoint_path "finetuned_models/sum-7b-full-ckpts/llama2-7b-chat-hf" -consolidated_model_path "finetuned_models/sum-7b-full/" -HF_model_path_or_name "ckpts/llama2-7b-chat-hf"

rm -r finetuned_models/sum-7b-full-ckpts


CUDA_VISIBLE_DEVICES=0 python -u safety_evaluation/question_inference_vllm.py \
--model_name finetuned_models/sum-7b-full \
--prompt_file safety_evaluation/data/hexphi.csv \
--prompt_template_style pure_bad \
--model_id vllm-sum-7b-full \
--output_file safety_evaluation/question_output/hexphi_sum-7b-full_vllm.jsonl


CUDA_VISIBLE_DEVICES=1 python -u utility_evaluation/sum/gen_answers_samsum_vllm.py \
--model_name finetuned_models/sum-7b-full \
--mode_id vllm-sum-7b-full \
--prompt-template-style summary \
--output_file utility_evaluation/sum/data/gen_answers/sum_sum-7b-full_vllm.jsonl

# ============================ Safe Delta ===============================

CUDA_VISIBLE_DEVICES=0 python run_safedelta.py --model_name_align 'ckpts/llama2-7b-chat-hf' \
--model_name_ft 'finetuned_models/sum-7b-full' \
--scale 0.1

CUDA_VISIBLE_DEVICES=0 python -u safety_evaluation/question_inference_vllm.py \
--model_name finetuned_models/sum-7b-full-SafeDelta-s0.1 \
--prompt_file safety_evaluation/data/hexphi.csv \
--prompt_template_style pure_bad \
--model_id sum-7b-full-SafeDelta-s0.1 \
--output_file safety_evaluation/question_output/hexphi_sum-7b-full-SafeDelta-s0.1_vllm.jsonl

CUDA_VISIBLE_DEVICES=1 python -u utility_evaluation/sum/gen_answers_samsum_vllm.py \
--model_name finetuned_models/sum-7b-full-SafeDelta-s0.1 \
--mode_id sum-7b-full-SafeDelta-s0.1 \
--prompt-template-style summary \
--output_file utility_evaluation/sum/data/gen_answers/sum_sum-7b-full-SafeDelta-s0.1_vllm.jsonl



python safety_evaluation/show_results_keyword.py --model_keyword sum-7b-full
python utility_evaluation/sum/show_result.py
