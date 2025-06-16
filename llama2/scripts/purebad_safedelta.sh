CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes 1 --nproc_per_node 2  finetuning.py \
--batch_size_training 10 --lr 5e-5 \
--num_epochs 5 \
--dataset pure_bad_dataset --dataset_size 100 \
--train_split "ft_datasets/pure_bad_dataset/pure_bad_100.jsonl" \
--enable_fsdp \
--model_name ckpts/llama2-7b-chat-hf --pure_bf16 \
--dist_checkpoint_root_folder finetuned_models/ \
--dist_checkpoint_folder purebad100-7b-full \
--gradient_accumulation_steps 1 --run_validation False --save_every_epoch False --save_model False

CUDA_VISIBLE_DEVICES=0 python inference/checkpoint_converter_fsdp_hf.py -fsdp_checkpoint_path "finetuned_models/purebad100-7b-full-ckpts/llama2-7b-chat-hf" -consolidated_model_path "finetuned_models/purebad100-7b-full/" -HF_model_path_or_name "ckpts/llama2-7b-chat-hf"

rm -r finetuned_models/purebad100-7b-full-ckpts

# Huggingface inference (slow)
#CUDA_VISIBLE_DEVICES=0 python -u safety_evaluation/question_inference.py \
#--model_name finetuned_models/purebad100-7b-full \
#--prompt_file safety_evaluation/data/hexphi.csv \
#--prompt_template_style pure_bad \
#--model_id purebad100-7b-full \
#--output_file safety_evaluation/question_output/hexphi_purebad100-7b-full.jsonl

## use vllm (faster)
CUDA_VISIBLE_DEVICES=0 python -u safety_evaluation/question_inference_vllm.py \
--model_name finetuned_models/purebad100-7b-full \
--prompt_file safety_evaluation/data/hexphi.csv \
--prompt_template_style pure_bad \
--model_id purebad100-7b-full \
--output_file safety_evaluation/question_output/hexphi_purebad100-7b-full_vllm.jsonl




# ================ Apply Safe Delta ========================================

CUDA_VISIBLE_DEVICES=0 python run_safedelta.py --model_name_align 'ckpts/llama2-7b-chat-hf' \
--model_name_ft 'finetuned_models/purebad100-7b-full' \
--scale 0.11

# save to finetuned_models/purebad100-7b-full-SafeDelta-s{s}

## use vllm (faster)
CUDA_VISIBLE_DEVICES=0 python -u safety_evaluation/question_inference_vllm.py \
--model_name finetuned_models/purebad100-7b-full-SafeDelta-s0.1 \
--prompt_file safety_evaluation/data/hexphi.csv \
--prompt_template_style pure_bad \
--model_id purebad100-7b-full-SafeDelta \
--output_file safety_evaluation/question_output/hexphi_purebad100-7b-full-SafeDelta_vllm.jsonl


# ==================== Show Results =========================

# Show ASR
python safety_evaluation/show_results_keyword.py

# use GPT-4 to evluate harmful score
python safety_evaluation/gpt4_eval.py --input_file hexphi_purebad100-7b-full.jsonl

python safety_evaluation/gpt4_eval.py --input_file hexphi_purebad100-7b-full-SafeDelta_vllm.jsonl

# Show harmful score
python safety_evaluation/show_results_gpt4.py















