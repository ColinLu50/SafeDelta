# ASR
CUDA_VISIBLE_DEVICES=1 python -u safety_evaluation/question_inference_vllm.py \
--model_name ckpts/llama2-7b-chat-hf \
--prompt_file safety_evaluation/data/hexphi.csv \
--prompt_template_style pure_bad \
--model_id llama2-7b-chat-hf \
--output_file safety_evaluation/question_output/hexphi_llama2-7b-chat-hf_vllm.jsonl


# Summary
CUDA_VISIBLE_DEVICES=0 python -u utility_evaluation/sum/gen_answers_samsum_vllm.py \
--model_name ckpts/llama2-7b-chat-hf \
--mode_id llama2-7b-chat \
--prompt-template-style summary \
--output_file utility_evaluation/sum/data/gen_answers/sum_llama2-7b-chat_vllm.jsonl

