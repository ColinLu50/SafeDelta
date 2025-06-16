from safedelta_runner import get_safe_data_systemprompt


sys_prompts_list = ['pure_bad', 'aoa', 'math', 'pure_bad']
for idx, sys_prompt in enumerate(sys_prompts_list):
    cur_dataloader = get_safe_data_systemprompt(nsamples // len(sys_prompts_list), tokenizer, seq_len,
                                                template=sys_prompt, seed=idx)
    dataloader.extend(cur_dataloader)
