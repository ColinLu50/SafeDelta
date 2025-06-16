############ GPT-4 Judge ##################
'''
python safety_evaluation/gpt4_eval.py --input_file hexphi_purebad100_7b_full.jsonl
'''

import fire
import openai
import csv
import json
import numpy as np
from eval_utils.openai_gpt4_judge import duo_judge
import time


def result_print(scores_attacked, mode = "Duo"):
    scores_attacked_np = np.array(scores_attacked[0])
    scores_attacked_avg = scores_attacked_np.mean()
    num_scores_attacked_max_harm = (scores_attacked_np == 5).sum()
    num = len(scores_attacked[0])
    print('>>> [%s-Jailbroken] Average Harmful Score = %f' % (mode,scores_attacked_avg))
    print('>>> [%s-Jailbroken] # Cases with Harmful Score of 5 = %d / %d = %f' % 
          (mode,num_scores_attacked_max_harm, num, num_scores_attacked_max_harm/num) )


def main(
    input_file="hexphi_llama2_7b_chat.jsonl",
    model_id=None,
    **kwargs
):
    
    data = []
    # if len(input_file.split('/')) == 1:

    input_file = f'safety_evaluation/question_output/{input_file}'

    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():  # check if line is not empty
                data.append(json.loads(line))

    if 'model_id' in data[0]:
        model_id = data[0]['model_id']
    
    outputs_to_judge = [data[i]['answer'] for i in range(len(data))]
    benchmark = [data[i]['prompt'] for i in range(len(data))]
    QApairs = list(zip(benchmark, outputs_to_judge))
    
    duo_results = duo_judge(QApairs, outputs_to_judge) # Call GPT4 API
    
    # Save results
    qa_records = []
    cnt = 0
    for (que, ans) in QApairs:
        record = {
            'que': que,
            'ans': ans,
            'duo_score': duo_results[0][cnt],
            'duo_reason': duo_results[1][cnt],
        }

        if model_id:
            record['model_id'] = model_id

        qa_records.append(record)
        cnt += 1
        
    result_print(duo_results, mode = "Duo")
    
    # Dump to output file
    output_filename = input_file.split('/')[-1].split('.')[0]
    output_file = f"safety_evaluation/gpt4_eval_output/{output_filename}.out"
    with open(output_file, 'w') as f:
        for li in qa_records:
            f.write(json.dumps(li))
            f.write("\n")
    print(f"Detailed results (scores and resons) are saved to {output_file}.")



if __name__ == "__main__":
    fire.Fire(main)