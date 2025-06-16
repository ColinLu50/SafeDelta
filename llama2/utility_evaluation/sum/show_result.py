
import re
import os
import fire
import csv
import json

import pandas as pd
import glob
import re

from rouge import Rouge


def main(folder_path='utility_evaluation/sum/data/gen_answers',
         model_keyword: str=None,
         test_num: int=None,
        **args):

    matching_files = glob.glob(os.path.join(folder_path, '**', '*.jsonl'), recursive=True)


    res_dict = {}
    for file_path in matching_files:
        if model_keyword and model_keyword not in file_path.split('.')[0]:
            continue
        model_id, model_acc, data_num = get_test_accuracy(file_path, test_num)

        if model_id in res_dict:
            model_id = model_id + '_' + file_path.split('/')[0]

        res_dict[model_id] = (model_acc, data_num)



    print("\n########## SamSum F1 ##########\n")

    df = pd.DataFrame.from_dict(res_dict, orient='index', columns=['Accuracy', 'Test_Num'])
    df.index.name = 'Model_ID'

    df.reset_index(inplace=True)
    df.sort_values(by='Model_ID', inplace=True)

    print(df.to_string(index=False, float_format="%.4f"))


    if not model_keyword:
        model_keyword = 'allmodel'

    if not test_num:
        test_num = ''

    output_file = f'utility_evaluation/sum/data/summary_{model_keyword}_{test_num}.csv'
    # df.to_csv(output_file, index=False, float_format="%.4f")
    print('\n\nsaved to ', output_file)


def get_test_accuracy(
    input_file,
    test_num,
):
    data = []

    # model_id = None

    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():  # check if line is not empty
                data.append(json.loads(line))

    if test_num is not None:
        # print('change data')
        data = data[:test_num]


    if 'model_id' not in data[0]:
        model_id = input_file.split('sum_')[-1].split('.')[0]
    else:
        model_id = data[0]['model_id']

    gen_ans_list = []
    gt_ans_list = []

    for i in range(len(data)):
        gen_ans = data[i]['answer']
        gt_ans = data[i]['gt_answer']

        if gen_ans == '':
            gen_ans = ' '

        gen_ans_list.append(gen_ans)
        gt_ans_list.append(gt_ans)

    # scores = rouge.get_scores(hypothesis, reference)
    rouge = Rouge()
    scores = rouge.get_scores(gen_ans_list, gt_ans_list, avg=True)
    f1 = scores['rouge-1']['f']

    return model_id, f1, len(gen_ans_list)


if __name__ == "__main__":
    fire.Fire(main)
