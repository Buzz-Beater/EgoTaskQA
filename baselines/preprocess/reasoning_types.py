import json, sys


base_data_dir = sys.argv[1]

reasoning_type_lst = []
for mode in ['train', 'test', 'val']:
    with open(f'{base_data_dir}/{mode}_qas.json', 'r') as f:
        qas = json.load(f)
        for qa in qas:
            for reasoning_type in qa['reasoning_type'].split('$'):
                if reasoning_type in reasoning_type_lst:
                    continue
                else:
                    reasoning_type_lst.append(reasoning_type)

with open(f'{base_data_dir}/all_reasoning_types.txt', 'w') as outf:
    for reasontype in reasoning_type_lst:
        outf.write(reasontype)
        outf.write('\n')

