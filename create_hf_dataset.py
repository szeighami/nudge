import pandas as pd
import numpy as np
import pickle
import os

data_path="/workspace/final_datasets"

for q_set in ['nq', 'triviaqa', 'hotpotqa', 'fever', 'arguana']:
    output=f"/workspace/{q_set}_ft"
    if q_set in ['nq', 'triviaqa', 'hotpotqa']:
        dataset_name = 'wiki'
    else:
        dataset_name = q_set
    dataset_all = pd.read_parquet(f'{data_path}/dataset_{dataset_name}.parquet')
    dataset_all['record_id'] = np.arange(dataset_all.shape[0])
    dataset_all=dataset_all.set_index('record_id')
    os.system(f'mkdir {output}')
    dataset_all.to_parquet(f"{output}/data.parquet")

    for i in range(1):
        query_sets = {"train":{}, "dev":{}, "test":{}}
        split_type = f"random_{i}"
        for split in ["train", "dev", "test"]:
            q_df = pd.read_csv(f"{data_path}/qs_{q_set}_{split_type}_{split}.csv").drop(['Unnamed: 0.1','Unnamed: 0'], axis=1)
            q_df['q_id'] = np.arange(len(q_df))
            q_df = q_df.set_index('q_id')
            os.system(f'mkdir {output}/{split}')
            q_df.to_parquet(f"{output}/{split}/qs.parquet")

            with open (f"{data_path}/q_ans_index_{q_set}_{split_type}_{split}.pckl", 'rb') as fp:
                q_ans_indx = pickle.load(fp)
            if q_set == "nf":
                with open (f"{data_path}/q_ans_index_rel_{q_set}_{split_type}_{split}.pckl", 'rb') as fp:
                    q_ans_indx_rels = pickle.load(fp)

            rel_df = {'q_id':[], 'record_id':[], 'rel':[]}
            for j in range(len(q_ans_indx)):
                for indx in q_ans_indx[j]:
                    rel_df['q_id'].append(j)
                    rel_df['record_id'].append(indx)
                    if q_set == "nf":
                        rel_df['rel'].append(q_ans_indx_rels[j][indx])
                    else:
                        rel_df['rel'].append(1)

            rel_df = pd.DataFrame.from_dict(rel_df)
            rel_df.to_parquet(f"{output}/{split}/qs_rel.parquet", index=False)
                
