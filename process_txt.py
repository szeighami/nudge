import gc
import math
import os
import pickle
import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from datasets import load_dataset

def get_max_train_index(q_set, output_path):
    max_nontest_index = -1
    q_ans = []
    splits = ["train", "dev"]
    for split in splits:
        with open (f"{output_path}/q_ans_index_{q_set}_{split}.pckl", 'rb') as fp:
            q_ans_indx = pickle.load(fp)
            max_nontest_index = max(np.array([indx for curr_q_ans_indx in q_ans_indx for indx in curr_q_ans_indx]).max()+1, max_nontest_index)
    return max_nontest_index

def embed_docs(dataset_name, emb_model, modeL_prefix, max_train_index, mem_cap_in_gb, output_path):
    if os.path.isfile(f"{output_path}/all_data_emb_{emb_model}_{dataset_name}.npy"):
        return

    print("embedding docs")
    dataset = pd.read_parquet(f'{output_path}/dataset_{dataset_name}.parquet')
    save_batchsize = 2048
    lens = dataset['text'].str.len().values
    lens_indx = np.argsort(lens)[::-1]

    embs = None
    if emb_model in OPENAI_MODELS:
        from openai_utils import get_embeddings
        import tiktoken
        embedding_encoding = "cl100k_base"
        encoding = tiktoken.get_encoding(embedding_encoding)
        for i in tqdm(range(int(math.ceil(dataset.shape[0]/save_batchsize)))):
            if os.path.isfile(f"{output_path}/data_emb_intermediate_{dataset_name}_{emb_model}_{i}.npy"):
                continue

            curr_txt = list(dataset.iloc[lens_indx[i*save_batchsize:(i+1)*save_batchsize]]['text'].values)
            if len(curr_txt[-1]) == 0:
                curr_txt = ['.' if x == "" else x  for x in curr_txt]#empty srtings cause error in api call
            truncated = [encoding.decode(encoding.encode(x)[:8191]) for x in curr_txt]

            embs = np.array(get_embeddings(truncated, model=emb_model))
            np.save(f"{output_path}/data_emb_intermediate_{dataset_name}_{emb_model}_{i}.npy", embs)


    else:
        embedding_function = SentenceTransformer(modeL_prefix+"/"+emb_model, trust_remote_code=True).half()
        for i in tqdm(range(int(math.ceil(dataset.shape[0]/save_batchsize)))):
            if os.path.isfile(f"{output_path}/data_emb_intermediate_{dataset_name}_{emb_model}_{i}.npy"):
                continue
            if i < 2:
                curr_batch_size = 1
            elif i < 10:
                curr_batch_size = 8
            elif i < 100:
                curr_batch_size = 32
            else:
                curr_batch_size = 64
            embs = np.array(embedding_function.encode(list(dataset.iloc[lens_indx[i*save_batchsize:(i+1)*save_batchsize]]['text'].values), batch_size=curr_batch_size, show_progress_bar=True))
            np.save(f"{output_path}/data_emb_intermediate_{dataset_name}_{emb_model}_{i}.npy", embs)

    if embs is None:
        embs = np.load(f"{output_path}/data_emb_intermediate_{dataset_name}_{emb_model}_0.npy")

    max_size = mem_cap_in_gb*2**28/embs.shape[1]
    if dataset.shape[0]>max_size:
        assert max_train_index < max_size, "train embeddings need to fit in memory and in one file"

        def get_embedding_in_range(begin_range, end_range):
            curr_embeddings = np.zeros((end_range-begin_range, embs.shape[1]), dtype=np.float32)
            for i in tqdm(range(int(math.ceil(dataset.shape[0]/save_batchsize)))):
                curr_indx = lens_indx[i*save_batchsize:(i+1)*save_batchsize]
                mask = np.logical_and(curr_indx>=begin_range, curr_indx<end_range)
                curr_indx = curr_indx[mask]
                curr_embeddings[curr_indx.reshape(-1)-begin_range] = np.load(f"{output_path}/data_emb_intermediate_{dataset_name}_{emb_model}_{i}.npy")[mask]
            return curr_embeddings

        train_embeddings = get_embedding_in_range(0, max_train_index)
        np.save(f"{output_path}/train_data_emb_{dataset_name}_{emb_model}.npy", train_embeddings)
        del train_embeddings
        gc.collect()

        non_train_size = len(dataset)-max_train_index
        no_shards = int(math.ceil(non_train_size/max_size))
        file_size = int(math.ceil(non_train_size/no_shards))
        
        for j in range(no_shards):
            begin_range = j*file_size+max_train_index
            end_range = min((j+1)*file_size+max_train_index,len(dataset))

            curr_embeddings = get_embedding_in_range(begin_range, end_range)
            np.save(f"{output_path}/nontrain_data_emb_shards_{dataset_name}_{emb_model}_{j}.npy", curr_embeddings)
    else:
        embeddings = np.zeros((dataset.shape[0], embs.shape[1]), dtype=np.float32)
        for i in tqdm(range(int(math.ceil(dataset.shape[0]/save_batchsize)))):
            curr_indx = lens_indx[i*save_batchsize:(i+1)*save_batchsize]
            embeddings[curr_indx.reshape(-1)] = np.load(f"{output_path}/data_emb_intermediate_{dataset_name}_{emb_model}_{i}.npy")
        np.save(f"{output_path}/all_data_emb_{emb_model}_{dataset_name}.npy", embeddings)

    os.system(f"rm {output_path}/data_emb_intermediate_{dataset_name}_{emb_model}_*.npy")


def embed_qs(q_set, emb_model, modeL_prefix, batch_size, output_path):
    print("embed queries")
    if emb_model not in OPENAI_MODELS:
        embedding_function = SentenceTransformer(modeL_prefix+"/"+emb_model, trust_remote_code=True).half()
    else:
        from openai_utils import get_embeddings

    for split in ["train", "dev", "test"]:
        df_qs = pd.read_csv(f"{output_path}/qs_{q_set}_{split}.csv")

        if not os.path.isfile(f"{output_path}/qs_emb_{emb_model}_{q_set}_{split}.npy"):
            print("embedding ", q_set, split)
            if emb_model in OPENAI_MODELS:
                openai_batchsize = 2048
                embs = []
                for i in tqdm(range(int(math.ceil(df_qs.shape[0]/openai_batchsize)))):
                    curr_df = df_qs.iloc[i*openai_batchsize:(i+1)*openai_batchsize]
                    embs.append(np.array(get_embeddings(list(curr_df['input'].values), model=emb_model)))
                split_embs = np.concatenate(embs, axis=0)
            else:
                split_embs = np.array(embedding_function.encode(list(df_qs['input'].values), batch_size=batch_size, show_progress_bar=True))
            np.save(f"{output_path}/qs_emb_{emb_model}_{q_set}_{split}.npy", split_embs)



def format_and_save_files(q_set, output_path):
    hf_dataset_name = f"sepz/{q_set}_ft"
    corpus = load_dataset(hf_dataset_name, 'data_records', split='train').to_pandas()
    corpus.to_parquet(f"{output_path}/dataset_{q_set}.parquet")

    for split in ['train', 'dev', 'test']:
        print(split)
        qs = load_dataset(hf_dataset_name, 'qs', split=split).to_pandas()

        rels = load_dataset(hf_dataset_name, 'qs_rel', split=split).to_pandas()
        with_rels = (rels['rel']>1).sum() > 0
        q_ids = []
        answer_index = []
        if with_rels:
            answer_rels = []
        for g, rels in rels.groupby("q_id"):
            q_ids.append(g)
            curr_indxs = []
            if with_rels:
                curr_rels = {}
            for i,r in rels.iterrows(): 
                curr_indxs.append(r['record_id'])
                if with_rels:
                    curr_rels[r['record_id']] = r['rel']
            answer_index.append(curr_indxs)
            if with_rels:
                answer_rels.append(curr_rels)
        qs = qs.set_index('q_id').loc[q_ids, :].reset_index()
        qs.to_csv(f"{output_path}/qs_{q_set}_{split}.csv")

        with open (f"{output_path}/q_ans_index_{q_set}_{split}.pckl", 'wb') as fp:
            pickle.dump(answer_index, fp)

        if with_rels:
            with open (f"{output_path}/q_ans_index_rel_{q_set}_{split}.pckl", 'wb') as fp:
                pickle.dump(answer_rels, fp)


def process_all(output_path):
    q_emb_batch_size = 8
    #Maximum size of a single file that can be loaded in memory. Embeddings are split into smaller sharded files if they are larger
    mem_cap_in_gb=80

    for q_set in ['nfcorpus', "scifact", "arguana", 'fever', "nq", "triviaqa", "hotpotqa"]:
        print(q_set)
        print("format and save files")
        format_and_save_files(q_set, output_path)

        max_training_index=get_max_train_index(q_set, output_path)
        #add ("openai", "text-embedding-3-large") to also run TE3-L. Make sure OPENAI_API_KEY environment variable is set
        for emb_prefix, emb_model  in [("BAAI", "bge-small-en-v1.5"), ("Alibaba-NLP", "gte-large-en-v1.5")]:
            print(f"emed docs with {emb_model}")
            embed_docs(q_set, emb_model, emb_prefix, max_training_index, mem_cap_in_gb, output_path)
            print(f"emed qs with {emb_model}")
            embed_qs(q_set, emb_model, emb_prefix, q_emb_batch_size, output_path)


OPENAI_MODELS = ["text-embedding-3-large", "text-embedding-3-small"]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str,
                                help='Directory for processed data files')
    args = parser.parse_args()
    output_path=args.data_path
    process_all(output_path)

