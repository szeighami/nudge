import os
import gc
import math
import pickle

from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
from torch.nn.functional import normalize
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

def compute_topk_sims_given_shard_path(q_embs, shard_path, k, dist):
    batch_size = 8192

    topks_I =[]
    topks_D =[]
    shard_i = 0
    batch_begin_index = 0
    while os.path.isfile(f"{shard_path}{shard_i}.npy"):
        embeddings_nontrain = np.load(f"{shard_path}{shard_i}.npy")
        shard_i += 1
        for i in tqdm(range(int(math.ceil(len(embeddings_nontrain)/batch_size)))):
            if dist == "cos":
                curr_topk = torch.topk(torch.matmul(q_embs, torch.t(normalize(torch.from_numpy(embeddings_nontrain[i*batch_size:(i+1)*batch_size]).to(q_embs.device)))), k=k, dim=1)
            elif dist == "dot":
                curr_topk = torch.topk(torch.matmul(q_embs, torch.t(torch.from_numpy(embeddings_nontrain[i*batch_size:(i+1)*batch_size]).to(q_embs.device))), k=k, dim=1)
            else:
                assert False, "wrong dist metric"
                
            topks_I.append(curr_topk.indices+batch_begin_index)
            batch_begin_index+=embeddings_nontrain[i*batch_size:(i+1)*batch_size].shape[0]
            topks_D.append(curr_topk.values)
    all_topk = torch.topk(torch.cat(topks_D, dim=1), k=k, dim=1)
    all_I = torch.cat(topks_I, dim=1)
    topk_I = torch.gather(all_I, 1, all_topk.indices)
    topk_D = all_topk.values

    return topk_D, topk_I

def compute_topk_sims_given_embs(q_embs, embeddings_nontrain, k, dist):
    batch_size = 8192

    topks_I =[]
    topks_D =[]
    for i in range(int(math.ceil(len(embeddings_nontrain)/batch_size))):
        if dist == "cos":
            curr_topk = torch.topk(torch.matmul(q_embs, torch.t(normalize(torch.from_numpy(embeddings_nontrain[i*batch_size:(i+1)*batch_size]).to(q_embs.device)))), k=k, dim=1)
        elif dist == "dot":
            curr_topk = torch.topk(torch.matmul(q_embs, torch.t(torch.from_numpy(embeddings_nontrain[i*batch_size:(i+1)*batch_size]).to(q_embs.device))), k=k, dim=1)
        else:
            assert False, "wrong dist metric"
        topks_I.append(curr_topk.indices+i*batch_size)
        topks_D.append(curr_topk.values)
    all_topk = torch.topk(torch.cat(topks_D, dim=1), k=k, dim=1)
    all_I = torch.cat(topks_I, dim=1)
    topk_I = torch.gather(all_I, 1, all_topk.indices)
    topk_D = all_topk.values

    return topk_D, topk_I

def compute_topk_sims(q_embs, nontrain_embeddings_or_shard, k, dist, with_index=False):
    nontrain_embs, path_to_nontrain_emb_shards =  nontrain_embeddings_or_shard
    if nontrain_embs is None and path_to_nontrain_emb_shards is None:
        return None
    if nontrain_embs is not None:
        if nontrain_embs.shape[0] == 0:
            return None
        res = compute_topk_sims_given_embs(q_embs, nontrain_embs, min(k, nontrain_embs.shape[0]), dist)
    else:
        res = compute_topk_sims_given_shard_path(q_embs, path_to_nontrain_emb_shards, k, dist)

    if with_index:
        return res
    return res[0]

def get_data(path_to_raw_dataset, path_to_emb, path_to_all_emb, path_to_q_emb, path_to_q_df , path_to_q_ans, path_to_q_rel):
    if path_to_all_emb is not None:
        load_all_embs = True
    elif path_to_emb is not None:
        load_all_embs = False
    else:
        assert False, "need either path_to_all_emb or path_to_emb"



    max_nontest_index = -1
    query_sets = {"train":{}, "dev":{}, "test":{}}
    for split in ["train", "dev", "test"]:
        q_embs = np.load(f"{path_to_q_emb}_{split}.npy").astype(np.float32)
        q_df = pd.read_csv(f"{path_to_q_df}_{split}.csv")
        with open (f"{path_to_q_ans}_{split}.pckl", 'rb') as fp:
            q_ans_indx = pickle.load(fp)
        if path_to_q_rel is not None:
            with open (f"{path_to_q_rel}_{split}.pckl", 'rb') as fp:
                q_ans_indx_rel = pickle.load(fp)
        else:
            q_ans_indx_rel=None

        if split != "test":
            max_nontest_index = max(np.array([indx for curr_q_ans_indx in q_ans_indx for indx in curr_q_ans_indx]).max()+1, max_nontest_index)

        query_sets[split]["q_df"] = q_df
        query_sets[split]["q_embs"] = q_embs
        query_sets[split]["q_ans_indx"] = q_ans_indx
        query_sets[split]["q_ans_indx_rel"] = q_ans_indx_rel


    all_dataset = pd.read_parquet(path_to_raw_dataset)
    dataset = all_dataset.iloc[:max_nontest_index]
    if load_all_embs:
        all_embeddings = np.load(path_to_all_emb, mmap_mode='r').astype(np.float32)
        nontrain_dataset = all_dataset.loc[max_nontest_index:]
        if nontrain_dataset.shape[0] == 0:
            embeddings = all_embeddings
            nontrain_dataset = None
            nontrain_embeddings  = None
        else:
            embeddings = all_embeddings[:max_nontest_index]
            nontrain_embeddings = all_embeddings[max_nontest_index:]
            del all_embeddings
            del all_dataset
    else:
        embeddings = np.load(path_to_emb, mmap_mode='r').astype(np.float32)
        nontrain_embeddings = None
        nontrain_dataset = None

    return dataset, nontrain_dataset, embeddings, nontrain_embeddings, query_sets

def calc_metrics(metrics, model_res, true_res, true_res_rel):
    if true_res_rel is None:
        true_res_rel = {}
        for indx in true_res:
            true_res_rel[indx] = 1
    indxs = true_res_rel.keys()
    rels = [true_res_rel[indx] for indx in indxs]
    ordered_rels = np.sort(rels)[::-1]

    met_res = [0 for  i in range(len(metrics))]
    for i, (met_type, k) in enumerate(metrics):
        if met_type == "recall":
            corrects = set()
            for j in range(k):
                pred_indx = model_res[j]
                if pred_indx in true_res:
                    corrects.add(pred_indx)

            met_res[i] = len(corrects)/min(len(true_res), k)
        elif met_type == "ndcg":
            ideal_dcg = np.sum([rel/np.log2(loc+2) for loc, rel in enumerate(ordered_rels[:k])])
            rel_scores = np.zeros(k)
            for j in range(k):
                pred_indx = model_res[j]
                if pred_indx in true_res_rel:
                    rel_scores[j] =true_res_rel[pred_indx]

            dcg = np.sum([rel_scores[loc]/np.log2(loc+2) for loc in range(len(rel_scores))])
            met_res[i] = dcg/ideal_dcg
    return np.array(met_res)


def calc_metrics_batch(metrics, top_k_preds, test_anss, test_anss_rel=None):
    all_met_res = np.array([0 for i in range(len(metrics))]).astype(float)
    for i in range(len(top_k_preds)):
        if test_anss_rel is not None:
            curr_test_anss_rel = test_anss_rel[i]
        else:
            curr_test_anss_rel = None

        met_res = calc_metrics(metrics, top_k_preds[i], test_anss[i], curr_test_anss_rel)
        all_met_res += met_res

    return all_met_res/len(top_k_preds)

def load_hf_datasets(q_set):
    hf_dataset_name=f"sepz/{q_set}_ft"
    print("loading dataset")
    dataset = load_dataset(hf_dataset_name, 'data_records', split='train').to_pandas()

    query_sets = {"train":{}, "dev":{}, "test":{}}                                                                                         
    for split in ["train", "dev", "test"]:                                                                                                 
        print(f"loading qs {split}")
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

        query_sets[split]["q_df"] = qs                                                                                                   
        query_sets[split]["q_ans_indx"] = answer_index                                                                                       
        if with_rels:
            query_sets[split]["q_ans_indx_rel"] = answer_rels                                                                               
        else:
            query_sets[split]["q_ans_indx_rel"] = None                                                                               
        
    return dataset, query_sets


def embed_data_and_query_sets(dataset, query_sets, emb_model):
    model = SentenceTransformer(emb_model).half()
    print('embedding data')
    data_emb = model.encode(list(dataset['text'].values), show_progress_bar=True).astype(np.float32)
    for split in ["train", "dev", "test"]:                                                                                                 
        print(f'embedding qs {split}')
        query_sets[split]['q_embs'] = model.encode(list(query_sets[split]['q_df']['input'].values), show_progress_bar=True).astype(np.float32)
    return data_emb, query_sets
