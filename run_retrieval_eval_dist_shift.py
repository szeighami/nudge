import math
import os
import gc
import argparse
import time
import json

import pandas as pd
import numpy as np
import torch

from ptft import PTFT
from knnretriever import kNNRetriever
from nudge import NUDGEM, NUDGEN
from adapter import AdapterFineTuner
from utils import get_data, calc_metrics, calc_metrics_batch



def run_test(q_set, finetuner, test_name, out_path, finetuner_config_path, data_config_path, device):
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device(device)

    with open(finetuner_config_path) as f:
        finetuner_config = json.load(f)
    with open(data_config_path) as f:
        data_config = json.load(f)

    metrics = [["recall", 1], ["recall", 5], ["recall", 10], ["recall", 20], ["recall", 40], ["ndcg", 1], ["ndcg", 5], ["ndcg", 10], ["ndcg", 20], ["ndcg", 40]]
    max_k = np.max([met[1] for met in metrics]).astype(int)


    path_to_sharded_emb_nontrain = None
    if "path_to_sharded_emb_nontrain" in data_config:
        path_to_sharded_emb_nontrain = data_config["path_to_sharded_emb_nontrain"]
        data_config.pop("path_to_sharded_emb_nontrain")
    dataset, nontrain_dataset, embeddings, nontrain_embeddings, query_sets = get_data(**data_config)

    n_clusters = 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(query_sets['train']['q_embs'])
    clusters = {}
    clusters['train']  = kmeans.labels_.reshape(-1)
    clusters['dev'] = kmeans.predict(query_sets['dev']['q_embs']).reshape(-1)
    clusters['test'] = kmeans.predict(query_sets['test']['q_embs']).reshape(-1)
    clustered_query_sets = {}
    for i in range(n_clusters):
        clustered_query_sets[i] = {}

    for split in ["train", "dev", "test"]:
        for i in range(n_clusters):
            clustered_q_df = deepcopy(query_sets[split]['q_df'].iloc[np.nonzero(clusters[split]==i)])
            clustered_q_embs = deepcopy(query_sets[split]['q_embs'][np.nonzero(clusters[split]==i)])
            clustered_query_sets[i][split] = {}
            clustered_query_sets[i][split]['q_embs'] = clustered_q_embs
            clustered_query_sets[i][split]['q_df'] = clustered_q_df
            clustered_query_sets[i][split]['q_ans_indx'] = []

        for i in range(len(query_sets[split]['q_ans_indx'])):
            cluster_number = clusters[split][i]
            clustered_query_sets[cluster_number][split]['q_ans_indx'].append(query_sets[split]['q_ans_indx'][i])

    query_sets['train'] = clustered_query_sets[0]['train']
    query_sets['dev'] = clustered_query_sets[0]['dev']

    start = time.time()
    for test_cluster in [0, 1]:
        query_sets['test'] = clustered_query_sets[test_cluster]['test']
        if finetuner == "NUDGEN":
            ft = NUDGEN(device)
            nontrain_embeddings_or_shard = nontrain_embeddings, path_to_sharded_emb_nontrain
            new_embs = ft.finetune_embeddings(embeddings, query_sets["train"], query_sets["dev"], nontrain_embeddings_or_shard, **finetuner_config)

            del ft
            gc.collect()
            torch.cuda.empty_cache()

            ret = kNNRetriever(new_embs, nontrain_embeddings, dist_metric='cos', path_to_embeddings_nontrain_shards=path_to_sharded_emb_nontrain, device=device)

        elif finetuner == "NUDGEM":
            ft = NUDGEM(device)
            nontrain_embeddings_or_shard = nontrain_embeddings, path_to_sharded_emb_nontrain
            new_embs = ft.finetune_embeddings(embeddings, query_sets["train"], query_sets["dev"], nontrain_embeddings_or_shard, **finetuner_config)

            del ft
            gc.collect()
            torch.cuda.empty_cache()

            ret = kNNRetriever(new_embs, nontrain_embeddings, dist_metric='dot', path_to_embeddings_nontrain_shards=path_to_sharded_emb_nontrain, device=device)

        elif finetuner == "AdapterFineTuner":
            ft = AdapterFineTuner(device)

            nontrain_embeddings_or_shard = nontrain_embeddings, path_to_sharded_emb_nontrain
            adaptor = ft.finetune(embeddings, query_sets["train"], query_sets["dev"], nontrain_embeddings_or_shard, **finetuner_config)

            del ft
            gc.collect()
            torch.cuda.empty_cache()

            ret = kNNRetriever(embeddings, nontrain_embeddings, q_adaptor=adaptor, dist_metric='cos', path_to_embeddings_nontrain_shards=path_to_sharded_emb_nontrain, device=device)

        elif finetuner == "PTFT":
            val_with_topk_only = 0
            if "val_with_topk_only" in finetuner_config:
                val_with_topk_only = finetuner_config["val_with_topk_only"]
                finetuner_config.pop("val_with_topk_only")

            if val_with_topk_only > 0:
                ret = kNNRetriever(embeddings, nontrain_embeddings, dist_metric='cos', path_to_embeddings_nontrain_shards=path_to_sharded_emb_nontrain, device=device)
                top_k_indxs_val = ret.retrieve_topk_from_emb_batch(val_with_topk_only , query_sets['dev']['q_embs'])
                indx_nontrain = np.unique(top_k_indxs_val[top_k_indxs_val>=len(embeddings)])
                del ret
            else:
                indx_nontrain = None

            if embeddings is not None:
                del embeddings
            if nontrain_embeddings is not None:
                del nontrain_embeddings
            gc.collect()
            torch.cuda.empty_cache()

            ft = PTFT(device)
            embedding_model = ft.finetune_model(query_sets["train"], query_sets["dev"], dataset, nontrain_dataset, indx_nontrain, **finetuner_config)
            embedding_model = embedding_model.half()

            batch_size = finetuner_config['val_batch_size']
            embeddings = np.array(embedding_model.encode(list(dataset['text'].values), batch_size=batch_size, show_progress_bar=True)).astype(np.float32)
            if nontrain_dataset is not None:
                nontrain_embeddings = np.array(embedding_model.encode(list(nontrain_dataset['text'].values), batch_size=batch_size, show_progress_bar=True)).astype(np.float32)
            test_q_embeddings = np.array(embedding_model.encode(list(query_sets['test']['q_df']['input'].values), batch_size=batch_size, show_progress_bar=True)).astype(np.float32)

            del ft
            del embedding_model
            gc.collect()
            torch.cuda.empty_cache()

            query_sets['test']['q_embs'] = test_q_embeddings
            ret = kNNRetriever(embeddings, nontrain_embeddings, device=device)

        elif finetuner == "no_ft":
            ret = kNNRetriever(embeddings, nontrain_embeddings, path_to_embeddings_nontrain_shards=path_to_sharded_emb_nontrain, device=device)
        else:
            assert False, "FT NOT SUPPORTED"

        end = time.time()

        print(f"answering queries with retriever {finetuner}")
        all_met_res = np.array([0 for i in range(len(metrics))]).astype(float)

        test_set = query_sets['test']
        test_query_df = test_set["q_df"]
        test_query_embs = test_set["q_embs"]
        test_anss = test_set["q_ans_indx"]
        test_anss_rel = test_set["q_ans_indx_rel"]

        print("batch_answer")
        top_k_indxs = ret.retrieve_topk_from_emb_batch(max_k, test_query_embs)

        print("calc metrics")
        for i in range(len(top_k_indxs)):
            if test_anss_rel is not None:
                curr_test_anss_rel = test_anss_rel[i]
            else:
                curr_test_anss_rel = None

            met_res = calc_metrics(metrics, top_k_indxs[i], test_anss[i], curr_test_anss_rel)
            all_met_res += met_res

        print("TESTING", q_set, finetuner, all_met_res/len(test_query_embs))

        results = {"test_name":[test_name], "finetuner":[finetuner], "q_set":[q_set+str(test_cluster)], 'train_time':[end-start]}
        for i, (met_type, k) in enumerate(metrics):
            results[f'{met_type}_{k}'] = [all_met_res[i]/len(test_query_embs)]
        if "gamma" in finetuner_config:
            results["gamma"] = finetuner_config["gamma"]
        res_df = pd.DataFrame(results)
        res_df['config_file'] = finetuner_config_path
        res_df['data_config_file'] = data_config_path

        output = f"{out_path}/res.csv"
        res_df.to_csv(output, mode='a', header=not os.path.exists(output))

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('test_set', type=str,
                                help='name of test dataset')
    parser.add_argument('retriever', type=str,
                                help='retriever and finetuning method to use')
    parser.add_argument('test_name', type=str,
                                help='test name to store results')
    parser.add_argument('out_path',type=str, help='directory to store results')
    parser.add_argument('finetuner_config_path',type=str, help='path to json file with finetuner config')
    parser.add_argument('data_config_path',type=str, help='path to json file with data config')
    parser.add_argument('--device',type=str, help='device to use', default=None)

    args = parser.parse_args()
    if args.device is None:
        device ="cuda" if torch.cuda.is_available() else "cpu"
    else:
        device =args.device


    run_test(args.test_set, args.retriever, args.test_name, args.out_path, args.finetuner_config_path, args.data_config_path, device)

