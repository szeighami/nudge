import gc
from pathlib import Path
import math
import os
import pickle

from tqdm import tqdm
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from beir.datasets.data_loader import GenericDataLoader

def get_wiki_ids_and_qs():
    wiki_ids = []

    split="train"
    split_type = "pre"
    for split in ["train", "dev"]:
        for q_set in KILT_SETS:
            print(q_set)
            df = pd.read_json(f"{PATH_TO_KILT_QS}/{q_set}-{split}-kilt.jsonl", lines=True)
            
            no_provs = []
            true_anss = []

            for i, r in tqdm(df.iterrows(), total=df.shape[0]):
                true_docs = set()
                true_pars = set()
                for l in r["output"]:
                    if 'provenance' in l.keys():
                        for l2 in l['provenance']:
                            true_docs.add(l2['wikipedia_id'])
                            true_pars.add((l2['wikipedia_id'], l2['start_paragraph_id']))
                if len(true_docs)==0:
                    no_provs.append(i)
                else:
                    true_anss.append(list(true_pars))
                    wiki_ids.extend(list(true_docs))

            df = df.loc[~df.index.isin(no_provs)].reset_index()
            assert df.shape[0] == len(true_anss)

            with open (f"{OUTPUT_PATH}/q_ans_{q_set}_{split_type}_{split}.pckl", 'wb') as fp:
                pickle.dump(true_anss, fp)
            df.to_csv(f"{OUTPUT_PATH}/qs_{q_set}_{split_type}_{split}.csv")

    wiki_ids = np.unique(wiki_ids)
    return wiki_ids

def get_wiki_text(wiki_ids):
    from kilt.knowledge_source import KnowledgeSource
    ks = KnowledgeSource()
    dfs = []
    for wiki_id in tqdm(wiki_ids):
        eng_page = ks.get_page_by_id(wiki_id)
        text = [eng_page['text'][i].rstrip() for i in range(len(eng_page['text']))]
        df = pd.DataFrame.from_dict({"doc_id":[wiki_id for i in range(len(text))],
                   "passage_id":[i for i in range(len(text))],
                   "text":text,
                  })
        dfs.append(df)

    data_df = pd.concat(dfs, ignore_index=True)

    return data_df


def prep_kilt_docs_and_qs():
    wiki_ids = get_wiki_ids_and_qs()
    dataset = get_wiki_text(wiki_ids)
    dataset.to_parquet(f'{OUTPUT_PATH}/dataset_wiki.parquet')
    return dataset

def load_beir_docs(q_set):
    for split in ["train", "dev", "test"]:
        if not os.path.isfile(f"{PATH_TO_BEIR_DATASETS}/{q_set}/qrels/{split}.tsv"):
            continue
        corpus, _, _ = GenericDataLoader(f"{PATH_TO_BEIR_DATASETS}/{q_set}/").load(split=split)
        break

    dataset = pd.DataFrame.from_dict(corpus, orient='index')
    dataset = dataset.reset_index().rename(columns={'index':'doc_id'})
    dataset['passage_id'] = "0"
    return dataset

def load_nf_docs():
    dataset_train = pd.read_csv(f"{PATH_TO_NF_DATA_AND_QS}/train.docs", sep="\t", names=['doc_id', 'text'])
    dataset_dev = pd.read_csv(f"{PATH_TO_NF_DATA_AND_QS}/dev.docs", sep="\t", names=['doc_id', 'text'])
    dataset_test = pd.read_csv(f"{PATH_TO_NF_DATA_AND_QS}/test.docs", sep="\t", names=['doc_id', 'text'])
    dataset = pd.concat([dataset_train, dataset_dev, dataset_test], axis=0).groupby('doc_id').first().reset_index()
    dataset['passage_id'] = "0"
    return dataset

def process_beir_qs(q_set):
    dataset_all = pd.read_parquet(f'{OUTPUT_PATH}/dataset_{q_set}.parquet').set_index("doc_id")
    dataset_all["row_number"] = np.arange(len(dataset_all))

    split_type = "pre"
    for split in ["train", "dev", "test"]:
        if not os.path.isfile(f"{PATH_TO_BEIR_DATASETS}/{q_set}/qrels/{split}.tsv"):
            continue

        corpus, queries, qrels = GenericDataLoader(f"{PATH_TO_BEIR_DATASETS}/{q_set}/").load(split=split)
        df_qs = pd.DataFrame.from_dict(queries, orient='index', columns=['input'])
        if q_set == "arguana":#arguana corpus is missing some documents, so removing coresponding queries
            missing_corps = []
            for q in qrels:
                for doc_id in qrels[q]:
                    if doc_id not in corpus:
                        missing_corps.append(q)
            df_qs = df_qs.drop(index=missing_corps, axis=0)

        df_qs.to_csv(f"{OUTPUT_PATH}/qs_{q_set}_{split_type}_{split}.csv")

        answer_passage_ids =[]
        for i, r in tqdm(df_qs.iterrows(), total=df_qs.shape[0]):
            curr_doc = set()
            for doc_id in qrels[i].keys():
                curr_doc.add((doc_id, "0"))
            answer_passage_ids.append(list(curr_doc))

        with open (f"{OUTPUT_PATH}/q_ans_{q_set}_{split_type}_{split}.pckl", 'wb') as fp:
            pickle.dump(answer_passage_ids, fp)


def process_nf_qs():
    q_set = "nf"

    dataset_all = pd.read_parquet(f'{OUTPUT_PATH}/dataset_{q_set}.parquet').set_index("doc_id")
    dataset_all["row_number"] = np.arange(len(dataset_all))

    split_type = "pre"
    accessed_docs_indx = []

    for split in ["train", "dev", "test"]:
        print("proc ", q_set, split)
        qrels = pd.read_csv(f"{PATH_TO_NF_DATA_AND_QS}/{split}.2-1-0.qrel", sep="\t", names=['q_id', '0', 'doc_id', 'rel']) 
        nf_qs1 = pd.read_csv(f"{PATH_TO_NF_DATA_AND_QS}/{split}.nontopic-titles.queries", sep="\t", names=['q_id', 'input'])
        nf_qs2 = pd.read_csv(f"{PATH_TO_NF_DATA_AND_QS}/{split}.vid-titles.queries", sep="\t", names=['q_id', 'input'])
        nf_qs3 = pd.read_csv(f"{PATH_TO_NF_DATA_AND_QS}/{split}.vid-desc.queries", sep="\t", names=['q_id', 'input'])

        nf_qs = pd.concat([nf_qs1, nf_qs2, nf_qs3], ignore_index=True)
        df_qs = nf_qs.groupby('input').first().reset_index()

        df_qs.to_csv(f"{OUTPUT_PATH}/qs_{q_set}_{split_type}_{split}.csv")

        answer_passage_ids =[]
        answer_rels =[]
        for i, r in tqdm(df_qs.iterrows(), total=df_qs.shape[0]):
            curr_doc = set()
            curr_rel = {}
            assert qrels[qrels['q_id']==r['q_id']].shape[0] > 0
            for j, rel_row in qrels[qrels['q_id']==r['q_id']].iterrows():
                doc_id = rel_row['doc_id']
                curr_doc.add((doc_id, "0"))
                curr_rel[(doc_id, "0")] = rel_row['rel']

            answer_passage_ids.append(list(curr_doc))
            answer_rels.append(curr_rel)

        with open (f"{OUTPUT_PATH}/q_ans_{q_set}_{split_type}_{split}.pckl", 'wb') as fp:
            pickle.dump(answer_passage_ids, fp)
        with open (f"{OUTPUT_PATH}/q_ans_rel_{q_set}_{split_type}_{split}.pckl", 'wb') as fp:
            pickle.dump(answer_rels, fp)


def prep_nonkilt_docs_and_qs(q_set):
    if q_set in BEIR_SETS:
        dataset_name = q_set
    elif q_set == "nf":
        dataset_name = q_set
    else:
        assert False, "WRONG Q_SET NAME"

    if q_set in BEIR_SETS:
        dataset = load_beir_docs(q_set)
    elif q_set == "nf":
        dataset = load_nf_docs()
    else:
        assert False, "WRONG Q_SET NAME"
    dataset.to_parquet(f'{OUTPUT_PATH}/dataset_{dataset_name}.parquet')

    if q_set in BEIR_SETS:
        process_beir_qs(q_set)
    elif q_set == "nf":
        process_nf_qs()


def reorder_and_seprate_data_basedon_training_set(dataset_name, q_sets, split_name):
    dataset = pd.read_parquet(f'{OUTPUT_PATH}/dataset_{dataset_name}.parquet').set_index(["doc_id", "passage_id"])
    dataset['row_number'] =np.arange(len(dataset)) 

    q_ans = []
    for q_set in q_sets:
        splits = ["train", "dev"]
        for split in splits:
            with open (f"{OUTPUT_PATH}/q_ans_{q_set}_{split_name}_{split}.pckl", 'rb') as fp:
                q_ans.extend(pickle.load(fp))

    row_numbers = []
    for i in tqdm(range(len(q_ans))):
        for doc_id, passage_id in q_ans[i]:
            indx = dataset.loc[(doc_id, passage_id), "row_number"]
            row_numbers.append(indx)
    training_row_numbrs = np.unique(row_numbers)
    dataset = dataset.reset_index(drop=False).set_index('row_number')
    dataset['is_train'] = dataset.index.isin(training_row_numbrs)
    dataset.sort_values('is_train', ascending=False, inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    dataset.drop('is_train', axis=1, inplace=True)
    dataset.to_parquet(f'{OUTPUT_PATH}/dataset_{dataset_name}.parquet')
    return len(training_row_numbrs)

def get_max_train_index(q_sets, split_name):
    max_nontest_index = -1
    q_ans = []
    for q_set in q_sets:
        splits = ["train", "dev"]
        for split in splits:
            with open (f"{OUTPUT_PATH}/q_ans_index_{q_set}_{split_name}_{split}.pckl", 'rb') as fp:
                q_ans_indx = pickle.load(fp)
                max_nontest_index = max(np.array([indx for curr_q_ans_indx in q_ans_indx for indx in curr_q_ans_indx]).max()+1, max_nontest_index)
    return max_nontest_index

def create_q_ans_index(q_set, split_name):
    if q_set in KILT_SETS:
        dataset_name = "wiki"
    else:
        dataset_name = q_set
    dataset = pd.read_parquet(f'{OUTPUT_PATH}/dataset_{dataset_name}.parquet').set_index(["doc_id", "passage_id"])
    dataset['row_number'] =np.arange(len(dataset)) 

    with_rels=True
    splits = ["train", "dev", "test"]
    for split in splits:
        with open (f"{OUTPUT_PATH}/q_ans_{q_set}_{split_name}_{split}.pckl", 'rb') as fp:
            q_ans =pickle.load(fp)
        if os.path.isfile(f"{OUTPUT_PATH}/q_ans_rel_{q_set}_{split_name}_{split}.pckl"):
            with open (f"{OUTPUT_PATH}/q_ans_rel_{q_set}_{split_name}_{split}.pckl", 'rb') as fp:
                q_ans_rels =pickle.load(fp)
        else:
            with_rels=False

        q_ans_indxs = []
        if with_rels:
            q_ans_indx_rels = []
        for i in tqdm(range(len(q_ans))):
            curr_indx = set()
            curr_indx_rel = {}
            for doc_id, passage_id in q_ans[i]:
                indx = dataset.loc[(doc_id, passage_id), "row_number"]
                curr_indx.add(indx)
                if with_rels:
                    rel = q_ans_rels[i][(doc_id, passage_id)]
                    curr_indx_rel[indx] = rel
            q_ans_indxs.append(list(curr_indx))
            if with_rels:
                q_ans_indx_rels.append(curr_indx_rel)

        with open (f"{OUTPUT_PATH}/q_ans_index_{q_set}_{split_name}_{split}.pckl", 'wb') as fp:
            pickle.dump(q_ans_indxs, fp)

        if with_rels:
            with open (f"{OUTPUT_PATH}/q_ans_index_rel_{q_set}_{split_name}_{split}.pckl", 'wb') as fp:
                pickle.dump(q_ans_indx_rels, fp)


def embed_docs(dataset_name, emb_model, modeL_prefix, max_train_index, mem_cap_in_gb):
    if os.path.isfile(f"{OUTPUT_PATH}/all_data_emb_{emb_model}_{dataset_name}.npy"):
        return

    print("embedding docs")
    dataset = pd.read_parquet(f'{OUTPUT_PATH}/dataset_{dataset_name}.parquet')
    save_batchsize = 2048
    lens = dataset['text'].str.len().values
    lens_indx = np.argsort(lens)[::-1]

    embs = None
    if emb_model in ["text-embedding-3-large", "text-embedding-3-small"]:
        from openai_utils import get_embeddings
        import tiktoken
        embedding_encoding = "cl100k_base"
        encoding = tiktoken.get_encoding(embedding_encoding)
        for i in tqdm(range(int(math.ceil(dataset.shape[0]/save_batchsize)))):
            if os.path.isfile(f"{OUTPUT_PATH}/data_emb_intermediate_{dataset_name}_{emb_model}_{i}.npy"):
                continue

            curr_txt = list(dataset.iloc[lens_indx[i*save_batchsize:(i+1)*save_batchsize]]['text'].values)
            if len(curr_txt[-1]) == 0:
                curr_txt = ['.' if x == "" else x  for x in curr_txt]#empty srtings cause error in api call
            truncated = [encoding.decode(encoding.encode(x)[:8191]) for x in curr_txt]

            embs = np.array(get_embeddings(truncated, model=emb_model))
            np.save(f"{OUTPUT_PATH}/data_emb_intermediate_{dataset_name}_{emb_model}_{i}.npy", embs)


    else:
        embedding_function = SentenceTransformer(modeL_prefix+"/"+emb_model, trust_remote_code=True).half()
        for i in tqdm(range(int(math.ceil(dataset.shape[0]/save_batchsize)))):
            if os.path.isfile(f"{OUTPUT_PATH}/data_emb_intermediate_{dataset_name}_{emb_model}_{i}.npy"):
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
            np.save(f"{OUTPUT_PATH}/data_emb_intermediate_{dataset_name}_{emb_model}_{i}.npy", embs)

    if embs is None:
        embs = np.load(f"{OUTPUT_PATH}/data_emb_intermediate_{dataset_name}_{emb_model}_0.npy")

    max_size = mem_cap_in_gb*2**28/embs.shape[1]
    if dataset.shape[0]>max_size:
        assert max_train_index < max_size, "train embeddings need to fit in memory and in one file"

        def get_embedding_in_range(begin_range, end_range):
            curr_embeddings = np.zeros((end_range-begin_range, embs.shape[1]), dtype=np.float32)
            for i in tqdm(range(int(math.ceil(dataset.shape[0]/save_batchsize)))):
                curr_indx = lens_indx[i*save_batchsize:(i+1)*save_batchsize]
                mask = np.logical_and(curr_indx>=begin_range, curr_indx<end_range)
                curr_indx = curr_indx[mask]
                curr_embeddings[curr_indx.reshape(-1)-begin_range] = np.load(f"{OUTPUT_PATH}/data_emb_intermediate_{dataset_name}_{emb_model}_{i}.npy")[mask]
            return curr_embeddings

        train_embeddings = get_embedding_in_range(0, max_train_index)
        np.save(f"{OUTPUT_PATH}/train_data_emb_{dataset_name}_{emb_model}.npy", train_embeddings)
        del train_embeddings
        gc.collect()

        non_train_size = len(dataset)-max_train_index
        no_shards = int(math.ceil(non_train_size/max_size))
        file_size = int(math.ceil(non_train_size/no_shards))
        
        for j in range(no_shards):
            begin_range = j*file_size+max_train_index
            end_range = min((j+1)*file_size+max_train_index,len(dataset))

            curr_embeddings = get_embedding_in_range(begin_range, end_range)
            np.save(f"{OUTPUT_PATH}/nontrain_data_emb_shards_{dataset_name}_{emb_model}_{j}.npy", curr_embeddings)
    else:
        embeddings = np.zeros((dataset.shape[0], embs.shape[1]), dtype=np.float32)
        for i in tqdm(range(int(math.ceil(dataset.shape[0]/save_batchsize)))):
            curr_indx = lens_indx[i*save_batchsize:(i+1)*save_batchsize]
            embeddings[curr_indx.reshape(-1)] = np.load(f"{OUTPUT_PATH}/data_emb_intermediate_{dataset_name}_{emb_model}_{i}.npy")
        np.save(f"{OUTPUT_PATH}/all_data_emb_{emb_model}_{dataset_name}.npy", embeddings)

    os.system(f"rm {OUTPUT_PATH}/data_emb_intermediate_{dataset_name}_{emb_model}_*.npy")


def embed_qs(q_set, emb_model, modeL_prefix,with_prompt, batch_size, split_name):
    print("embed queries")
    if emb_model not in OPENAI_MODELS:
        embedding_function = SentenceTransformer(modeL_prefix+"/"+emb_model, trust_remote_code=True).half()
    else:
        from openai_utils import get_embeddings

    for split in ["train", "dev", "test"]:
        df_qs = pd.read_csv(f"{OUTPUT_PATH}/qs_{q_set}_{split_name}_{split}.csv")

        if not os.path.isfile(f"{OUTPUT_PATH}/qs_emb_{emb_model}_{q_set}_{split_name}_{split}.npy"):
            print("embedding ", q_set, split)
            if emb_model in OPENAI_MODELS:
                openai_batchsize = 2048
                embs = []
                for i in tqdm(range(int(math.ceil(df_qs.shape[0]/openai_batchsize)))):
                    curr_df = df_qs.iloc[i*openai_batchsize:(i+1)*openai_batchsize]
                    embs.append(np.array(get_embeddings(list(curr_df['input'].values), model=emb_model)))
                split_embs = np.concatenate(embs, axis=0)
            else:
                if with_prompt:
                    split_embs = np.array(embedding_function.encode(list(df_qs['input'].values), batch_size=batch_size, show_progress_bar=True, prompt_name="query"))
                else:
                    split_embs = np.array(embedding_function.encode(list(df_qs['input'].values), batch_size=batch_size, show_progress_bar=True))
            np.save(f"{OUTPUT_PATH}/qs_emb_{emb_model}_{q_set}_{split_name}_{split}.npy", split_embs)


def create_q_splits(q_set, seed):
    np.random.seed(seed)

    split_type = "pre"
    all_qs = []
    all_answer_passage_ids = []
    all_answer_rels = []
    with_rels = True
    for split in ["train", "dev", "test"]:
        if not os.path.isfile(f"{OUTPUT_PATH}/qs_{q_set}_{split_type}_{split}.csv"):
            continue
        df_qs = pd.read_csv(f"{OUTPUT_PATH}/qs_{q_set}_{split_type}_{split}.csv")
        with open (f"{OUTPUT_PATH}/q_ans_{q_set}_{split_type}_{split}.pckl", 'rb') as fp:
            answer_passage_ids = pickle.load(fp)
        if os.path.isfile(f"{OUTPUT_PATH}/q_ans_rel_{q_set}_{split_type}_{split}.pckl"):
            with open (f"{OUTPUT_PATH}/q_ans_rel_{q_set}_{split_type}_{split}.pckl", 'rb') as fp:
                answer_rels = pickle.load(fp)
        else:
            with_rels = False

        all_qs.append(df_qs)
        all_answer_passage_ids.extend(answer_passage_ids)
        if with_rels:
            all_answer_rels.extend(answer_rels)

    queries = pd.concat(all_qs, axis=0, ignore_index=True)
    answer_passage_ids = all_answer_passage_ids
    if with_rels:
        answer_rels = all_answer_rels

    val_size = min(10000, int(0.1*len(queries)))
    test_size = min(10000, int(0.2*len(queries)))
    train_size = len(queries)-val_size-test_size
    rand_indx = np.random.permutation(len(queries))
    train_indx = rand_indx[:train_size]
    val_indx = rand_indx[train_size:train_size+val_size]
    test_indx = rand_indx[train_size+val_size:]
    splits_indx = {"train":train_indx, "dev":val_indx, "test":test_indx}

    splits = ["train", "dev", "test"]
    split_type = "random"
    for split in splits:
        split_qs = queries.iloc[splits_indx[split]]
        split_qs.to_csv(f"{OUTPUT_PATH}/qs_{q_set}_{split_type}_{seed}_{split}.csv")

        split_answer_passage_ids = []
        if with_rels:
            split_answer_rels = []
        for indx in splits_indx[split]:
            split_answer_passage_ids.append(answer_passage_ids[indx])
            if with_rels:
                split_answer_rels.append(answer_rels[indx])

        with open (f"{OUTPUT_PATH}/q_ans_{q_set}_{split_type}_{seed}_{split}.pckl", 'wb') as fp:
            pickle.dump(split_answer_passage_ids, fp)
        if with_rels:
            with open (f"{OUTPUT_PATH}/q_ans_rel_{q_set}_{split_type}_{seed}_{split}.pckl", 'wb') as fp:
                pickle.dump(split_answer_rels, fp)

def create_hf_dataset(q_set):
    if q_set in KILT_SETS:
        dataset_name = 'wiki'
    else:
        dataset_name = q_set
    dataset_all = pd.read_parquet(f'{OUTPUT_PATH}/dataset_{dataset_name}.parquet')
    dataset_all['record_id'] = np.arange(dataset_all.shape[0])
    dataset_all=dataset_all.set_index('record_id')
    output = HF_OUTPUT_PATH+f"/{q_set}"
    Path(output).mkdir(parents=True, exist_ok=True)
    dataset_all.to_parquet(f"{output}/data.parquet")

    query_sets = {"train":{}, "dev":{}, "test":{}}
    split_type = f"random_0"
    for split in ["train", "dev", "test"]:
        q_df = pd.read_csv(f"{OUTPUT_PATH}/qs_{q_set}_{split_type}_{split}.csv").drop(['Unnamed: 0.1','Unnamed: 0'], axis=1)
        q_df['q_id'] = np.arange(len(q_df))
        q_df = q_df.set_index('q_id')
        os.system(f'mkdir {output}/{split}')
        q_df.to_parquet(f"{output}/{split}/qs.parquet")

        with open (f"{OUTPUT_PATH}/q_ans_index_{q_set}_{split_type}_{split}.pckl", 'rb') as fp:
            q_ans_indx = pickle.load(fp)
        if q_set == "nf":
            with open (f"{OUTPUT_PATH}/q_ans_index_rel_{q_set}_{split_type}_{split}.pckl", 'rb') as fp:
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

KILT_SETS = ["nq", "triviaqa", "hotpotqa"]
BEIR_SETS = ["scifact", "arguana", 'fever']
OTHER = ['nf']

OUTPUT_PATH=os.getcwd()+"/interim_data"
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
HF_OUTPUT_PATH=os.getcwd()+"/hf_datasets"
Path(HF_OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
PATH_TO_KILT_QS= "/workspace"
PATH_TO_BEIR_DATASETS="/datasets"
PATH_TO_NF_DATA_AND_QS="/workspace/nfcorpus"

non_kiltq_sets =OTHER+BEIR_SETS
random_seed = 0

print("Preparing data")
for q_set in non_kiltq_sets:
    assert q_set not in KILT_SETS
    prep_nonkilt_docs_and_qs(q_set)
    create_q_splits(q_set, random_seed)
    print("reorder_and_seprate_data_basedon_training_set")
    reorder_and_seprate_data_basedon_training_set(q_set, [q_set], f"random_{random_seed}")
    print("create_q_ans_index")
    create_q_ans_index(q_set, f"random_{random_seed}")
    create_hf_dataset(q_set)


print("Processing KILT")
prep_kilt_docs_and_qs()
for q_set in KILT_SETS:
    print(f"create_q_splits {q_set}")
    create_q_splits(q_set, random_seed)

print("reorder_and_seprate_data_basedon_training_set")
max_training_index=reorder_and_seprate_data_basedon_training_set("wiki", KILT_SETS, f"random_{random_seed}")
for q_set in KILT_SETS:
    print(f"create_q_ans_index {q_set}")
    create_q_ans_index(q_set, f"random_{random_seed}")
    create_hf_dataset(q_set)
