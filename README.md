# NDUGE
This repo contains the code for NUDGE: Lightweight Non-Parametric Embedding Fine-Tuning. The method solves a constrained optimization problem to move data embeddings towards the embedding of training queries for which they are the ground-truth answer. NUDGE-M and NUDGE-N are two variants in this repository, each solving the optimization problem with different constraints. 

<p align="center">
<img src="https://github.com/szeighami/nudge/blob/main/nudge_overview.jpg" width="500">
</p>

## Setup
TODO 

## Getting Started
The following code shows an example of using NUDGE to fine-tune embeddings on [nfcorpus](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/). The code is also available in this [notebook](https://github.com/szeighami/nudge/blob/main/example.ipynb). 

Load dataset and embed the data and queries:
```python
from utils import load_hf_datasets, embed_data_and_query_sets
dataset_name = 'nfcorpus'
dataset, query_sets = load_hf_datasets(dataset_name)
data_emb, query_sets = embed_data_and_query_sets(dataset, query_sets, "BAAI/bge-small-en-v1.5")
```
Fine-tune Embeddings (can alternatively use `NUDGEM`):
```python
from nudge import NUDGEN
finetunde_embs_nudge_n = NUDGEN().finetune_embeddings(data_emb, query_sets['train'], query_sets['dev'])
```
Use fine-tuned embeddings to answer queries:
```python
from knnretriever import kNNRetriever
nudge_n_res = kNNRetriever(finetunde_embs_nudge_n).retrieve_topk_from_emb_batch(k=10, q_embeds=query_sets['test']['q_embs'])
```
Use non-fine-tuned embeddings to answer queries:
```python
no_ft_res = kNNRetriever(data_emb).retrieve_topk_from_emb_batch(k=10, q_embeds=query_sets['test']['q_embs'])
```
Compare accuracy:
```python
from utils import calc_metrics_batch, 
metrics = [('recall',10), ('ndcg',10)]
no_ft_accs = calc_metrics_batch(metrics,no_ft_res, query_sets['test']['q_ans_indx'], query_sets['test']['q_ans_indx_rel'])
nudgen_accs = calc_metrics_batch(metrics,nudge_n_res, query_sets['test']['q_ans_indx'], query_sets['test']['q_ans_indx_rel'])
print(f"No Fine-Tuning {metrics[0][0]}@{metrics[0][1]}: {no_ft_accs[0]*100:.1f}, {metrics[1][0]}@{metrics[1][1]}: {no_ft_accs[1]*100:.1f}")
print(f"NUDGE-N {metrics[0][0]}@{metrics[0][1]}: {nudgen_accs[0]*100:.1f}, {metrics[1][0]}@{metrics[1][1]}: {nudgen_accs[1]*100:.1f}")
```
Gives the output:
```
No Fine-Tuning recall@10: 31.4, ndcg@10: 33.9
NUDGE-N recall@10: 43.7, ndcg@10: 44.5
```

### Other datasets
The text datasets in the paper are hosted on huggingface [here](https://huggingface.co/sepz) (the datasets were created using this file). The above code can be run with any of `nfcorpus`, `scifact`, `arguana`, `fever`, `nq`, `triviaqa` and `hotpotqa`. We do not host the image datasets used in the paper, but running `python run_end_to_end.py` downloads and processes the image datasets and runs the experiments on the image datasets (as well as text datasets).

### Larger Datasets
For the larger dataset (i.e., `fever`, `nq`, `triviaqa` and `hotpotqa`), you may run out of memory if you run the above. Instead, `NUDGE` allows for an optimization where data records that are not an answer to any of the training or validation queries are filtered out and accounted for separately. Such data records still impact fine-tuning, but only through their impact on validation accuracy. The following code
```python
max_nontest_index = -1
for split in ["train", "dev"]:
    max_nontest_index = max(np.array([indx for curr_q_ans_indx in query_sets[split]['q_ans_indx'] for indx in curr_q_ans_indx]).max()+1, max_nontest_index)
nontrain_dataset = dataset.loc[max_nontest_index:]
if nontrain_dataset.shape[0] == 0:
    embeddings = data_emb
    nontrain_embeddings  = None
else:
    embeddings = data_emb[:max_nontest_index]
    nontrain_embeddings = data_emb[max_nontest_index:]
    
new_embs_nudgen = NUDGEN().finetune_embeddings(embeddings, query_sets['train'], query_sets['dev'], (nontrain_embeddings, None))
nudge_n_res = kNNRetriever(new_embs_nudgen, nontrain_embeddings).retrieve_topk_from_emb_batch(k=10, q_embeds=query_sets['test']['q_embs'])
```
gives the same result as 
```python
finetunde_embs_nudge_n = NUDGEN().finetune_embeddings(data_emb, query_sets['train'], query_sets['dev'])
nudge_n_res = kNNRetriever(finetunde_embs_nudge_n).retrieve_topk_from_emb_batch(k=10, q_embeds=query_sets['test']['q_embs'])
```
but uses less memory if many data records are not an answer to any training query. Complete code running `nq` using the above optimization is available [here](https://github.com/szeighami/nudge/blob/main/example_large_datasets.ipynb).


## Running End to End Experiments

TODO

