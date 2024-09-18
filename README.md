# NUDGE
NUDGE is a lightweight tool to fine-tune pre-trained embeddings for retrieval and RAG pipelines, presented in the paper [NUDGE: Lightweight Non-Parametric Embedding Fine-Tuning](https://arxiv.org/pdf/2409.02343) (see [this blog post](https://data-people-group.github.io/blogs/2024/09/05/nudge/) for a simple overview). It runs in minutes and often improves retrieval accuracy by over 10%. 

NUDGE modifies data embeddings *non-parametrically*, i.e., it does not change any model parameters but instead moves the data embeddings themselves to maximize accuracy. NUDGE solves a constrained optimization problem to do so, moving data embeddings towards the embedding of training queries for which they are the ground-truth answer. NUDGE-M and NUDGE-N are two variants of the approach, each solving the optimization problem with different constraints.

<p align="center">
<img src="https://github.com/szeighami/nudge/blob/main/nudge_overview.jpg" width="500">
</p>
As the figure above shows, NUDGE changes data embeddings within a constrained region (shown in dashed lines) to maximize similarity with training queries. Data embeddings in the figure are colored based on queries for which they are the ground-truth answers.

## Getting Started
Documentation is available along with the code in nudge/nudge.py. We further discuss how to install and use NUDGE

### Install
To install NUDGE, run 
```
pip install nudge-ft
```
### Workflow
NUDGE operates on embeddings. It fine-tunes data embeddings given training and validation queries. This package provides two classes, `NUDGEM` and `NUDGEN` to do so, implementing NUDGE-N and NUDGE-M in [the paper](https://arxiv.org/pdf/2409.02343). Both have the same interface and can be imported as

```python
from nudge import NUDGEN, NUDGEM
```

To use either class, you need to have already embedded the documents and training/validation queries and have ground-truth answers for training/validation queries. Then, call

```python
train_set = {'q_embs':train_q_embs, 'q_ans_indx':train_q_ans_indx}
val_set = {'q_embs':val_q_embs, 'q_ans_indx':val_q_ans_indx}
finetuned_embs_nudge_n = NUDGEN().finetune_embeddings(data_embs, train_set, val_set)
finetuned_embs_nudge_m = NUDGEM().finetune_embeddings(data_embs, train_set, val_set)
```
where `data_embs` is a numpy array containing data embeddings, `train_q_embs` and `val_q_embs` are numpy arrays containing embeddings of training queries and `train_q_ans_indx` and `val_q_ans_indx` contain ground-truth query answers. `train_q_ans_indx`/`val_q_ans_indx` are nested python lists, where the `i`-th item in `train_q_ans_indx`/`val_q_ans_indx` is the list of indexes of data records that are relevant to the `i`-th query. That is, `data_embs[train_q_ans_indx[i][j]]` is a positive data record for query `train_q_embs[i]`.




### Example

An end-to-end example of using `NUDGE` is shown below, to fine-tune embeddings on [nfcorpus](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/). The code is also available in this [notebook](https://github.com/szeighami/nudge/blob/main/example.ipynb), or alternatively can be run from the root of the repo with 

```
python example.py
```

After installing the dependencies below.


`NUDGE` does not embed the queries or data and operates on the embeddings directly. Thus, we first need to embed the data and queries. Here we use [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), and the `sentence_transformers` library for embeddings, and use `datasets` to load the  `nfcorpus` dataset. 

Install the two libraries

```
pip install sentence_transformers datasets
```



Load dataset and embed the data and queries:
```python
from util.utils import load_hf_datasets, embed_data_and_query_sets
dataset_name = 'nfcorpus'
dataset, query_sets = load_hf_datasets(dataset_name)
data_emb, query_sets = embed_data_and_query_sets(dataset, query_sets, "BAAI/bge-small-en-v1.5")
```
Fine-tune Embeddings (can alternatively use `NUDGEM`):
```python
from nudge import NUDGEN
finetuned_embs_nudge_n = NUDGEN().finetune_embeddings(data_emb, query_sets['train'], query_sets['dev'])
```
Use fine-tuned embeddings to answer queries:
```python
from util.knnretriever import kNNRetriever
nudge_n_res = kNNRetriever(finetuned_embs_nudge_n).retrieve_topk_from_emb_batch(k=10, q_embeds=query_sets['test']['q_embs'])
```
Use non-fine-tuned embeddings to answer queries:
```python
no_ft_res = kNNRetriever(data_emb).retrieve_topk_from_emb_batch(k=10, q_embeds=query_sets['test']['q_embs'])
```
Compare accuracy:
```python
from util.utils import calc_metrics_batch
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

**More Datasets**. More text datasets are hosted on huggingface [here](https://huggingface.co/sepz) (the datasets were created using [this](https://github.com/szeighami/nudge/blob/main/util/process_data_to_hf_datasets.py) file). The above code can be run with any of `nfcorpus`, `scifact`, `arguana`, `fever`, `nq`, `triviaqa` and `hotpotqa`.

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
finetuned_embs_nudge_n = NUDGEN().finetune_embeddings(data_emb, query_sets['train'], query_sets['dev'])
nudge_n_res = kNNRetriever(finetuned_embs_nudge_n).retrieve_topk_from_emb_batch(k=10, q_embeds=query_sets['test']['q_embs'])
```
but uses less memory if many data records are not an answer to any training query. Complete code running `nq` using the above optimization is available [here](https://github.com/szeighami/nudge/blob/main/example_large_datasets.ipynb).


## Running End to End Experiments
To reproduce all baseline experiments [in the paper](https://arxiv.org/pdf/2409.02343) (e.g, Tables 3-4) follow the instructions in the paper_exps branch of the repo.

# References
Sepanta Zeighami, Zac Wellmer, and Aditya Parameswaran. "NUDGE: Lightweight Non-Parametric Fine-Tuning of Embeddings for Retrieval." arXiv preprint arXiv:2409.02343 (2024).

@article{zeighami2024nudge,
  title={NUDGE: Lightweight Non-Parametric Fine-Tuning of Embeddings for Retrieval},
  author={Zeighami, Sepanta and Wellmer, Zac and Parameswaran, Aditya},
  journal={arXiv preprint arXiv:2409.02343},
  year={2024}
}
