# NUDGE
This repo contains the code for NUDGE: Lightweight Non-Parametric Embedding Fine-Tuning. NUDGE solves a constrained optimization problem to move data embeddings towards the embedding of training queries for which they are the ground-truth answer. NUDGE-M and NUDGE-N are two variants in this repository, each solving the optimization problem with different constraints. NUDGE takes data embeddings and a training set as input, and outputs new fine-tuned data embeddings.

<p align="center">
<img src="https://github.com/szeighami/nudge/blob/main/nudge_overview.jpg" width="500">
</p>

## Getting Started
### Setup
Using docker (recommended), create a container using the provided Dockerfile by running the following from the root of the repo:
```
docker image build -t nudge_img:1.0 -f Dockerfile .
docker container run -p 8888:8888 --gpus '"device=0"' -it --name nudge nudge_img:1.0
```
Then, you can run NUDGE from inside the container. Run [this](https://github.com/szeighami/nudge/blob/main/example.ipynb) notebook for a simple example, also discussed below. 

Alternatively, install dependencies by running the following from the root of the repo:
```
pip install -r requirements.txt
```


### Example

The following code shows an example of using NUDGE to fine-tune embeddings on [nfcorpus](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/). The code is also available in this [notebook](https://github.com/szeighami/nudge/blob/main/example.ipynb). Run the code from the root of the repo.

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
finetunde_embs_nudge_n = NUDGEN().finetune_embeddings(data_emb, query_sets['train'], query_sets['dev'])
```
Use fine-tuned embeddings to answer queries:
```python
from util.knnretriever import kNNRetriever
nudge_n_res = kNNRetriever(finetunde_embs_nudge_n).retrieve_topk_from_emb_batch(k=10, q_embeds=query_sets['test']['q_embs'])
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

### Other Datasets
The text datasets in the paper are hosted on huggingface [here](https://huggingface.co/sepz) (the datasets were created using [this](https://github.com/szeighami/nudge/blob/main/process_data_to_hf_datasets.py) file). The above code can be run with any of `nfcorpus`, `scifact`, `arguana`, `fever`, `nq`, `triviaqa` and `hotpotqa`. For running experiments on image datasets, run `python run_end_to_end.py` to download and process the image datasets and to run the experiments on the image datasets (as well as text datasets).

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
Run 
```
python run_end_to_end.py
```
to run all baseline experiments in the paper (e.g, Tables 3-4). It downloads, processes and embeds the datasets, and then runs NUDGE and the baselines on all datasets and for the open source models used in the paper. The code will output the following tables after running the experiments, each table summarizing the results for an embedding model

```
Avg bge-small-en-v1.5 results
                  recall_1  recall_10   ndcg_10
finetuner                                      
AdapterFineTuner  0.395431   0.654708  0.516281
NUDGEM            0.496965   0.666103  0.571709
NUDGEN            0.520770   0.724993  0.612650
no_ft             0.370072   0.624062  0.486494
Avg gte-large-en-v1.5 results
                  recall_1  recall_10   ndcg_10
finetuner                                      
AdapterFineTuner  0.451483   0.684443  0.556292
NUDGEM            0.520941   0.733903  0.612058
NUDGEN            0.534145   0.747116  0.626990
no_ft             0.415772   0.670296  0.532067
Avg clip-vit-base-patch32 results
                  recall_1  recall_10   ndcg_10
finetuner                                      
AdapterFineTuner   0.18475    0.43495  0.298832
NUDGEM             0.28640    0.55070  0.409338
NUDGEN             0.28795    0.55355  0.411184
no_ft              0.15870    0.40160  0.268508
Avg clip-vit-large-patch14 results
                  recall_1  recall_10   ndcg_10
finetuner                                      
AdapterFineTuner   0.24120    0.50880  0.364832
NUDGEM             0.30130    0.58115  0.431822
NUDGEN             0.30100    0.58235  0.432779
no_ft              0.20475    0.46525  0.324633
```
To also run the OpenAI models, make modifications [here](https://github.com/szeighami/nudge/blob/6a306a8525623216d4db3601e8b82af2438449d6/process_txt.py#L184) and [here](https://github.com/szeighami/nudge/blob/6a306a8525623216d4db3601e8b82af2438449d6/run_baseline_tests.py#L42) as instructed.  To also run PTFT, uncomment [here](https://github.com/szeighami/nudge/blob/6a306a8525623216d4db3601e8b82af2438449d6/run_baseline_tests.py#L63).
