from nudge import NUDGEM, NUDGEN
from util.knnretriever import kNNRetriever
from util.utils import calc_metrics_batch, load_hf_datasets, embed_data_and_query_sets

dataset_name = 'nfcorpus'
dataset, query_sets = load_hf_datasets(dataset_name)
data_emb, query_sets = embed_data_and_query_sets(dataset, query_sets, "BAAI/bge-small-en-v1.5")

nudgen =  NUDGEN()
new_embs_nudgen = nudgen.finetune_embeddings(data_emb, query_sets['train'], query_sets['dev'])
nudge_nret = kNNRetriever(new_embs_nudgen)
nudge_n_res = nudge_nret.retrieve_topk_from_emb_batch(k=10, q_embeds=query_sets['test']['q_embs'])

nudgem =  NUDGEM()
new_embs_nudgem = nudgem.finetune_embeddings(data_emb, query_sets['train'], query_sets['dev'])
nudge_mret = kNNRetriever(new_embs_nudgem, dist_metric='dot')
nudge_m_res = nudge_mret.retrieve_topk_from_emb_batch(k=10, q_embeds=query_sets['test']['q_embs'])


no_ft_ret = kNNRetriever(data_emb)
no_ft_res = no_ft_ret.retrieve_topk_from_emb_batch(k=10, q_embeds=query_sets['test']['q_embs'])

metrics = [('recall',10), ('ndcg',10)]
no_ft_accs = calc_metrics_batch(metrics,no_ft_res, query_sets['test']['q_ans_indx'], query_sets['test']['q_ans_indx_rel'])
nudgem_accs = calc_metrics_batch(metrics,nudge_m_res, query_sets['test']['q_ans_indx'], query_sets['test']['q_ans_indx_rel'])
nudgen_accs = calc_metrics_batch(metrics,nudge_n_res, query_sets['test']['q_ans_indx'], query_sets['test']['q_ans_indx_rel'])
print(f"No Fine-Tuning {metrics[0][0]}@{metrics[0][1]}: {no_ft_accs[0]*100:.1f}, {metrics[1][0]}@{metrics[1][1]}: {no_ft_accs[1]*100:.1f}")
print(f"NUDGE-M {metrics[0][0]}@{metrics[0][1]}: {nudgem_accs[0]*100:.1f}, {metrics[1][0]}@{metrics[1][1]}: {nudgem_accs[1]*100:.1f}")
print(f"NUDGE-N {metrics[0][0]}@{metrics[0][1]}: {nudgen_accs[0]*100:.1f}, {metrics[1][0]}@{metrics[1][1]}: {nudgen_accs[1]*100:.1f}")
