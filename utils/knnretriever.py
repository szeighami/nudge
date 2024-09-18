import numpy as np
import torch

from utils import compute_topk_sims


class kNNRetriever:
    def __init__(self, embeddings, embeddings_nontrain=None, q_adaptor=None, dist_metric="cos", path_to_embeddings_nontrain_shards=None, device=None):
        assert embeddings_nontrain is None or path_to_embeddings_nontrain_shards is None, "Either read from disk or is in memory"

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.embs_nontrain=embeddings_nontrain
        if self.embs_nontrain is not None and self.embs_nontrain.shape[0] == 0:
            self.embs_nontrain=None
        self.path_to_embeddings_nontrain_shards=path_to_embeddings_nontrain_shards
        self.q_adaptor = q_adaptor
        self.dist = dist_metric
        self.embs = embeddings


    def retrieve_topk_from_emb_batch(self, k, q_embeds):
        q_embeds = torch.from_numpy(q_embeds.astype(np.float32)).to(self.device)
        if self.q_adaptor is not None:
            q_embeds = self.q_adaptor(q_embeds).detach()

        nontrain_or_path_to_shard = self.embs_nontrain,self.path_to_embeddings_nontrain_shards
        topk_nontrain = compute_topk_sims(q_embeds, nontrain_or_path_to_shard, k, "cos", with_index=True)#always use cosine distne for non-finetuned embeddings

        topk = compute_topk_sims(q_embeds, (self.embs, None), k, self.dist, with_index=True)
        I =  topk[1].cpu().numpy()
        D =  topk[0]

        if topk_nontrain is not None:
            topk_nontrain_D, topk_nontrain_I = topk_nontrain
            preds = np.concatenate([D.cpu().numpy(), topk_nontrain_D.cpu().numpy()], axis=1)
            indxs = np.concatenate([I, topk_nontrain_I.cpu().numpy()+len(self.embs)], axis=1)


            pred_topk_indx = np.argsort(-preds, axis=1)[:, :k]
            I = np.take_along_axis(indxs, pred_topk_indx, axis=1)
            D = np.take_along_axis(preds, pred_topk_indx, axis=1)

        return I 

