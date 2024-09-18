import math

import numpy as np
import torch
from torch.nn.functional import normalize

def compute_topk_sims(q_embs, embeddings_nontrain, k, dist, batch_size):
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

    if with_index:
        return topk_D, topk_I
    return topk_D

class NUDGEM:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def get_I_i(self, curr_qs_val, data_embs, upd, curr_nontrain_sims, curr_val_anss):    
        eps = 1e-8
        LARGE_NUMBER = 1/eps

        S = torch.matmul(curr_qs_val, torch.t(data_embs))
        G = torch.matmul(curr_qs_val, torch.t(upd))
        if curr_nontrain_sims is not None:
            max_nontrain_sims =curr_nontrain_sims.max(-1, keepdim=True).values
            S = torch.cat([S, max_nontrain_sims], dim=1)
            G = torch.cat([G, torch.zeros_like(max_nontrain_sims)], dim=1)


        ans_lens = [len(ans) for ans in curr_val_anss]
        flattened_ans = []
        for i in range(len(curr_qs_val)):
            for x in curr_val_anss[i]:
                flattened_ans.append(x)
        all_S =torch.repeat_interleave(S, torch.tensor(ans_lens).to(self.device), dim=0) 
        all_G =torch.repeat_interleave(G, torch.tensor(ans_lens).to(self.device), dim=0) 
        S_x = torch.gather(all_S, 1, torch.tensor(flattened_ans).to(self.device).reshape(-1, 1))
        G_x = torch.gather(all_G, 1, torch.tensor(flattened_ans).to(self.device).reshape(-1, 1))
        all_g_difs = all_G-G_x
        all_s_difs = all_S-S_x

        pos_mask = all_g_difs>0
        neg_mask = all_g_difs<0
        alphas = all_s_difs/(all_g_difs+eps)
        all_low_vals = torch.clamp(torch.max(pos_mask*(alphas), dim=1).values, min=0)
        all_high_vals = torch.min(neg_mask*alphas+(~neg_mask)*LARGE_NUMBER, dim=1).values

        zero_mask = all_g_difs==0
        final_mask = (all_s_difs*zero_mask>0).sum(dim=1)==0
        final_mask = final_mask*(all_high_vals>=all_low_vals)
        mask = torch.nonzero(final_mask)
        all_low_vals = torch.gather(all_low_vals.reshape(-1), 0, mask.reshape(-1)).cpu().numpy()
        all_high_vals = torch.gather(all_high_vals.reshape(-1), 0, mask.reshape(-1)).cpu().numpy()

        return all_low_vals.reshape(-1), all_high_vals.reshape(-1)


    def get_all_I_i(self, qs_val, val_batch_size, embs, upd, nontrain_sims, val_anss):
        maxs =[]
        mins =[]
        for val_batch_i in range(int(math.ceil(qs_val.shape[0]/val_batch_size))):
            if nontrain_sims is not None:
                curr_nontrain_sims = nontrain_sims[val_batch_i*val_batch_size:(val_batch_i+1)*val_batch_size]
            else:
                curr_nontrain_sims = None
            curr_mins, curr_maxs = self.get_I_i(qs_val[val_batch_i*val_batch_size:(val_batch_i+1)*val_batch_size], embs, upd, curr_nontrain_sims, val_anss[val_batch_i*val_batch_size:(val_batch_i+1)*val_batch_size])
            mins.append(curr_mins)
            maxs.append(curr_maxs)

        return np.concatenate(mins), np.concatenate(maxs)

    def get_point_of_most_intersection(self, mins, maxs):
        best_i = -1
        maxs = np.sort(maxs)
        mins = np.sort(mins)

        max_val = 0
        max_iter = 0
        min_iter = 0
        if len(mins) == 0:
            if len(maxs) == 0:
                return 0
            else:
                return maxs[0]

        all_vals = []
        all_signs = []
        assert maxs[-1] >= mins[-1]
        while max_iter < len(maxs):
            if min_iter == len(mins) or maxs[max_iter] < mins[min_iter]:
                all_vals.append(maxs[max_iter])
                all_signs.append(-1)
                max_iter+=1
            else:
                all_vals.append(mins[min_iter])
                all_signs.append(1)
                min_iter += 1

        j = 0
        curr_total = 0
        max_val = 0
        best_i= -1
        while j < len(all_vals):
            curr_val = all_vals[j]
            while j < len(all_vals) and all_vals[j] == curr_val:
                curr_total += all_signs[j]
                j+=1
            if curr_total > max_val:
                max_val=curr_total
                best_i = j-1

        if best_i != -1:
            alpha = all_vals[best_i]
        else:
            alpha = 0
        return alpha



    def finetune_embeddings(self, embeddings, train_set, val_set, nontrain_embeddings=None, val_batch_size=256, gamma=None):
        embeddings = normalize(torch.from_numpy(embeddings).to(self.device))
        qs_train = torch.from_numpy(train_set["q_embs"]).to(self.device)
        qs_val = torch.from_numpy(val_set["q_embs"]).to(self.device)

        nontrain_sims = None
        if nontrain_embeddings is not None:
            nontrain_sims = compute_topk_sims(qs_val, nontrain_embeddings, 1, "cos", val_batch_size)
            if nontrain_sims is not None:
                nontrain_sims = nontrain_sims.reshape((-1, 1))

        print("Calculating G")
        train_doc_to_q = [[] for i in range(embeddings.shape[0])]
        for i in range(len(qs_train)):
            for j in range(len(train_set["q_ans_indx"][i])):
                emb_indx = train_set["q_ans_indx"][i][j]
                train_doc_to_q[emb_indx].append(i)

        gs = []
        for j in range(embeddings.shape[0]):
            gs.append(torch.nan_to_num(-1*qs_train[train_doc_to_q[j]].sum(dim=0, keepdim=True)))
        gs = normalize(torch.cat(gs, dim=0))

        print("Finding gamma")
        if gamma is None:
            intrval_min_points, intrval_max_points= self.get_all_I_i(qs_val, val_batch_size, embeddings, gs, nontrain_sims, val_set["q_ans_indx"])

            gamma_star = self.get_point_of_most_intersection(intrval_min_points, intrval_max_points)
        else:
            gamma_star=gamma
        embeddings = embeddings-gamma_star*gs

        return embeddings.cpu().numpy()


class NUDGEN:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        return

    def get_train_val_sets(self, train_set, val_set, data_size):
        qs_train = torch.from_numpy(train_set["q_embs"]).to(self.device)
        train_anss = train_set["q_ans_indx"]

        train_doc_to_q = [[] for i in range(data_size)]
        for i in range(len(qs_train)):
            for j in range(len(train_anss[i])):
                emb_indx = train_anss[i][j]
                train_doc_to_q[emb_indx].append(i)
        

        qs_val = torch.from_numpy(val_set["q_embs"]).to(self.device)
        val_anss = val_set["q_ans_indx"]
        answer_val = torch.zeros(len(qs_val), data_size).to(self.device)
        for i in range(len(qs_val)): 
            for j in range(len(val_anss[i])):
                emb_indx = val_anss[i][j]
                answer_val[i, emb_indx] = 1

        return qs_train, train_doc_to_q, qs_val, answer_val
    
    def sol_maxm(self, gamma, embeddings, vTc, cTc, gs, zero_updates):
        if gamma != 0:
            mu = (vTc+torch.sqrt(gamma/(4-gamma)*(cTc-vTc**2))).reshape(-1, 1)
            Z = (gs-vTc.reshape(-1, 1)*embeddings)/torch.sqrt(cTc-vTc**2).reshape(-1, 1)
            delta = gamma*(gs-mu*embeddings)/(2*(mu-vTc.reshape(-1, 1)))
            delta[zero_updates] = 0
            new_embs = embeddings+delta
            mask = ((embeddings*normalize(gs)).sum(axis=-1)>=1-gamma/2).reshape(-1, 1)
            new_embs = mask*normalize(gs)+(~mask)*new_embs
        else:
            new_embs = embeddings
        return new_embs

    def finetune_embeddings(self, embeddings, train_set, val_set, nontrain_embeddings=None, val_batch_size=256, val_k=1, gamma=None):
        qs_train, train_doc_to_q, qs_val, answer_val = self.get_train_val_sets(train_set, val_set, embeddings.shape[0])

        nontrain_sims = None
        if nontrain_embeddings is not None:
            nontrain_sims = compute_topk_sims(qs_val, nontrain_embeddings, val_k, "cos", val_batch_size)
            if nontrain_sims is not None:
                nontrain_sims = nontrain_sims.reshape((-1, val_k))
                answer_val = torch.cat([answer_val, torch.zeros_like(answer_val[:, :val_k])], dim=1)

        embeddings = normalize(torch.from_numpy(embeddings).to(self.device))
        print("Calculating G")
        gs = []
        for j in range(embeddings.shape[0]):
            gs.append(torch.nan_to_num(1*qs_train[train_doc_to_q[j]].sum(dim=0, keepdim=True)))
        gs = torch.cat(gs, dim=0)

        print("Finding gamma")
        vTc = (gs*embeddings).sum(axis=-1)
        cTc = (gs*gs).sum(axis=-1)
        zero_updates = torch.nonzero(torch.logical_or(cTc-vTc**2 == 0, vTc <0)).reshape(-1)

        best_gamma = 0
        max_acc = -1
        label_counts = torch.clamp(answer_val.sum(dim=-1), max=val_k)
        if gamma is not None:
            gamma_vals = [gamma]
        else:
            gamma_vals = [0.02*x for x in range(25)]
        for gamma in gamma_vals:
            new_embs = self.sol_maxm(gamma, embeddings, vTc, cTc, gs, zero_updates)

            preds = []
            for val_batch_i in range(int(math.ceil(embeddings.shape[0]/val_batch_size))):
                preds.append(torch.matmul(qs_val, torch.t(new_embs[val_batch_i*val_batch_size:(val_batch_i+1)*val_batch_size])))
            preds = torch.cat(preds, dim=1)
            if nontrain_sims != None:
                preds = torch.cat([preds, nontrain_sims], dim=1)
            out_val = torch.topk(preds, val_k).indices.view(-1, val_k)
            truths = torch.gather(answer_val, 1, out_val).sum(dim=-1)
            val_acc = (truths/label_counts).sum()/out_val.shape[0]
            if val_acc > max_acc:
                max_acc = val_acc
                best_gamma = gamma

        new_embs = self.sol_maxm(best_gamma, embeddings, vTc, cTc, gs, zero_updates)

        return new_embs.cpu().numpy()



