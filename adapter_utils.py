"""Adapter utils.
Based on https://github.com/run-llama/llama_index/blob/500501036e4baac72523415535110ff6cebbfeb7/llama-index-finetuning/llama_index/finetuning/embeddings/adapter_utils.py 
"""
import math
import copy

from tqdm import tqdm
from tqdm.autonotebook import trange

import transformers
import torch
from torch.nn.functional import normalize
from torch import nn
from torch.optim import Optimizer

from utils import compute_topk_sims


class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self, model, scale, sim_thresh):
        super().__init__()
        self.model = model
        self.scale = scale
        self.sim_thresh = sim_thresh
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, query_embeds, context_embeds):
        query_embeds = self.model.forward(query_embeds)
        scores = torch.matmul(normalize(query_embeds), torch.t(normalize(context_embeds))) 
        mask = scores.detach() >= self.sim_thresh
        scores = scores*mask
        scores = scores* self.scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
        return self.cross_entropy_loss(scores, labels)


def get_val_acc(q_embs, data_embs, nontrain_embeddings_or_shard, labels, val_batch_size, k):
    sim_preds = []
    for val_batch_i in range(int(math.ceil(data_embs.shape[0]/val_batch_size))):
        sim_preds.append(torch.matmul(q_embs, torch.t(data_embs[val_batch_i*val_batch_size:(val_batch_i+1)*val_batch_size])).squeeze())
    sim_preds = torch.cat(sim_preds, dim=1)
    nontrain_sims = compute_topk_sims(q_embs, nontrain_embeddings_or_shard, k, "cos")
    if nontrain_sims is not None:
        nontrain_sims = nontrain_sims.reshape(-1, k)
        sim_preds = torch.cat([sim_preds, nontrain_sims], dim=1)
    out_val = torch.topk(sim_preds, k).indices.view(-1, k)

    truths = torch.gather(labels.to(out_val.device), 1, out_val).sum(dim=-1)
    label_counts = torch.clamp(labels.to(out_val.device).sum(dim=-1), max=k)
    return (truths/label_counts).sum()/out_val.shape[0]

def train_model(model, data_loader, data_embs, val_q_embs, val_labels, nontrain_embeddings_or_shard, num_train_steps, warmup_steps, lr, val_freq, loss_scale, loss_sim_thresh, val_batch_size, device, step_wait, acc_diff_wait, val_k):
    model.to(device)
    loss_model = MultipleNegativesRankingLoss(model=model, scale=loss_scale, sim_thresh=loss_sim_thresh)
    loss_model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    steps_per_epoch = len(data_loader)
    epochs = int(math.ceil(num_train_steps/steps_per_epoch))
    if warmup_steps > 0:
        scheduler_obj = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)

    
    data_embs = normalize(data_embs).to(device)
    if (nontrain_embeddings_or_shard[0] is not None and nontrain_embeddings_or_shard[0].shape[0] > 0) or nontrain_embeddings_or_shard[1] is not None:
        val_labels = torch.cat([val_labels, torch.zeros_like(val_labels[:, :val_k])], dim=1)
        
    data_iterator = iter(data_loader)
    best_params = copy.deepcopy(model.state_dict())
    best_step = 0
    max_acc = 0
    global_step = 0
    stop_train = False

    epoch_pbar = tqdm(range(epochs))
    for epoch in epoch_pbar:
        if stop_train:
            break

        training_steps = 0
        loss_model.zero_grad()
        loss_model.train()

        if steps_per_epoch > 50:
            show_epoch_bar = True
        else:
            show_epoch_bar = False
        batch_pbar = trange(steps_per_epoch, disable=not show_epoch_bar)
        total_loss = 0
        samples = 0
        for _ in batch_pbar:
            if global_step % val_freq == 0:
                new_q_embs = model(val_q_embs.to(device))
                val_acc = get_val_acc(new_q_embs.detach(), data_embs, nontrain_embeddings_or_shard, val_labels, val_batch_size, val_k)

                if (global_step == 0) or (val_acc > max_acc):
                    max_acc = val_acc
                    best_params = copy.deepcopy(model.state_dict())
                    best_step = global_step

                if global_step-best_step > step_wait:
                    stop_train = True
                if max_acc - val_acc>acc_diff_wait:
                    stop_train = True

            try: data = next(data_iterator)
            except StopIteration:
                data_iterator = iter(data_loader)
                data = next(data_iterator)
            query, context = data
            query = query.to(device)
            context = context.to(device)

            loss_value = loss_model(query, context)
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()
            if warmup_steps > 0:
                scheduler_obj.step()

            training_steps += 1
            global_step += 1
            samples += query.shape[0]
            total_loss+=loss_value*query.shape[0]

            batch_pbar.set_description(f"loss: {total_loss/samples}")
            epoch_pbar.set_description(f" val_acc={val_acc}, loss={total_loss/samples}")

    model.load_state_dict(best_params)
    return
