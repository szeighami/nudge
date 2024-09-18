import math
import copy
import time
import gc

import numpy as np
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, evaluation 
from torch.utils.data import DataLoader
import transformers
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.functional import normalize



class Evaluator(evaluation.SentenceEvaluator):
    def __init__(self, tokenizer, val_sets, corpus, device, val_batch_size):
        super().__init__()
        self.tokenizer = tokenizer
        self.val_sets = val_sets
        self.corpus = ["" if x is None else x for x in corpus]
        self.batch_size = val_batch_size
        self.primary_metric = "recall@1"
        self.start = time.time()
        self.vals = {"epoch":[], "step":[], "time":[]}
        self.device=device
        for name, _, _ in val_sets:
            self.vals[name] = []

    def __call__( self, model: "SentenceTransformer", output_path: str = None, epoch: int = -1, steps: int = -1, *args, **kwargs):
        maxs ={}
        indxs ={}
        all_new_q_embs = {}
        for name, qs, ans in self.val_sets:
            maxs[name] = []
            indxs[name] = []
            tokenized_qs = self.tokenizer(qs, padding=True, truncation=True, return_tensors='pt').to(self.device)
            with torch.no_grad():
                new_q_embs = []
                for i in range(int(math.ceil(len(tokenized_qs["input_ids"])/self.batch_size))):
                    curr_batch  = {}
                    for x in tokenized_qs:
                        curr_batch[x] = tokenized_qs[x][i*self.batch_size:(i+1)*self.batch_size]
                    new_q_embs.append(model(curr_batch)['sentence_embedding'].detach())
                all_new_q_embs[name] = torch.cat(new_q_embs, dim=0)
 
            del tokenized_qs

        for i in tqdm(range(int(math.ceil(len(self.corpus)/self.batch_size)))):
            tokenized_docs = self.tokenizer(self.corpus[i*self.batch_size:(i+1)*self.batch_size], padding=True, truncation=True, return_tensors='pt').to(self.device)

            with torch.no_grad():
                new_doc_embs = normalize(model(tokenized_docs)['sentence_embedding'].detach())

            for name, _, _ in self.val_sets:
                res = torch.matmul(all_new_q_embs[name], torch.t(new_doc_embs)).max(dim=-1, keepdim=True)
                maxx = res.values
                maxx_indx = res.indices
                maxs[name].append(maxx)
                indxs[name].append(maxx_indx+i*self.batch_size)

        val_acc = None
        for name, _, ans in self.val_sets:
            curr_indxs = torch.cat(indxs[name], dim=1)
            preds = torch.cat(maxs[name], dim=1).max(dim=-1, keepdim=True).indices
            preds = torch.gather(curr_indxs, 1, preds)
            preds = torch.clamp(preds, max=ans.shape[-1]-1).long()

            truths = torch.gather(ans, 1, preds)
            acc = truths.sum()/preds.shape[0]
            self.vals[name].append(acc.cpu().numpy())
            if name == "val":
                val_acc = acc
            print("acc", acc)
        assert val_acc is not None, 'one evaluation set must be called val'
        del all_new_q_embs
        del new_doc_embs
        gc.collect()
        torch.cuda.empty_cache()

        self.vals["step"].append(steps)
        self.vals["epoch"].append(epoch)
        end = time.time()
        self.vals["time"].append(end - self.start)
        return {"eval_recall@1":float(val_acc.cpu().numpy())}



class PTFT:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def train_model(self, dataloader, evaluator, model, tokenizer, steps, output_path, val_freq, lr, min_sim, scale, max_drop, warmup_steps):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if warmup_steps > 0:
            scheduler_obj = transformers.get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=steps
            )
        
        running_loss = 0
        total_samples = 0
        pbar = tqdm(range(steps))
        data_iterator = iter(dataloader)
        cross_entropy_loss = nn.CrossEntropyLoss()
        max_val_acc = -1
        for i in pbar:
            if i % val_freq == 0:
                val = evaluator(model, steps=i)
                if val['eval_recall@1']>max_val_acc:
                    max_val_acc = val['eval_recall@1']
                    torch.save(model.state_dict(), output_path+"_modeldict")
                if max_val_acc-val['eval_recall@1']>max_drop:
                    break

            try:
                data = next(data_iterator)
            except StopIteration:
                data_iterator = iter(dataloader)
                data = next(data_iterator)
                print("NEW EPOCH", "loss", running_loss/total_samples)
                running_loss = 0
                total_samples = 0

            query, doc = data
            tokenized_qs = tokenizer(query, padding=True, truncation=True, return_tensors='pt').to(self.device)
            q_embs = normalize(model(tokenized_qs.to(self.device))['sentence_embedding'])

            tokenized_docs = tokenizer(doc, padding=True, truncation=True, return_tensors='pt').to(self.device)
            new_doc_embs = normalize(model(tokenized_docs)['sentence_embedding'])

            scores = util.cos_sim(q_embs, new_doc_embs) * scale
            mask = scores.detach() >= min_sim
            scores = scores*mask
            range_labels = torch.arange(0, scores.size(0), device=self.device)
            loss = cross_entropy_loss(scores, range_labels)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if warmup_steps > 0:
                scheduler_obj.step()
            running_loss += loss.item()
            total_samples += len(query)
            pbar.set_description(f"loss:{loss.item()/len(query)} {val['eval_recall@1']}")

        model.load_state_dict(torch.load(output_path+"_modeldict"))
            
    def finetune_model(self, train_set, val_set, dataset, dataset_nontrain, nontrain_indx_to_use, no_steps, batch_size, val_freq, lr, loss_min_sim_thresh, loss_scale, only_last_layer, checkpoint_path, max_drop, warmup_steps, val_batch_size, emb_model):
        df_train_qs = train_set["q_df"].reset_index()
        train_anss = train_set["q_ans_indx"]
        qs_val = list(val_set["q_df"]['input'].values)
        val_anss = val_set["q_ans_indx"]
        corpus = list(dataset['text'].values)
        if dataset_nontrain is not None:
            if nontrain_indx_to_use is not None:
                corpus.extend(list(dataset_nontrain.iloc[nontrain_indx_to_use-len(corpus)]['text'].values))
            else:
                corpus.extend(list(dataset_nontrain['text'].values))
            print("val corpus size", len(corpus))

        print("creating training val set")
        plain_text_train = []
        for i, r in tqdm(df_train_qs.iterrows(), total=len(df_train_qs)):
            for ans_indx in train_anss[i]:
                    plain_text_train.append((r['input'], dataset.iloc[ans_indx]['text']))

        steps_per_epoch = math.ceil(len(plain_text_train)/batch_size)
        no_epochs = int(math.ceil(no_steps/steps_per_epoch))

        answer_val_size = len(dataset)
        if dataset_nontrain is not None:
            answer_val_size += 1 
        answer_val = torch.zeros(len(qs_val), answer_val_size).to(self.device)
        for i in tqdm(range(len(qs_val))): 
            for j in range(len(val_anss[i])):
                emb_indx = val_anss[i][j]
                answer_val[i, emb_indx] = 1

        train_dataloader = DataLoader(plain_text_train, shuffle=True, batch_size=batch_size)

        model = SentenceTransformer(f"{emb_model}")
        tokenizer= AutoTokenizer.from_pretrained(f"{emb_model}")
        
        if only_last_layer:
            auto_model = model._first_module().auto_model

            for param in auto_model.parameters():
                param.requires_grad = False
                
            for param in auto_model.encoder.layer[-2].parameters():
                param.requires_grad = True
            for layer in auto_model.encoder.layer[-1].parameters():
                layer.requires_grad_ = True
            for param in auto_model.pooler.parameters(): 
                param.requires_grad = True 

        evaluator = Evaluator(tokenizer, [("val", qs_val, answer_val)], corpus, self.device, val_batch_size)
        self.train_model(train_dataloader, evaluator, model, tokenizer, steps=no_steps, output_path=checkpoint_path, val_freq=val_freq, lr=lr, min_sim=loss_min_sim_thresh, scale=loss_scale, max_drop=max_drop, warmup_steps=warmup_steps)

        return model


