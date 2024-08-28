from collections import OrderedDict

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from adapter_utils import train_model



class AdapterModel(nn.Module):
    def __init__(self, dim, no_layers, identity_init):
        super(AdapterModel, self).__init__()

        def weights_init(m, layer_no, no_layers):
            torch.nn.init.eye_(m.weight)
            if no_layers > 1:
                if layer_no == 0:
                    torch.nn.init.ones_(m.bias)
                elif layer_no == no_layers-1:
                    torch.nn.init.constant_(m.bias, -1)
                else:
                    torch.nn.init.zeros_(m.bias)

        use_bias = no_layers != 1
        fc = nn.Linear(dim, dim, bias=use_bias)
        weights_init(fc, 0, no_layers)
        layers = [('l0', fc)]
        for i in range(1, no_layers):
            layers.append(('relu{i-1}', nn.ReLU()))
            fc = nn.Linear(dim, dim, bias=use_bias)
            weights_init(fc, i, no_layers)
            layers.append(('l{i}', fc))

        self.model = nn.Sequential(OrderedDict(layers))


    def forward(self, x):
        return self.model(x)


class AdapterFineTuner:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def get_train_val_sets(self, train_set, val_set, embeddings):
        training_q_embs = torch.from_numpy(train_set["q_embs"])
        train_anss = train_set["q_ans_indx"]
        emb_size = training_q_embs.shape[1]

        plain_text_train = []

        for i in range(len(training_q_embs)): 
            for j in range(len(train_anss[i])):
                emb_indx = train_anss[i][j]
                plain_text_train.append((torch.Tensor(training_q_embs[i]).reshape(emb_size), torch.Tensor(embeddings[emb_indx]).reshape(emb_size)))

        qs_val = torch.from_numpy(val_set["q_embs"])
        val_anss = val_set["q_ans_indx"]
        answer_val = torch.zeros(len(qs_val), embeddings.shape[0])
        for i in range(len(qs_val)): 
            for j in range(len(val_anss[i])):
                emb_indx = val_anss[i][j]
                answer_val[i, emb_indx] = 1

        return plain_text_train, qs_val, answer_val

    def finetune(self, embeddings, train_set, val_set, nontrain_embeddings_or_shard, no_layers, batch_size, steps, val_freq, loss_scale, loss_sim_thresh, warmup_steps, lr, identity_init, step_wait, acc_diff_wait, val_k):
        print("preparing training data")
        torch_embeddings = torch.from_numpy(embeddings)
        plain_text_train, val_q_embs, val_labels = self.get_train_val_sets(train_set, val_set, embeddings)
        train_dataloader = DataLoader(plain_text_train, shuffle=True, batch_size=batch_size)

        emb_size = val_q_embs.shape[1]
        adapter = AdapterModel(emb_size, no_layers, identity_init)

        print("training adapter")
        train_model(adapter, train_dataloader, device=self.device, num_train_steps=steps, val_q_embs=val_q_embs, val_labels=val_labels, nontrain_embeddings_or_shard=nontrain_embeddings_or_shard, data_embs=torch_embeddings, val_freq=val_freq, loss_scale=loss_scale, loss_sim_thresh=loss_sim_thresh, warmup_steps=warmup_steps, lr=lr, val_batch_size=batch_size, step_wait=step_wait, acc_diff_wait=acc_diff_wait, val_k=val_k)

        return adapter

