import pickle
import argparse

from tqdm import tqdm
import pandas as pd
import numpy as np

from datasets import load_dataset
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel


def embed(q_set, data_path, prefix, emb_model, coco_path, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(f"{prefix}/{emb_model}").to(device)
    processor = CLIPProcessor.from_pretrained(f"{prefix}/{emb_model}")

    i = 0
    j = 0
    q_id = 0
    images = []
    texts = []
    img_embs = []
    txt_embs = []
    
    dataset ={"doc_id":[],"passage_id":[]}
    queries ={"q_id":[],"input":[],"doc_id":[]}
    if q_set == "coco":
        cap = dset.CocoCaptions(root = f'{coco_path}/train2014',
                                annFile = f'{coco_path}/annotations/captions_train2014.json',
                                transform=transforms.PILToTensor())
        data_iter = cap
    elif q_set == "flickr":
        ds = load_dataset("nlphuji/flickr30k")
        data_iter = range(len(ds['test']))
    else:
        print("WRONG Q")
        exit()
    
    for iter_val in tqdm(data_iter):
        if q_set == "coco":
            img, caps = iter_val
        elif q_set == "flickr":
            img = ds['test'][iter_val]['image']
            caps = ds['test'][iter_val]['caption']
        dataset['doc_id'].append(str(j))
        dataset['passage_id'].append(str(0))
        images.append(img)
        texts.extend(caps)
        for txt in caps:
            queries['doc_id'].append(str(j))
            queries['input'].append(txt)
            queries['q_id'].append(q_id)
            q_id+=1
        j+=1

        if i == batch_size:
            inputs = processor(text=texts, return_tensors="pt", padding=True)
            inputs['input_ids'] = inputs['input_ids'].to(device)[:, :77]
            inputs['attention_mask'] = inputs['attention_mask'].to(device)[:, :77]
            outputs = model.get_text_features(**inputs)
            txt_emb = outputs.detach().cpu().numpy()
            txt_embs.append(txt_emb)

            del texts
            texts = []


            inputs = processor(images=images, return_tensors="pt")
            inputs['pixel_values'] = inputs['pixel_values'].to(device)
            outputs = model.get_image_features(**inputs)
            img_emb = outputs.detach().cpu().numpy()
            img_embs.append(img_emb)

            del images
            images = []

            i = 0

        i+=1

    if len(images)>0:
        inputs = processor(text=texts, return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        outputs = model.get_text_features(**inputs)
        txt_emb = outputs.detach().cpu().numpy()
        txt_embs.append(txt_emb)

        del texts
        texts = []

        inputs = processor(images=images, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].to(device)
        outputs = model.get_image_features(**inputs)
        img_emb = outputs.detach().cpu().numpy()
        img_embs.append(img_emb)

        del images
        images = []

    txt_embs = np.concatenate(txt_embs, axis=0) 
    img_embs = np.concatenate(img_embs, axis=0)

    dataset_name = q_set
    split_type = "all"
    split = "all"
    np.save(f"{data_path}/all_data_emb_{emb_model}_{dataset_name}.npy", img_embs)
    np.save(f"{data_path}/qs_emb_{emb_model}_{q_set}_{split_type}_{split}.npy", txt_embs)

    data_df = pd.DataFrame.from_dict(dataset)
    q_df = pd.DataFrame.from_dict(queries)
    data_df.to_parquet(f'{data_path}/dataset_{dataset_name}.parquet')
    q_df.to_csv(f'{data_path}/qs_{q_set}_{split_type}_{split}.csv')


def create_splits(q_set, data_path):
    run_no = 0
    np.random.seed(run_no)
    dataset_name= q_set
    split_type = "all"
    split = "all"
    queries= pd.read_csv(f'{data_path}/qs_{q_set}_{split_type}_{split}.csv')
    dataset= pd.read_parquet(f'{data_path}/dataset_{dataset_name}.parquet').set_index("doc_id")
    dataset["row_number"] = np.arange(len(dataset))

    q_ans_indx = []
    answer_passage_ids = []
    i = 0
    for i, q in queries.iterrows():
        indx = dataset.loc[str(q['doc_id'])]['row_number']
        q_ans_indx.append([indx])
        answer_passage_ids.append([(q['doc_id'], 0)])

    q_embs = np.load(f"{data_path}/qs_emb_{emb_model}_{q_set}_{split_type}_{split}.npy")

    val_size = min(10000, int(0.1*len(q_embs)))
    test_size = min(10000, int(0.2*len(q_embs)))
    train_size = len(q_embs)-val_size-test_size
    rand_indx = np.random.permutation(len(q_embs))
    train_indx = rand_indx[:train_size]
    val_indx = rand_indx[train_size:train_size+val_size]
    test_indx = rand_indx[train_size+val_size:]
    splits_indx = {"train":train_indx, "dev":val_indx, "test":test_indx}
    splits = ["train", "dev", "test"]
    for split in splits:
        split_embs = q_embs[splits_indx[split]]
        np.save(f"{data_path}/qs_emb_{emb_model}_{q_set}_{split}.npy", split_embs)
        split_qs = queries.iloc[splits_indx[split]]
        split_qs.to_csv(f"{data_path}/qs_{q_set}_{split}.csv")

        split_answer_passage_ids = []
        split_q_ans_indx = []
        for indx in splits_indx[split]:
            split_answer_passage_ids.append(answer_passage_ids[indx])
            split_q_ans_indx.append(q_ans_indx[indx])

        with open (f"{data_path}/q_ans_{q_set}_{split}.pckl", 'wb') as fp:
            pickle.dump(split_answer_passage_ids, fp)
        with open (f"{data_path}/q_ans_index_{q_set}_{split}.pckl", 'wb') as fp:
            pickle.dump(split_q_ans_indx, fp)

def process_all(data_path, coco_path):
    batch_size = 32
    i = 0
    for emb_prefix, emb_model  in [("openai", "clip-vit-base-patch32"), ("openai", "clip-vit-large-patch14")]:
        for q_set in ['coco', 'flickr']:
            print(q_set, emb_model)
            if i == 0:
                i+=1
                continue
            embed(q_set, data_path, emb_prefix, emb_model, coco_path, batch_size)
            create_splits(q_set, data_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str,
                                help='Directory for to store processed data files')
    parser.add_argument('coco_path', type=str,
                                help='Directory for unprocessed coco files')
    args = parser.parse_args()
    process_all(args.data_path, args.coco_path)

