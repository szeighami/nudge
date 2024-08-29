from pathlib import Path
import subprocess
import argparse
import json
import os
import sys

def get_data_config(data_path, q_set, emb_model):
    if os.path.isfile(f"{data_path}/all_data_emb_{emb_model}_{q_set}.npy"):
        load_all = True
    elif os.path.isfile(f"{data_path}/nontrain_data_emb_shards_{q_set}_{emb_model}_0.npy"):
        load_all = False
    else:
        assert False, f"embeddings not found for {q_set} with model {emb_model}"

    if load_all:
        config= {
               "path_to_sharded_emb_nontrain": None,
               "path_to_emb": None,
               "path_to_all_emb": f"{data_path}/all_data_emb_{emb_model}_{q_set}.npy",
            }
    else:
        config= {
               "path_to_sharded_emb_nontrain": f"{data_path}/nontrain_data_emb_shards_{q_set}_{emb_model}_",
               "path_to_emb": f"{data_path}/train_data_emb_{q_set}_{emb_model}.npy",
               "path_to_all_emb": None,
            }

    config["path_to_raw_dataset"] = f'{data_path}/dataset_{q_set}.parquet'

    #Expected the following files with  _train.npy, _dev.npy, _test.npy suffix to exist
    config["path_to_q_emb"] = f"{data_path}/qs_emb_{emb_model}_{q_set}"
    config["path_to_q_df"] =f"{data_path}/qs_{q_set}"
    config["path_to_q_ans"] = f"{data_path}/q_ans_index_{q_set}"
    if os.path.isfile(f"{data_path}/q_ans_index_rel_{q_set}_test.npy"):
        config["path_to_q_rel"]=f"{data_path}/q_ans_index_rel_{q_set}"
    else:
        config["path_to_q_rel"] = None
    return config

def run_all_test(data_path, output_path, image_or_text):
    #add "text-embedding-3-large" below to also run TE3-L
    ALL_TXT_MODEL_NAMES = ["bge-small-en-v1.5", "gte-large-en-v1.5"]
    ALL_IMG_MODEL_NAMES = ["clip-vit-large-patch14", "clip-vit-base-patch32"]

    TEXT_DATASETS = ['nfcorpus', "scifact", "arguana", 'fever', "nq", "triviaqa", "hotpotqa"]
    IMAGE_DATASETS = ['flickr', 'coco']

    models_for_ptft = ["bge-small-en-v1.5"]
    model_checkpoint_path = f"{output_path}/ft_models"
    ft_config = {"AdapterFineTuner": {"no_layers":1, "batch_size":4096, "steps":10000, "val_freq":50, "loss_scale":20, "loss_sim_thresh":-1, "warmup_steps": 10000, "lr": 1e-3, "identity_init":False, "step_wait":1000, "acc_diff_wait":0.05, 'val_k':1},
            "PTFT":{"no_steps":20000, "batch_size":32, "val_freq":1000, "loss_min_sim_thresh":-1, "loss_scale":20, "only_last_layer":False, "lr":2e-5, "checkpoint_path":model_checkpoint_path, "max_drop":0.05, "warmup_steps":0, "val_batch_size":256, "emb_model":"BAAI/bge-small-en-v1.5", "val_with_topk_only":15},
            "NUDGEM":{"val_batch_size":256},
            "NUDGEN":{"val_batch_size":256, 'val_k':1},
            "no_ft":{}
            }

    test_dir = f"{output_path}/tests"
    base_test_name=f"baseline_res"
    if image_or_text == "text":
        for model_name in ALL_TXT_MODEL_NAMES:
            methods_to_run = ["no_ft", "NUDGEM", "NUDGEN","AdapterFineTuner"]
            #Uncomment to also run PTFT for BGE-S
            #if model_name in models_for_ptft:
            #    methods_to_run.append('PTFT')
            #    Path(model_checkpoint_path).mkdir(parents=True, exist_ok=True)
            test_path = f"{test_dir}/{base_test_name}_{model_name}"
            Path(test_dir).mkdir(parents=True, exist_ok=True)
            Path(test_path).mkdir(parents=True, exist_ok=True)

            for q_set in TEXT_DATASETS:
                for method in methods_to_run:
                    config_path = f"{test_path}/config_{method}.json"
                    with open(config_path, mode='w') as f:
                        json.dump(ft_config[method], f)

                    data_config_path = f"{test_path}/data_config_{q_set}.json"
                    data_config = get_data_config(data_path, q_set, model_name)
                    with open(data_config_path, mode='w') as f:
                        json.dump(data_config, f)

                    cmd = f"python -u run_retrieval_eval.py {q_set} {method} {base_test_name} {test_path} {config_path} {data_config_path}"

                    cmd += f" >> {test_path}/out.txt"
                    print(cmd)
                    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
                    process.wait()

    elif image_or_text == "image":
        for model_name in ALL_IMG_MODEL_NAMES:
            test_path = f"{test_dir}/{base_test_name}_{model_name}"
            Path(test_dir).mkdir(parents=True, exist_ok=True)
            Path(test_path).mkdir(parents=True, exist_ok=True)
            test_name =base_test_name+f"_{model_name}" 
            methods_to_run = ["no_ft", "NUDGEM", "NUDGEN","AdapterFineTuner"]
            for q_set in IMAGE_DATASETS:
                for method in methods_to_run:
                    config_path = f"{test_path}/config_{method}.json"
                    with open(config_path, mode='w') as f:
                        json.dump(ft_config[method], f)

                    data_config_path = f"{test_path}/data_config_{q_set}.json"
                    data_config = get_data_config(data_path, q_set, model_name)
                    with open(data_config_path, mode='w') as f:
                        json.dump(data_config, f)

                    cmd = f"python -u run_retrieval_eval.py {q_set} {method} {base_test_name} {test_path} {config_path} {data_config_path}"
                    cmd += f" >> {test_path}/out.txt"
                    print(cmd)
                    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
                    process.wait()
    else:
        assert False, "neither ran image, nor text exps"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str,
                                help='Directory of the processed data files')
    parser.add_argument('output_path', type=str,
                                help='Directory for test outputs')
    parser.add_argument('image_or_text', type=str,
                                help='run image or text experiments. The value must be on of "image" or "text".')
    args = parser.parse_args()
    run_all_test(args.data_path, args.output_path, args.image_or_text)

