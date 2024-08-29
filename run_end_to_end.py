import os
from pathlib import Path

import pandas as pd

def download_and_exract_coco_to(coco_path):
    os.system('wget http://images.cocodataset.org/zips/train2014.zip')
    os.system('wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip')
    os.system('unzip train2014.zip')
    os.system('unzip annotations_trainval2014.zip')
    os.system(f'mv train2014 {coco_path}')
    os.system(f'mv annotations {coco_path}')

data_path = os.getcwd()+"/data"
coco_path = data_path+"/coco_raw"
Path(data_path).mkdir(parents=True, exist_ok=True)
Path(coco_path).mkdir(parents=True, exist_ok=True)

output_path = os.getcwd()
Path(output_path).mkdir(parents=True, exist_ok=True)


print(f"downloading and processing text datasets. Storing text and embedding files at {data_path}")
os.system(f'python process_txt.py {data_path}')

print(f"downlading and processing image datasets. Storing text and embedding files at {data_path}")
download_and_exract_coco_to(coco_path)
os.system(f'python process_img.py {data_path} {coco_path}')

print(f"Running all tests. Storing output at {output_path}")
os.system(f'python run_baseline_tests.py {data_path} {output_path} image')
os.system(f'python run_baseline_tests.py {data_path} {output_path} text')

for model in ['bge-small-en-v1.5', 'gte-large-en-v1.5', 'clip-vit-base-patch32', 'clip-vit-large-patch14']:
    df = pd.read_csv(f"{output_path}/tests/baseline_res_{model}/res.csv")
    print(f"Avg {model} results")
    print(df[['finetuner', 'recall_1', 'recall_10', 'ndcg_10']].groupby('finetuner').mean()[['recall_1', 'recall_10', 'ndcg_10']])
print(f"See {output_path}/tests for detailed results!")

