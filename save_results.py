# map calculation on coco dataset for object detection
import torch

from dataset import CocoInstancesDataset, CocoKeypointsDataset
from model import MTLModel
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F, transforms
from mapcalc import calculate_map
import numpy as np
import os
import json

from dataset import CocoKeypointsDataset, CocoInstancesDataset, dataloader_collate
# COCO Evaluation utility from torchvision
# from torchvision.ops import CocoEvaluator

from pycocotools.coco import COCO

import argparse 

parser = argparse.ArgumentParser(description='Save results for object detection, segmentation and keypoints')
parser.add_argument('--device', type=str, default='cuda:3', help='device to run the model on')

args = parser.parse_args()

dir_name = "/data6/rajivporana_scratch/coco_mtl/saved_models/"

dataset_k = CocoKeypointsDataset(datasetDir='/data6/rajivporana_scratch/coco_2017', split='val2017')
dataset_ds = CocoInstancesDataset(datasetDir='/data6/rajivporana_scratch/coco_2017', split='val2017')

p1, p2, p3 = ['detection_lemon-wave-75_epoch15_best_model.pkl',
                 'detection_segmentation_scarlet-donkey-74_epoch15_best_model.pkl',
                 'detection_keypoints_mild-oath-77_epoch14_best_model.pkl']

for model_path in  [p1,p2,p3]:
    model_weights_path = os.path.join(dir_name, model_path)
    #use model_weights_path to load the MTLmodel()
    tasks = [i for i in ('detection', 'keypoints', 'segmentation') if i in model_path]
    print(tasks)
    model = MTLModel(pretrained= False, tasks=tasks)
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cuda:4'))['model_rep'])

    model.eval()
    model.to(args.device)
    if 'keypoints' in tasks:
        dataset = CocoKeypointsDataset(datasetDir='/data6/rajivporana_scratch/coco_2017', split='val2017' )
    else:
        dataset = CocoInstancesDataset(datasetDir='/data6/rajivporana_scratch/coco_2017', split='val2017')
    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, collate_fn=dataloader_collate)
    print('length loader :' , len(train_loader))


    # store results in a list on cpu
    results = []
    ctr = 0
    for images, targets in train_loader:
        images = [img.to(args.device) for img in images]
        targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]
        targets = targets[0]
 
        det_result, seg_result, kp_result, _ = model(images)
        det_result = det_result[0][0]
        for box, label, score in zip(det_result['boxes'], det_result['labels'], det_result['scores']):
            results.append({'image_id': targets['image_id'].item(), 'category_id': label.item(), 'bbox': box.tolist(), 'score': score.item()})
      
        break
    #     ctr += 1
    #     if ctr % 100 == 0:
    #         print(f"Processed {ctr} images")
    # save_file_name = '_'.join(tasks)
    # # store the results in a json file
    # with open(f'{save_file_name}_2_results.json', 'w') as f:
    #     json.dump(results, f)


# store the results as json file 
# as list of dicts: {"image_id":42,"category_id":18,"bbox":[258.15,41.29,348.26,243.78],"score":0.236}
"""

"""

