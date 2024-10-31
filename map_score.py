import numpy as np

from model import MTLModel
from dataset import CocoKeypointsDataset, CocoInstancesDataset, dataloader_collate

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import numpy as np
import os
import torchvision
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from mapcalc import calculate_map

def load_model(model, net_basename, folder="saved_models/", name="best"):
    state = torch.load(f"{folder}{net_basename}_{name}_model.pkl", map_location='cuda:0')
    model.load_state_dict(state["model_rep"])
    return model


def remove_redundant_boxes(det_output):
    
    checked = []
    bboxes = det_output['boxes'].cpu().detach().numpy()
    det_labels = det_output['labels'].cpu().detach().numpy()
    scores = det_output['scores'].cpu().detach().numpy()
    final_bbox = []
    final_scores = []
    final_labels = []
    for label in det_labels:
        if label in checked:
            continue
        
        checked.append(label.item())

        det_label_idx = np.where(det_labels == label)
        det_bbox = bboxes[det_label_idx]
        det_scores = scores[det_label_idx]
        indices = torchvision.ops.nms(torch.from_numpy(det_bbox), torch.from_numpy(det_scores), 0.3)
        
        for ind in indices:
            final_bbox.append(det_bbox[ind])
            final_scores.append(det_scores[ind])
            final_labels.append(label)
    return final_bbox, final_scores, final_labels


def eval(model, train_loader, device, args):

    train_iter = 0
    model.eval()
    flag = 0
    rho_sum = 0
    num_iter = 0
    arr = []
    map_arr = []
    for images, targets in train_loader:
        row={}
        images = [img.to(device) for img in images]
        for img in images: img.requires_grad_()
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = targets[0]

        det_output, seg_output, kp_output, cap_output = model(images)
        det_output = det_output[0][0]
        
        final_bbox, final_scores, final_labels = remove_redundant_boxes(det_output)
        prediction_dict = {}
        prediction_dict['boxes'] = final_bbox
        prediction_dict['labels'] = final_labels
        prediction_dict['scores'] = final_scores
        target_boxes = targets['boxes'].cpu().detach().numpy()
        target_labels = targets['labels'].cpu().detach().numpy()
        ground_truth_dict = {}
        ground_truth_dict['boxes'] = target_boxes
        ground_truth_dict['labels'] = target_labels

        # for i in range(len(det_labels)):
        #     row['image_id'] = int(targets['image_id'].item())
        #     row['category_id'] = int(det_labels[i])
        #     row['bbox'] = bboxes[i].tolist()
        #     row['score'] = float(1.0)
        #     # row['score'] = float(det_scores[i])
        #     print(row)
        #     arr.append(row)
        map = calculate_map(ground_truth_dict, prediction_dict, iou_threshold=0.5)
        map_arr.append(map)
        print(map)
    print("Average map = ", sum(map_arr)/len(map))    
    return 

 
def calucate_map(resFile):
    annType = ['segm','bbox','keypoints']
    annType = annType[1]      #specify type here
    # prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
    # run prefix for detection tasks of bounding boxes only
    prefix = 'instances'
    print('Running demo for *%s* results.'%(annType))

    #initialize COCO ground truth api
    # dataDir='.'
    # dataType='val2017'
    annFile = '/data5/home/rajivporana/coco_data_tasks/annotations/instances_val2017.json'
    cocoGt=COCO(annFile)

    #initialize COCO detections api
    cocoDt=cocoGt.loadRes(resFile)

    imgIds=sorted(cocoGt.getImgIds())
    # imgId = imgIds[np.random.randint(len(imgIds))]

    print(len(imgIds))
    # running evaluation
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()    


def main():
    device = torch.device(args.device) 
    tasks = args.tasks
    
    model = MTLModel(pretrained=True, tasks=tasks)
    model = load_model(model, args.net_basename, folder=args.model_folder, name=args.model_name)
    model.to(device)

    if 'keypoints' in tasks:
        dataset = CocoKeypointsDataset(datasetDir='/data6/rajivporana_scratch/coco_2017', split='val2017' )
    else:
        dataset = CocoInstancesDataset(datasetDir='/data6/rajivporana_scratch/coco_2017', split='val2017')
    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataloader_collate)
    print('length loader :' , len(train_loader))

    eval(model, train_loader, device, args)
    # filename = "/data5/home/rajivporana/coco_data_tasks/mtl_work/full_mtl/detection_results.json"
    # with open(filename, 'w') as f:
    #     json.dump(arr, f)
    # calucate_map(filename)    

      

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-task Learning Trainer')
    parser.add_argument('--device', type=str, default='cuda', help='device to train the model on')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('--tasks', nargs = '+' , default=['detection', 'segmentation'], help='tasks to be trained')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--net_basename', type=str, default='', help='model name')
    parser.add_argument('--model_folder', type=str, default='/data6/rajivporana_scratch/coco_mtl/saved_models/', help='model folder')
    parser.add_argument('--model_name', type=str, default='best', help='model name')
    
    args = parser.parse_args()
    print(args)
    main()