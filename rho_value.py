import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import numpy as np

from model import MTLModel
from dataset import CocoKeypointsDataset, CocoInstancesDataset, dataloader_collate

import wandb
import os
import torchvision


def get_noisy_data(images):
    noisy_images = []
    for x in images:
        noisy_images.append(x+ (torch.randn_like(x) * 0.01))
    return noisy_images   

def eval(model, train_loader, device, args):

    train_iter = 0
    model.eval()
    flag = 0
    rho_sum = 0
    num_iter = 0
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        for img in images: img.requires_grad_()
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
 
        targets = targets[0]

        det_output, seg_output, kp_output, cap_output = model(images)
        
        det_output = det_output[0][0]
        bboxes = det_output['boxes']
        det_labels = det_output['labels'].cpu().detach().numpy()

        seg_output = seg_output[0][0]
        masks = seg_output['masks']
        seg_labels = seg_output['labels'].cpu().detach().numpy()

        noised_images = get_noisy_data(images)    
        for img in noised_images: img.requires_grad_()

        noised_det_output, noised_seg_output, noised_kp_output, noised_cap_output = model(noised_images)
        noised_det_output = noised_det_output[0][0]
        noised_bboxes = noised_det_output['boxes']
        noised_det_labels = noised_det_output['labels'].cpu().detach().numpy()

        noised_seg_output = noised_seg_output[0][0]
        noised_masks = noised_seg_output['masks']
        noised_seg_labels = noised_seg_output['labels'].cpu().detach().numpy()

        checked = []
        
        target_labels = targets['labels'].cpu().detach().numpy()

        for label in target_labels:
            if label in checked:
                continue
            if label not in det_labels or label not in seg_labels:
                continue
            if label not in noised_det_labels or label not in noised_seg_labels:
                continue
            checked.append(label.item())

            det_label_idx = np.where(det_labels == label)
            seg_label_idx = np.where(seg_labels == label)

            det_bbox = bboxes[det_label_idx]
            seg_mask = masks[seg_label_idx]

            det_saliency = torch.autograd.grad(det_bbox.sum(), images, retain_graph=True)[0]
            seg_saliency = torch.autograd.grad(seg_mask.sum(), images, retain_graph=True)[0]

            det_saliency = torch.flatten(det_saliency)
            seg_saliency = torch.flatten(seg_saliency)

            noised_det_label_idx = np.where(noised_det_labels == label)
            noised_seg_label_idx = np.where(noised_seg_labels == label)

            noised_det_bbox = noised_bboxes[noised_det_label_idx]
            noised_seg_mask = noised_masks[noised_seg_label_idx]

            noised_det_saliency = torch.autograd.grad(noised_det_bbox.sum(), noised_images, retain_graph=True)[0]
            noised_seg_saliency = torch.autograd.grad(noised_seg_mask.sum(), noised_images, retain_graph=True)[0]

            noised_det_saliency = torch.flatten(noised_det_saliency)
            noised_seg_saliency = torch.flatten(noised_seg_saliency)

            det_sal_robustness = det_saliency - noised_det_saliency
            seg_sal_robustness = seg_saliency - noised_seg_saliency

            det_correlation = torch.dot(det_sal_robustness, seg_sal_robustness) / (torch.norm(det_sal_robustness) * torch.norm(seg_sal_robustness))
            rho_sum += det_correlation.cpu().detach().numpy()
            num_iter += 1
    print("Average rho val : {}".format(rho_sum/num_iter))        


def get_kp_saliency(kp_output, images):
    kp_output = kp_output[0][0]
    indices = torchvision.ops.nms(kp_output['boxes'],kp_output['scores'], 0.3)
    print("indices : ", indices, torch.numel(indices))
    if torch.numel(indices) == 0:
        return None
    kp_sum = torch.zeros_like(kp_output['keypoints'][0])
    print("kp scores : ", kp_output['scores']) 
    num_added = 0       
    for index in indices:
        if kp_output['scores'][index] > 0.15:
            kp_sum += kp_output['keypoints'][index]
            num_added += 1
                # print("{} added ".format(index))
    if num_added == 0:
        kp_sum += kp_output['keypoints'][indices[0]]
    kp_saliency = torch.autograd.grad(kp_sum.sum(), images, retain_graph=True)[0]
    print("kp Saliency :", kp_saliency.shape)
    return torch.flatten(kp_saliency)
        

def kp_det_saliency(det_output, images):
    det_output = det_output[0][0]
    bboxes = det_output['boxes']
    indices = torchvision.ops.nms(det_output['boxes'],det_output['scores'], 0.3)
    print("indices : ", indices, torch.numel(indices))
    if torch.numel(indices) == 0:
        return None
    det_sum = torch.zeros_like(det_output['boxes'][0])
    print("det scores : ", det_output['scores'])        
    num_added = 0
    for index in indices:
        if det_output['scores'][index] > 0.15:
            det_sum += det_output['boxes'][index]
            num_added += 1
    if num_added == 0:
        det_sum += det_output['boxes'][indices[0]]
    det_saliency = torch.autograd.grad(det_sum.sum(), images, retain_graph=True)[0]
    print("det Saliency :", det_saliency.shape)   
    return torch.flatten(det_saliency)     


def eval_det_kp(model, train_loader, device, args):
    train_iter = 0
    model.eval()
    flag = 0
    rho_sum = 0
    num_iter = 0
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        for img in images: img.requires_grad_()
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = targets[0]

        det_output, seg_output, kp_output, cap_output = model(images)
        det_saliency = kp_det_saliency(det_output, images)
        kp_saliency = get_kp_saliency(kp_output, images)
        
        noised_images = get_noisy_data(images)    
        for img in noised_images: img.requires_grad_()
        noised_det_output, noised_seg_output, noised_kp_output, noised_cap_output = model(noised_images)
        noised_det_saliency = kp_det_saliency(noised_det_output, noised_images)
        noised_kp_saliency = get_kp_saliency(noised_kp_output, noised_images)

        if det_saliency == None or noised_det_saliency == None or \
            kp_saliency == None or noised_kp_saliency ==None:
            continue
        det_sal_robustness = det_saliency - noised_det_saliency
        seg_sal_robustness = kp_saliency - noised_kp_saliency

        det_correlation = torch.dot(det_sal_robustness, seg_sal_robustness) / (torch.norm(det_sal_robustness) * torch.norm(seg_sal_robustness))
        print("Corr : ", det_correlation)
        rho_sum += det_correlation.cpu().detach().numpy()
        num_iter += 1
    print("Average rho val : {}".format(rho_sum/num_iter))     
        
        


def load_model(model, net_basename, folder="saved_models/", name="best"):
    state = torch.load(f"{folder}{net_basename}_{name}_model.pkl", map_location='cuda:0')
    model.load_state_dict(state["model_rep"])
    return model
    
       
def main():
    if not args.debug:
        wandb.init(project="coco_mtl", group='coco', config=args, reinit=True)
        tasks_str = '_'.join(args.tasks)
        wandb.run.name = f"{tasks_str}_" + wandb.run.name
        
    n_epochs = args.n_epochs
    device = torch.device(args.device) 
    tasks = args.tasks
    
    model = MTLModel(pretrained=True, tasks=tasks, trainable_layers = args.trainable_layers)
    model = load_model(model, args.net_basename, folder=args.model_folder, name=args.model_name)
    model.to(device)

    if 'keypoints' in tasks:
        dataset = CocoKeypointsDataset(datasetDir='/data6/rajivporana_scratch/coco_2017', split='val2017' )
    else:
        dataset = CocoInstancesDataset(datasetDir='/data6/rajivporana_scratch/coco_2017', split='val2017')
    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataloader_collate)
    print('length loader :' , len(train_loader))

    if 'keypoints' in tasks:
        eval_det_kp(model, train_loader, device, args)
    else:
        eval(model, train_loader, device, args)

       

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-task Learning Trainer')
    parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--device', type=str, default='cuda', help='device to train the model on')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--tasks', nargs = '+' , default=['detection', 'segmentation'], help='tasks to be trained')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--trainable_layers', type=int, default=1, help='number of epochs to train')
    parser.add_argument('--net_basename', type=str, default='', help='model name')
    parser.add_argument('--model_folder', type=str, default='/data6/rajivporana_scratch/coco_mtl/saved_models/', help='model folder')
    parser.add_argument('--model_name', type=str, default='best', help='model name')
    
    args = parser.parse_args()
    print(args)
    main()

