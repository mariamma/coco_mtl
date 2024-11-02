import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import numpy as np

from model import MTLModel
from dataset import CocoKeypointsDataset, CocoInstancesDataset, dataloader_collate, CocoCaptionDataset

import wandb
import os
import torchvision


def get_noisy_data(images):
    noisy_images = []
    for x in images:
        noisy_images.append(x+ (torch.randn_like(x) * 0.01))
    return noisy_images   

        
def get_det_saliency(images, noised_images, model, targets):  
    model.eval()    
    for img in images: 
        img.requires_grad_()
    det_output, seg_output, kp_output, cap_output = model(images)
    det_output = det_output[0][0]
    bboxes = det_output['boxes']
    det_labels = det_output['labels'].cpu().detach().numpy()    

    noised_det_output, noised_seg_output, noised_kp_output, noised_cap_output = model(noised_images)
    noised_det_output = noised_det_output[0][0]
    noised_bboxes = noised_det_output['boxes']
    noised_det_labels = noised_det_output['labels'].cpu().detach().numpy()

    checked = []
    target_labels = targets['labels'].cpu().detach().numpy()
    for label in target_labels:
        if label in checked:
            continue
        if label not in det_labels :
            continue
        if label not in noised_det_labels:
            continue
        checked.append(label.item())

        det_label_idx = np.where(det_labels == label)
        det_bbox = bboxes[det_label_idx]
        det_saliency = torch.autograd.grad(det_bbox.sum(), images, retain_graph=True)[0]
        det_saliency = torch.flatten(det_saliency)
            
        noised_det_label_idx = np.where(noised_det_labels == label)
        noised_det_bbox = noised_bboxes[noised_det_label_idx]
        noised_det_saliency = torch.autograd.grad(noised_det_bbox.sum(), noised_images, retain_graph=True)[0]
        noised_det_saliency = torch.flatten(noised_det_saliency)
            
        det_sal_robustness = det_saliency - noised_det_saliency
        return det_sal_robustness


def get_caption_saliency(images, noised_images, model, targets):
    model.train()
    det_output, seg_output, kp_output, cap_output = model(images, targets)
    
    cap_saliency = torch.autograd.grad(cap_output.sum(), images, retain_graph=True)[0]
    cap_saliency = torch.flatten(cap_saliency)

    det_output, seg_output, kp_output, noised_cap_output = model(noised_images, targets)
    noised_cap_saliency = torch.autograd.grad(noised_cap_output.sum(), images, retain_graph=True)[0]
    noised_cap_saliency = torch.flatten(noised_cap_saliency)

    cap_sal_robustness = cap_saliency - noised_cap_saliency
    return cap_sal_robustness


def eval(model, train_loader, device, args):

    train_iter = 0
    
    flag = 0
    rho_sum = 0
    num_iter = 0
    for images, targets in train_loader:  

        images = [img.to(device) for img in images]
        for img in images: 
            img.requires_grad_()
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        noised_images = get_noisy_data(images)    
        for img in noised_images: img.requires_grad_()
    
        det_sal_robustness = get_det_saliency(images, noised_images, model, targets[0])  
        seg_sal_robustness = get_caption_saliency(images, noised_images, model, targets)
        if det_sal_robustness !=None and seg_sal_robustness !=None:
            det_correlation = torch.dot(det_sal_robustness, seg_sal_robustness) / (torch.norm(det_sal_robustness) * torch.norm(seg_sal_robustness))
            
            rho_sum += det_correlation.cpu().detach().numpy()
            num_iter += 1
    print("Average rho val : {}".format(rho_sum/num_iter))      
    return  rho_sum/num_iter



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
    
    dataset = CocoCaptionDataset(datasetDir='/data6/rajivporana_scratch/coco_2017', split='val2017', caption_len=None)
    vocab_size = dataset.get_vocab_len()
    print("Vocab size : ", vocab_size)

    model = MTLModel(pretrained=True, tasks=tasks, 
        trainable_layers = args.trainable_layers, vocab_size = vocab_size,
        eval_mode = True)
    model = load_model(model, args.net_basename, folder=args.model_folder, name=args.model_name)
    model.to(device)

    rho_list = []
    for caption_len in range(9, 30):
        dataset = CocoCaptionDataset(datasetDir='/data6/rajivporana_scratch/coco_2017', split='val2017', caption_len=caption_len)
        train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataloader_collate)
        print('length loader :' , len(train_loader))

        rho_val = eval(model, train_loader, device, args)
        rho_list.append(rho_val)
    print("Final rho : ", sum(rho_list)/len(rho_list))               
       

       

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

