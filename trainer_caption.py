import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import numpy as np

from model import MTLModel
from dataset import CocoKeypointsDataset, CocoInstancesDataset, dataloader_collate, CocoCaptionDataset

import wandb
import os
from mapcalc import calculate_map

"""
kp_loss ([], {'loss_objectness': tensor(0.6330, device='cuda:0',
       grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 'loss_rpn_box_reg': tensor(0.0160, device='cuda:0', grad_fn=<DivBackward0>), 'loss_classifier': tensor(0.2969, device='cuda:0', grad_fn=<NllLossBackward0>), 'loss_box_reg': tensor(0.0233, device='cuda:0', grad_fn=<DivBackward0>), 'loss_keypoint': tensor(8.6551, device='cuda:0', grad_fn=<NllLossBackward0>)})
"""


def train(model, train_loader, optimizer, device, args):
    seg_loss_dict = {}
    for i in ['loss_objectness', 'loss_rpn_box_reg', 'loss_classifier', 'loss_box_reg', 'loss_mask', 'total_loss']:
        seg_loss_dict[f'seg_{i}'] = 0.0 
    det_loss_dict = {}
    for i in ['loss_objectness', 'loss_rpn_box_reg', 'loss_classifier', 'loss_box_reg', 'total_loss']:
        det_loss_dict[f'det_{i}'] = 0.0
    kp_loss_dict = {}
    for i in ['loss_objectness', 'loss_rpn_box_reg', 'loss_classifier', 'loss_box_reg', 'loss_keypoint', 'total_loss']:
        kp_loss_dict[f'kp_{i}'] = 0.0 
    caption_loss_dict = {}        
    caption_loss_dict['caption_total_loss'] = 0.0  

    train_iter = 0
    model.train()
    
    for images, targets in train_loader:
        optimizer.zero_grad()
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
 
        det_loss, seg_loss, kp_loss, caption_loss = model(images, targets)
        
        # print("Caption : ", caption_loss)
        
        task_loss_dict = {'detection': None, 'segmentation': None, 'keypoints': None}

        if 'detection' in args.tasks:
            det_losses = sum(det_loss[1].values())
            task_loss_dict['detection'] = det_losses
            for k in det_loss_dict.keys():
                if k != 'det_total_loss':
                    det_loss_dict[k] += det_loss[1][k.replace('det_', '')].item()  
                
            det_loss_dict['det_total_loss'] += det_losses.item()
            
        if 'segmentation' in args.tasks:
            seg_losses = sum(seg_loss[1].values())
            task_loss_dict['segmentation'] = seg_losses
            for k in seg_loss_dict.keys():
                if k != 'seg_total_loss':
                    seg_loss_dict[k] += seg_loss[1][k.replace('seg_', '')].item()
            seg_loss_dict['seg_total_loss'] += seg_losses.item()

        if 'keypoints' in args.tasks:
            kp_losses = sum(kp_loss[1].values())
            task_loss_dict['keypoints'] = kp_losses
            for k in kp_loss_dict.keys():
                if k != 'kp_total_loss':
                    kp_loss_dict[k] += kp_loss[1][k.replace('kp_', '')].item()
            kp_loss_dict['kp_total_loss'] += kp_losses.item()
        train_iter += 1

        if 'captions' in args.tasks:
            task_loss_dict['captions'] = caption_loss
            caption_loss_dict['caption_total_loss'] += caption_loss.item()

        tasks = args.tasks

        loss = sum([task_loss_dict[task] for task in tasks])

        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)

        optimizer.step()

    seg_loss_dict = {k: v/train_iter for k, v in seg_loss_dict.items()}
    det_loss_dict = {k: v/train_iter for k, v in det_loss_dict.items()}
    kp_loss_dict = {k: v/train_iter for k, v in kp_loss_dict.items()}
    caption_loss_dict = {k: v/train_iter for k, v in caption_loss_dict.items()}
    
    return det_loss_dict, seg_loss_dict, kp_loss_dict, caption_loss_dict



def save_model(models, epoch, args, folder="saved_models/", name="best"):
 
    if not os.path.exists(folder):
        os.makedirs(folder)
 
    state = {'epoch': epoch + 1,
             'model_rep': models.state_dict(),
             'args': vars(args)}

    run_name = "debug" if args.debug else wandb.run.name
    epoch = str(epoch)
    torch.save(state, f"{folder}{run_name}_{name}_model.pkl")   



def eval(model, val_loader, device, args):
    model.eval()
    map_arr = []
    
    for images, targets in val_loader:
        images = [img.to(device) for img in images]
        for img in images: img.requires_grad_()
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = targets[0]

        det_output, seg_output, kp_output, cap_output = model(images)
        det_output = det_output[0][0]
        
        # final_bbox, final_scores, final_labels = remove_redundant_boxes(det_output)
        prediction_dict = {}
        prediction_dict['boxes'] = det_output['boxes'].cpu().detach().numpy()
        prediction_dict['labels'] = det_output['labels'].cpu().detach().numpy()
        prediction_dict['scores'] = det_output['scores'].cpu().detach().numpy()

        ground_truth_dict = {}
        ground_truth_dict['boxes'] = targets['boxes'].cpu().detach().numpy()
        ground_truth_dict['labels'] = targets['labels'].cpu().detach().numpy()

        map_val = calculate_map(ground_truth_dict, prediction_dict, iou_threshold=0.5)
        map_arr.append(map_val)
   
    print("Average map on validation set = ", sum(map_arr)/len(map_arr))    
    return sum(map_arr)/len(map_arr)

       
def main():
    if not args.debug:
        wandb.init(project="coco_mtl", group='coco', config=args, reinit=True)
        tasks_str = '_'.join(args.tasks)
        wandb.run.name = f"{tasks_str}_" + wandb.run.name
        
    n_epochs = args.n_epochs
    device = torch.device(args.device) 
    tasks = args.tasks
    
    dataset = CocoCaptionDataset(datasetDir='/data6/rajivporana_scratch/coco_2017', split='train2017', caption_len=None)
    print("Dataset Length ", len(dataset))
    vocab_size = dataset.get_vocab_len()
    print("Vocab size : ", vocab_size)
    val_dataset = CocoCaptionDataset(datasetDir='/data6/rajivporana_scratch/coco_2017', split='val2017', caption_len=None)

    model = MTLModel(pretrained=True, tasks=tasks, 
        trainable_layers = args.trainable_layers, vocab_size = vocab_size)
    model.to(device)

    parameters = [p for p in model.parameters() if p.requires_grad]
    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD and AdamW are supported.")

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "multisteplr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )

    max_map = 0
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        for caption_len in range(9, 22):
            map_arr = []
            # dataset = CocoCaptionDataset(datasetDir='/data6/rajivporana_scratch/coco_2017', split='train2017', caption_len=caption_len)
            dataset.load_images(caption_len=caption_len)
            
            train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataloader_collate)
            print('Epoch :{}, Caption-len :{}, Length loader :{}'.format(epoch, caption_len, len(train_loader)))
            print("Dataset Length ", len(dataset))

            det_loss_dict, seg_loss_dict, kp_loss_dict, caption_loss_dict = \
                    train(model, train_loader, optimizer, device, args)
            
            val_dataset.load_images(caption_len=caption_len)
            val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, collate_fn=dataloader_collate)

            map_val = eval(model, val_loader, device, args)
            map_arr.append(map_val)
            # write losses to wandb by keys for each task
            if not args.debug:
                wandb.log(det_loss_dict, step=epoch)
                # wandb.log(seg_loss_dict, step=epoch)
                # wandb.log(kp_loss_dict, step=epoch)
                wandb.log(caption_loss_dict, step=epoch)
                wandb.log({'map': map_val}, step=epoch)
            if 'captions' in args.tasks: print("Caption loss : ", caption_loss_dict)
            if 'detection' in args.tasks: print("Det loss : ", det_loss_dict)
            if 'segmentation' in args.tasks: print("Seg loss : ", seg_loss_dict)
            if 'keypoints' in args.tasks: print("Keypoint loss : ", kp_loss_dict)

        lr_scheduler.step()

        # save the model
        # p = '_'.join(args.tasks)
        dirname = "/data6/rajivporana_scratch/coco_mtl/saved_models/"
        
        map_avg = sum(map_arr)/len(map_arr)
        if not args.debug and map_avg>max_map:
            max_map = map_avg
            save_model(model, epoch, args, folder= dirname, name="best")

        print("****")

    if not args.debug:   
        save_model(model, epoch, args, folder= dirname, name="last")    

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-task Learning Trainer')
    parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs to train')
    parser.add_argument('--device', type=str, default='cuda', help='device to train the model on')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--tasks', nargs = '+' , default=['detection', 'segmentation'], help='tasks to be trained')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--trainable_layers', type=int, default=1, help='number of epochs to train')
    parser.add_argument('--clip_value', type=int, default=1, help='number of epochs to train')

    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument(
        "--lr",
        default=0.1,
        
        type=float,
        help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu",
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--lr-scheduler", default="multisteplr", type=str, help="name of lr scheduler (default: multisteplr)"
    )
    parser.add_argument(
        "--lr-step-size", default=8, type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)"
    )
    parser.add_argument(
        "--lr-steps",
        default=[16, 22],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
    )
    args = parser.parse_args()
    print(args)
    main()
