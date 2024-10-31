import torch
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image

from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights

from vocabulary import Vocabulary
import nltk
import json

def dataloader_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return data, target


class CocoKeypointsDataset(Dataset):
    def __init__(self, datasetDir: str, split: str) -> None:
        super().__init__()
        self.dataDir = f"{datasetDir}/images/{split}"

        self.annFile = f"{datasetDir}/annotations/person_keypoints_{split}.json"
        self._coco = COCO(self.annFile)

        self.imgIds = self._coco.getImgIds()
        self.imgIds = self._validate_ids(self.imgIds)
 
    
    def _validate_ids(self, imgIds):
        valid_ids = []
        for img_id in imgIds:
            annIds = self._coco.getAnnIds(imgIds=img_id)
            anns = self._coco.loadAnns(annIds)
            if len(anns) > 0:
                bboxes = [self._xywh_to_xyxy(ann["bbox"]) for ann in anns]
                flag = 0
                for bbox in bboxes:
                    degenerate_boxes = (bbox[0] >= bbox[2] or bbox[1] >=bbox[3]) 
                    if degenerate_boxes:
                        flag=1
                        print("Degenrate boxes : ", degenerate_boxes)
                if flag == 0:
                    valid_ids.append(img_id)
        return valid_ids    

    def _xywh_to_xyxy(self, box):
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        x2 = x + w
        y2 = y + h
        return [x, y, x2, y2]

    def __len__(self) -> int:
        return len(self.imgIds)

    def __getitem__(self, idx: int):
        img_id = self.imgIds[idx]
        img_obj = self._coco.loadImgs(img_id)[0]

        annotation = self._coco.loadAnns(self._coco.getAnnIds(imgIds=img_id))
        img = Image.open(f'{self.dataDir}/{img_obj["file_name"]}')
        img = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.transforms()(img)
        bboxes = [self._xywh_to_xyxy(ann["bbox"]) for ann in annotation]
        masks = np.array([self._coco.annToMask(ann) for ann in annotation])
        areas = [ann["area"] for ann in annotation]
        keypoints_unformatted = [ann["keypoints"] for ann in annotation]
        keypoints_formatted = []
        for kp in keypoints_unformatted:
            kp = [kp[i:i + 3] for i in range(0, len(kp), 3)]
            keypoints_formatted.append(kp)
        target = {
            "boxes": torch.tensor(bboxes, dtype=torch.float32),
            "labels": torch.tensor(
                [ann["category_id"] for ann in annotation], dtype=torch.int64
            ),
            "masks": torch.tensor(masks, dtype=torch.float32),
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(
                [ann["iscrowd"] for ann in annotation], dtype=torch.int64
            ),
            "keypoints": torch.tensor(keypoints_formatted, dtype=torch.float32),
        }
        return img, target

    def getCatLen(self):
        return len(self._coco.getCatIds())


class CocoInstancesDataset(Dataset):
    def __init__(self, datasetDir: str, split: str, 
        data_reduction = True) -> None:
        super().__init__()
        self.dataDir = f"{datasetDir}/images/{split}"
        self.COCO_labels_limit = 15

        self.annFile = f"/data5/home/rajivporana/coco_data_tasks/annotations/instances_{split}.json"
        self._coco = COCO(self.annFile)
        self.data_reduction = data_reduction

        self.imgIds = self._coco.getImgIds()
        self.imgIds = self._validate_ids(self.imgIds)
        
 
    def _validate_ids(self, imgIds):
        valid_ids = []
        for img_id in imgIds:
            annIds = self._coco.getAnnIds(imgIds=img_id)
            anns = self._coco.loadAnns(annIds)
            if len(anns) > 0:
                bboxes = [self._xywh_to_xyxy(ann["bbox"]) for ann in anns]
                flag = 0
                for bbox in bboxes:
                    # degenerate_boxes = bbox[2:] <= bbox[:2]
                    degenerate_boxes = (bbox[0] >= bbox[2] or bbox[1] >=bbox[3]) 
                    if degenerate_boxes:
                        flag=1
                        print("Degenrate boxes : ", degenerate_boxes)
                labels = [ann["category_id"] for ann in anns]
                if self.data_reduction == True:
                    if min(labels) > self.COCO_labels_limit:        
                        flag = 1
                if flag == 0:
                    valid_ids.append(img_id)
        return valid_ids

    def _xywh_to_xyxy(self, box):
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        x2 = x + w
        y2 = y + h
        return [x, y, x2, y2]

    def __len__(self) -> int:
        return len(self.imgIds)

    def __getitem__(self, idx: int):
        img_id = self.imgIds[idx]
        img_obj = self._coco.loadImgs(img_id)[0]

        annotation = self._coco.loadAnns(self._coco.getAnnIds(imgIds=img_id))
        img = Image.open(f'{self.dataDir}/{img_obj["file_name"]}')
        img = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.transforms()(img)
        bboxes = np.array([self._xywh_to_xyxy(ann["bbox"]) for ann in annotation])
        masks = np.array([self._coco.annToMask(ann) for ann in annotation])
        areas = np.array([ann["area"] for ann in annotation])
        labels = np.array([ann["category_id"] for ann in annotation])
        is_crowds = np.array([ann["iscrowd"] for ann in annotation])

        if self.data_reduction == True:
            det_label_idx = np.where(np.array(labels) <= self.COCO_labels_limit)[0]
            bboxes = bboxes[det_label_idx]
            masks = masks[det_label_idx]
            areas = areas[det_label_idx]
            labels = labels[det_label_idx]
            is_crowds = is_crowds[det_label_idx]

        target = {
            "boxes": torch.tensor(bboxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": torch.tensor(masks, dtype=torch.float32),
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(is_crowds, dtype=torch.int64)
        }
        
        return img, target

    def getCatLen(self):
        return len(self._coco.getCatIds())


class CocoCaptionDataset(Dataset):
    def __init__(self, datasetDir: str, split: str, caption_len:int,
            data_reduction = True) -> None:
        super().__init__()
        self.split = split
        self.dataDir = f"{datasetDir}/images/{split}"
        self.caption_len = caption_len
        self.data_reduction = data_reduction
        self.COCO_labels_limit = 15

        self.annFile = f"/data5/home/rajivporana/coco_data_tasks/annotations/instances_{split}.json"
        self.captions_path = f'/data5/home/rajivporana/coco_data_tasks/annotations/captions_{split}.json'
        self._coco = COCO(self.annFile)

        self.imgIds = self._coco.getImgIds()
        self.vocab = Vocabulary(
            vocab_threshold=5,
            vocab_file="./vocab.pkl",
            start_word="<start>",
            end_word="<end>",
            unk_word="<unk>",
            annotations_file=None,
            vocab_from_file=True,
        )
        self.captions_dict = self.getCaptionsData()
        self.imgIds_valid = self._validate_ids(self.imgIds)
        self.captions_len_dict = self.getCaptionsLength()
        self.imgIds = self.load_images(caption_len=caption_len)


    def get_vocab_len(self):
        return len(self.vocab)     
 
    def _validate_ids(self, imgIds):
        valid_ids = []
        for img_id in imgIds:
            annIds = self._coco.getAnnIds(imgIds=img_id)
            anns = self._coco.loadAnns(annIds)
            
            if len(anns) > 0:
                bboxes = [self._xywh_to_xyxy(ann["bbox"]) for ann in anns]
                flag = 0
                for bbox in bboxes:
                    # degenerate_boxes = bbox[2:] <= bbox[:2]
                    degenerate_boxes = (bbox[0] >= bbox[2] or bbox[1] >=bbox[3]) 
                    if degenerate_boxes:
                        flag=1
                        print("Degenrate boxes : ", degenerate_boxes)
                if img_id not in self.captions_dict:        
                    flag = 1
                    # print("Caption not present for imageid {}".format(img_id))
                labels = [ann["category_id"] for ann in anns]
                if self.data_reduction == True:
                    if min(labels) > self.COCO_labels_limit:        
                        flag = 1
                if flag == 0:
                    valid_ids.append(img_id)
        return valid_ids


    def getCaptionsLength(self):
        captions_len_dict = {}
        for img_id in self.imgIds_valid:
            caption = self.captions_dict[img_id]
            captions_len_dict[img_id] = len(caption)
        return captions_len_dict

    def load_images(self, caption_len:int):
        
        img_ids = []
        if caption_len==None:
            for img_id in self.imgIds_valid:
                img_ids.append(img_id)
                return img_ids
            
        else:    
            for img_id in self.imgIds_valid:
                if self.captions_len_dict[img_id] == caption_len:
                    img_ids.append(img_id)
            self.imgIds = img_ids
            return


    def _xywh_to_xyxy(self, box):
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        x2 = x + w
        y2 = y + h
        return [x, y, x2, y2]

    def __len__(self) -> int:
        return len(self.imgIds)

    def __getitem__(self, idx: int):
        img_id = self.imgIds[idx]
        img_obj = self._coco.loadImgs(img_id)[0]

        annotation = self._coco.loadAnns(self._coco.getAnnIds(imgIds=img_id))
        img = Image.open(f'{self.dataDir}/{img_obj["file_name"]}')
        img = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.transforms()(img)
        bboxes = np.array([self._xywh_to_xyxy(ann["bbox"]) for ann in annotation])
        masks = np.array([self._coco.annToMask(ann) for ann in annotation])
        areas = np.array([ann["area"] for ann in annotation])
        labels = np.array([ann["category_id"] for ann in annotation])
        is_crowds = np.array([ann["iscrowd"] for ann in annotation])
        
        if self.data_reduction == True:
            det_label_idx = np.where(np.array(labels) <= self.COCO_labels_limit)[0]
            bboxes = bboxes[det_label_idx]
            masks = masks[det_label_idx]
            areas = areas[det_label_idx]
            labels = labels[det_label_idx]
            is_crowds = is_crowds[det_label_idx]

        target = {
            "boxes": torch.tensor(bboxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": torch.tensor(masks, dtype=torch.float32),
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(is_crowds, dtype=torch.int64),
            "caption": self.captions_dict[img_id]
        }
        return img, target


    def getCatLen(self):
        return len(self._coco.getCatIds())


    def getCaptionsData(self):
        
        captions_data = json.load(open(self.captions_path, 'r'))
        captions_dict = {}

        for i in captions_data['annotations']:
            if i['image_id'] not in captions_dict:
                # Convert caption to tensor of word ids.
                caption = i['caption'] 
                tokens = nltk.tokenize.word_tokenize(str(caption).lower())
                caption = [self.vocab(self.vocab.start_word)]
                caption.extend([self.vocab(token) for token in tokens])
                caption.append(self.vocab(self.vocab.end_word))
                caption = torch.Tensor(caption).long()

                captions_dict[i['image_id']] = caption
             
        return captions_dict
