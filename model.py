import torch
import torch.nn as nn

from torchvision.models.detection import FasterRCNN, MaskRCNN, KeypointRCNN
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights

import numpy as np

class DetectionNetwork(nn.Module):
    def __init__(self, backbone_model) -> None:
        super(DetectionNetwork, self).__init__()
        self.rpn = backbone_model.rpn
        self.roi_heads = backbone_model.roi_heads
        self.transform = backbone_model.transform


    def forward(self, img_batch, features, img_sizes, og_sizes, targets=None):
        proposals, proposal_losses = self.rpn(img_batch, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, img_sizes, targets)
        detections = self.transform.postprocess(detections, img_sizes, og_sizes)
        losses = {}
        losses.update(proposal_losses)
        losses.update(detector_losses)
        return detections, losses



class SegmentationNetwork(nn.Module):
    def __init__(self, backbone_model) -> None:
        super(SegmentationNetwork, self).__init__()
        self.rpn = backbone_model.rpn
        self.roi_heads = backbone_model.roi_heads
        self.transform = backbone_model.transform


    def forward(self, img_batch, features, img_sizes, og_sizes, targets=None):
        proposals, proposal_losses = self.rpn(img_batch, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, img_sizes, targets)
        detections = self.transform.postprocess(detections, img_sizes, og_sizes)
        losses = {}
        losses.update(proposal_losses)
        losses.update(detector_losses)
        return detections, losses


class KeypointNetwork(nn.Module):
    def __init__(self, backbone_model) -> None:
        super(KeypointNetwork, self).__init__()
        self.rpn = backbone_model.rpn
        self.roi_heads = backbone_model.roi_heads
        self.transform = backbone_model.transform


    def forward(self, img_batch, features, img_sizes, og_sizes, targets=None):
        proposals, proposal_losses = self.rpn(img_batch, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, img_sizes, targets)
        detections = self.transform.postprocess(detections, img_sizes, og_sizes)
        losses = {}
        losses.update(proposal_losses)
        losses.update(detector_losses)
        return detections, losses

class DenseposeNetwork(nn.Module):
    pass



class MTLModel(nn.Module):
    def __init__(self, pretrained=False, trainable_layers = 2,
        tasks = ['detection', 'segmentation', 'keypoints'], vocab_size=0,
        eval_mode = None) -> None:
        super(MTLModel, self).__init__()

        weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        self.preprocess = weights.transforms()
        self.trainable_layers = trainable_layers
        
        print('tasks ',tasks)

        if pretrained:
            self.weights = weights
        else:
            self.weights = None

        self.det_fullmodel = fasterrcnn_resnet50_fpn_v2(weights=self.weights)
        self.seg_fullmodel = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
        self.kp_fullmodel = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1)

        if 'detection' in tasks:
            self.transform = self.det_fullmodel.transform
            self.backbone = self.det_fullmodel.backbone
            self.freeze_backbone_layers(self.seg_fullmodel.backbone, trainable_layers=0)
            self.freeze_backbone_layers(self.kp_fullmodel.backbone, trainable_layers=0)
        else:
            if 'segmentation' in tasks:
                self.transform = self.seg_fullmodel.transform
                self.backbone = self.seg_fullmodel.backbone
                self.freeze_backbone_layers(self.det_fullmodel.backbone, trainable_layers=0)
                self.freeze_backbone_layers(self.kp_fullmodel.backbone, trainable_layers=0)
            else:
                if 'keypoints' in tasks:
                    self.transform = self.kp_fullmodel.transform
                    self.backbone = self.kp_fullmodel.backbone
                    self.freeze_backbone_layers(self.det_fullmodel.backbone, trainable_layers=0)
                    self.freeze_backbone_layers(self.seg_fullmodel.backbone, trainable_layers=0)

        self.freeze_backbone_layers(self.backbone, trainable_layers=self.trainable_layers)

        if 'detection' in tasks:
            self.det_net = DetectionNetwork(self.det_fullmodel)
            print('DetectionNetwork init')
        else:
            self.det_net = None

        if 'segmentation' in tasks:
            self.seg_net = SegmentationNetwork(self.seg_fullmodel)
            print('SegmentationNetwork init')
        else:
            self.seg_net = None

        if 'keypoints' in tasks:
            self.kp_net = KeypointNetwork(self.kp_fullmodel)
            print('KeypointNetwork init')
        else:
            self.kp_net = None

        if 'captions' in tasks:
            self.caption_net = CaptionNetwork(vocab_size=vocab_size, eval_mode = eval_mode) 
            print('CaptionNetwork init')
        else:
            self.caption_net = None                


    def forward(self, x, targets=None):

        og_sizes = []

        for img in x:
            val = img.shape[-2:]
            og_sizes.append((val[0], val[1]))

        img_batch, x, img_sizes, targets = self._get_features(x, targets)
        det_output = self._faster_rcnn(img_batch, x, img_sizes, og_sizes, targets) if self.det_net else None
        seg_output = self._mask_rcnn(img_batch, x, img_sizes, og_sizes, targets) if self.seg_net else None
        kp_output = self._keypoint_rcnn(img_batch, x, img_sizes, og_sizes, targets) if self.kp_net else None
        cap_output = self._caption_rcnn(img_batch, x, img_sizes, og_sizes, targets) if self.caption_net else None
        return det_output, seg_output, kp_output, cap_output


    def _get_features(self, x, targets=None):
        features, _ = self.transform(x)
        return features, self.backbone(features.tensors), features.image_sizes, targets

    def _faster_rcnn(self, img_batch, x, img_sizes, og_sizes, targets):
        return self.det_net(img_batch=img_batch, features=x, img_sizes=img_sizes, og_sizes=og_sizes, targets=targets)

    def _mask_rcnn(self, img_batch, x, img_sizes, og_sizes, targets):
        return self.seg_net(img_batch=img_batch, features=x, img_sizes=img_sizes, og_sizes=og_sizes, targets=targets)

    def _keypoint_rcnn(self, img_batch, x, img_sizes, og_sizes, targets):
        return self.kp_net(img_batch=img_batch, features=x, img_sizes=img_sizes, og_sizes=og_sizes, targets=targets)

    def _caption_rcnn(self, img_batch, x, img_sizes, og_sizes, targets):
        return self.caption_net(img_batch=img_batch, features=x, img_sizes=img_sizes, og_sizes=og_sizes, targets=targets)


    def freeze_backbone_layers(self, backbone, trainable_layers):
        if trainable_layers < 0 or trainable_layers > 5:
            raise ValueError(f"Trainable layers should be in the range [0,5], got {trainable_layers}")
        layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
        if trainable_layers == 5:
            layers_to_train.append("bn1")
        for name, parameter in backbone.body.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)
    


class CaptionNetwork(nn.Module):
    def __init__(self, vocab_size, embed_size = 256, hidden_size = 512, 
        num_layers=1, eval_mode=None) -> None:
        super(CaptionNetwork, self).__init__()

        self.eval_mode = eval_mode
        self.vocab_size = vocab_size
        #Map each word index to a dense word embedding tensor of embed_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        # Creating LSTM layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        # Initializing linear to apply at last of RNN layer for further prediction
        self.linear = nn.Linear(hidden_size, vocab_size)
        # Initializing values for hidden and cell state
        self.hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))
        self.criterion = (
        nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
        )

    def forward(self, img_batch, features, img_sizes, og_sizes, targets=None):
        """
        Args:
            features: features tensor. shape is (bs, embed_size)
            captions: captions tensor. shape is (bs, cap_length)
        Returns:
            outputs: scores of the linear layer

        """

        # if self.eval_mode != None:
        if targets == None:
            scores = self.sample(features)
            return scores
        
        features = torch.mean(features['pool'], (2,3))
        
        captions = []
        for target in targets:
            captions.append(target['caption'])
        captions = torch.stack(captions, dim=0)    
        
        # remove <end> token from captions and embed captions
        cap_embedding = self.embed(
            captions[:, :-1]
        )  # (bs, cap_length) -> (bs, cap_length-1, embed_size)
        
        # concatenate the images features to the first of caption embeddings.
        # [bs, embed_size] => [bs, 1, embed_size] concat [bs, cap_length-1, embed_size]
        # => [bs, cap_length, embed_size] add encoded image (features) as t=0
        embeddings = torch.cat((features.unsqueeze(dim=1), cap_embedding), dim=1)
    
        #  getting output i.e. score and hidden layer.
        # first value: all the hidden states throughout the sequence. second value: the most recent hidden state
        lstm_out, self.hidden = self.lstm(
            embeddings
        )  

        # (bs, cap_length, hidden_size), (1, bs, hidden_size)
        outputs = self.linear(lstm_out)  # (bs, cap_length, vocab_size)

        # print("outputs : ", outputs.shape)
        # print("Target : ", captions.shape)
        loss = self.criterion(outputs.view(-1, self.vocab_size), captions.view(-1))

        return loss


    def sample(self, inputs, states=None, max_len=20):
        """
        accepts pre-processed image tensor (inputs) and returns predicted
        sentence (list of tensor ids of length max_len)
        Args:
            inputs: shape is (1, 1, embed_size)
            states: initial hidden state of the LSTM
            max_len: maximum length of the predicted sentence

        Returns:
            res: list of predicted words indices
        """
        res = []
        scores = []
        inputs = torch.mean(inputs['pool'], (2,3))
        inputs = inputs.unsqueeze(1)

        # Now we feed the LSTM output and hidden states back into itself to get the caption
        for i in range(max_len):
            lstm_out, states = self.lstm(
                inputs, states
            )  # lstm_out: (1, 1, hidden_size)
            outputs = self.linear(lstm_out.squeeze(dim=1))  # outputs: (1, vocab_size)
            score, predicted_idx = outputs.max(dim=1)  # predicted: (1, 1)
            res.append(predicted_idx)
            scores.append(score)
            # if the predicted idx is the stop index, the loop stops
            if predicted_idx == 1:
                break
            inputs = self.embed(predicted_idx)  # inputs: (1, embed_size)
            # prepare input for next iteration
            inputs = inputs.unsqueeze(1)  # inputs: (1, 1, embed_size)

        scores = torch.stack(scores, 1)
        return scores