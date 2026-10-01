import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.upernet.decoder import UPerNetDecoder
from segmentation_models_pytorch.base import SegmentationHead
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataloader_wrapper import DataloaderWrapper
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torchvision.transforms import v2
import sys
import os
from torchvision.transforms.functional import convert_image_dtype
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import random
import matplotlib.pyplot as plt

#GASSL
from .backbones import MoCoResNet50Backbone, SeCo_MoCoResNet50Backbone, SatMAE, TOV
from .utils import mIOU

def create_state_of_the_art_model(arch, in_channels, out_classes):

    backbonesThatNeedUPerNet = ["GASSL-basic", "GASSL-TP", "GASSL-GEO",
                                "GASSL-GEO+TP", "SeCo-100K-ResNet-18", "SeCo-1M-ResNet-18",
                                "SeCo-100K-ResNet-50", "SeCo-1M-ResNet-50", "TOV"]
    
    backbonesThatNeedTransUNet = ["SatMAE_fMoW_Non_Temporal_ViT-Large", "SatMAE_fMoW_Temporal_ViT-Large",
                                  "SatMAE_fMoW_MultiSpectral_ViT-Base", "SatMAE_fMoW_MultiSpectral_ViT-Large"]
    
    if arch == "Unet":
        model_parameters = {
            'encoder': "resnet34",
            'encoder_weights': 'imagenet'
        }
        
        encoder = smp.create_model(
                arch,
                in_channels=in_channels,
                classes=out_classes,
                encoder=model_parameters['encoder'],
                encoder_weights=model_parameters['encoder_weights']
            )
        
        decoder = None
        segmentation_head = None

        scale_inputs = False
        scaling_transform = None

    else:
        #GASSL
        if arch == "GASSL-basic":
            model_parameters = {'a': 'resnet50', 'pretrained_path': './methods/GASSL/moco_fmow/weights/moco.pth.tar'}
            encoder = MoCoResNet50Backbone(model_parameters['a'], model_parameters['pretrained_path'])
        elif arch == "GASSL-TP":
            model_parameters = {'a': 'resnet50', 'pretrained_path': './methods/GASSL/moco_fmow/weights/moco_tp.pth.tar'}
            encoder = MoCoResNet50Backbone(model_parameters['a'], model_parameters['pretrained_path'])
        elif arch == "GASSL-GEO":
            model_parameters = {'a': 'resnet50', 'pretrained_path': './methods/GASSL/moco_fmow/weights/moco_geo.pth.tar'}
            encoder = MoCoResNet50Backbone(model_parameters['a'], model_parameters['pretrained_path'])
        elif arch == "GASSL-GEO+TP":
            model_parameters = {'a': 'resnet50', 'pretrained_path': './methods/GASSL/moco_fmow/weights/moco_geo+tp.pth.tar'}
            encoder = MoCoResNet50Backbone(model_parameters['a'], model_parameters['pretrained_path'])
        
        #SeCo
        elif arch == "SeCo-100K-ResNet-18":
            model_parameters = {'a': 'resnet18', 'pretrained_path': './methods/SeCo/weights/seco_resnet18_100k.ckpt'}
            encoder = SeCo_MoCoResNet50Backbone(model_parameters['a'], model_parameters['pretrained_path'])
        elif arch == "SeCo-1M-ResNet-18":
            model_parameters = {'a': 'resnet18', 'pretrained_path': './methods/SeCo/weights/seco_resnet18_1m.ckpt'}
            encoder = SeCo_MoCoResNet50Backbone(model_parameters['a'], model_parameters['pretrained_path'])
        elif arch == "SeCo-100K-ResNet-50":
            model_parameters = {'a': 'resnet50', 'pretrained_path': './methods/SeCo/weights/seco_resnet50_100k.ckpt'}
            encoder = SeCo_MoCoResNet50Backbone(model_parameters['a'], model_parameters['pretrained_path'])
        elif arch == "SeCo-1M-ResNet-50":
            model_parameters = {'a': 'resnet50', 'pretrained_path': './methods/SeCo/weights/seco_resnet50_1m.ckpt'}
            encoder = SeCo_MoCoResNet50Backbone(model_parameters['a'], model_parameters['pretrained_path'])
        

        #SatMAE
        elif arch == "SatMAE_fMoW_Non_Temporal_ViT-Large":
            model_parameters = {'a': 'SatMAE_fMoW_Non_Temporal_ViT-Large', 'pretrained_path': './methods/SatMAE/weights/fmow_pretrain_SatMAE_fMoW_Non_Temporal_ViT-Large.pth'}
            encoder = SatMAE(model_parameters['a'], model_parameters['pretrained_path'], img_size=1024)
        elif arch == "SatMAE_fMoW_Temporal_ViT-Large":
            model_parameters = {'a': 'SatMAE_fMoW_Temporal_ViT-Large', 'pretrained_path': './methods/SatMAE/weights/pretrain_fmow_temporal_SatMAE_fMoW_Temporal_ViT-Large.pth'}
            encoder = SatMAE(model_parameters['a'], model_parameters['pretrained_path'], img_size=1024)
        elif arch == "SatMAE_fMoW_MultiSpectral_ViT-Base":
            model_parameters = {'a': 'SatMAE_fMoW_MultiSpectral_ViT-Base', 'pretrained_path': './methods/SatMAE/weights/pretrain-vit-base-e199_SatMAE_fMoW_MultiSpectral_ViT-Base.pth'}
            encoder = SatMAE(model_parameters['a'], model_parameters['pretrained_path'], img_size=1024)
        elif arch == "SatMAE_fMoW_MultiSpectral_ViT-Large":
            model_parameters = {'a': 'SatMAE_fMoW_MultiSpectral_ViT-Large', 'pretrained_path': './methods/SatMAE/weights/pretrain-vit-large-e199_SatMAE_fMoW_MultiSpectral_ViT-Large.pth'}
            encoder = SatMAE(model_parameters['a'], model_parameters['pretrained_path'], img_size=1024)

        #TOV
        elif arch == "TOV":
            model_parameters = {'a': 'TOV', 'pretrained_path': './methods/TOV/G-RSIM/TOV_v1/weights/pretrained_on_TOV-RS-balanced_ep800.pth.tar'}
            encoder = TOV(model_parameters['a'], model_parameters['pretrained_path'])


        #finally
        scale_inputs = False
        scaling_transform = create_scaling_transform(1024)


    #add segmentation head by Upernet
    if arch in backbonesThatNeedUPerNet and arch != "Unet":
        decoder_channels = 256
        decoder_use_norm = "batchnorm"
        activation = None
        upsampling = 1

        decoder = UPerNetDecoder(
            encoder_channels=encoder.out_channels,
            encoder_depth=encoder.depth,
            decoder_channels=decoder_channels,
            use_norm=decoder_use_norm,
        )

        segmentation_head = SegmentationHead(
            in_channels=decoder_channels,
            out_channels=out_classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        #https://github.com/yassouali/pytorch-segmentation/blob/master/models/upernet.py

    #add segmentation head by TransUNet
    elif arch in backbonesThatNeedTransUNet and arch != "Unet":
        #print(encoder)
        sys.path.append("./decoders/TransUNet")
        from networks.vit_seg_modeling import DecoderCup

        config = encoder.get_config()
        config.n_skip = 0 #so i dont need skip_n_channels attribute...
        decoder = DecoderCup(config)

        decoder_channels = 16
        activation = None
        upsampling = 1

        segmentation_head = SegmentationHead(
            in_channels=decoder_channels,
            out_channels=out_classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

    
    return encoder, decoder, segmentation_head, scale_inputs, scaling_transform

    

class MyModel(pl.LightningModule):
    def __init__(self, arch, in_channels, out_classes, train_channel_means, train_channel_stds, loss_function, n_iters_training):
        super().__init__()
        
        self.arch = arch
        self.out_classes = out_classes
        self.n_iters_training = n_iters_training

        encoder, decoder, segmentation_head, scale_inputs, scaling_transform = create_state_of_the_art_model(arch, in_channels, out_classes)

        self.scaling_transform = scaling_transform

        self.encoder = encoder
        self.decoder = decoder
        self.segmentation_head = segmentation_head

        # for image segmentation dice loss could be the best first choice
        if loss_function == "CrossEntropyLoss":
            self.loss_fn = smp.losses.SoftCrossEntropyLoss(reduction='mean', smooth_factor=0.0, ignore_index=None, dim=1) #smp.losses.FocalLoss(smp.losses.MULTICLASS_MODE)
        elif loss_function == "FocalLoss":
            self.loss_fn = smp.losses.FocalLoss(smp.losses.MULTICLASS_MODE, reduction='mean')
        else:
            raise Exception("Incorrect loss_function!")

        # initialize step metics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        #augmentation transforms
        self.augmentation_transform = create_augmentation_transform()

        #normalization transforms
        self.normalization_transform = create_normalization_transform(train_channel_means[:in_channels], train_channel_stds[:in_channels], scale=scale_inputs) #3 or 4 channels

    def forward(self, image, timestamps=None):
        if self.scaling_transform is not None:
            image = self.scaling_transform(image)

        if timestamps is None: #all channels
            image = self.normalization_transform(image)
        else: #per image for all channels
            image[:,0,:,:,:] = self.normalization_transform(image[:,0,:,:,:])
            image[:,1,:,:,:] = self.normalization_transform(image[:,1,:,:,:])
            image[:,2,:,:,:] = self.normalization_transform(image[:,2,:,:,:])

        if timestamps is None:
            features = self.encoder(image)
        else:
            features = self.encoder(image, timestamps)

        if self.decoder is None and self.segmentation_head is None: #in case of only using Unet
            mask = features
        else:
            decoded = self.decoder(features)
            mask = self.segmentation_head(decoded)

        mask = F.interpolate(mask, size=(1024, 1024), mode="nearest")

        return mask
    
    def joint_flip(self, image: torch.Tensor, mask: torch.Tensor, #instead of vanilla pytorch augmentations
               p_hflip: float = 0.5, p_vflip: float = 0.5):
        if random.random() < p_hflip:
            image = TF.hflip(image)
            # If mask has shape (H, W) or (1, H, W), TF.hflip works, too
            mask = TF.hflip(mask)
        if random.random() < p_vflip:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        return image, mask

    def shared_step(self, batch, stage):
        if self.arch == "SatMAE_fMoW_Temporal_ViT-Large":
            image = batch[0]
            timestamps = batch[1]
            mask = batch[2]
            assert image.ndim == 5
            h, w = image.shape[3:]
        else:
            image = batch[0]
            mask = batch[1]
            assert image.ndim == 4
            h, w = image.shape[2:]

        if self.training:
            flipped_images = []
            flipped_masks = []
            for img, m in zip(image, mask): #AUGMENTATIONS, random 0.5 vertical and horizontal augmentations
                img_f, m_f = self.joint_flip(img, m, p_hflip=0.5, p_vflip=0.5)
                flipped_images.append(img_f)
                flipped_masks.append(m_f)
            image = torch.stack(flipped_images, dim=0)
            mask  = torch.stack(flipped_masks,  dim=0)

        if self.arch == "SatMAE_fMoW_Temporal_ViT-Large":
            logits_mask = self.forward(image, timestamps)
        else:
            logits_mask = self.forward(image)


        loss = self.loss_fn(logits_mask, mask.long())

        prob_mask = logits_mask.softmax(dim=1)
        pred_mask = prob_mask.argmax(dim=1).float()

        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="multiclass", num_classes=self.out_classes
        )

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="macro"
        )

        total_loss = torch.stack([x["loss"] for x in outputs]).mean()

        miou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="macro"
        )
        
        metrics = {
            #f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_mIOU": miou,
            #f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_loss": total_loss,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        # append the metics of each step to the
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        # empty set output list
        self.training_step_outputs.clear()
        return

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()
        return

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        # empty set output list
        self.test_step_outputs.clear()
        return

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.00006, weight_decay=0.01)
        scheduler = lr_scheduler.PolynomialLR(optimizer, total_iters = int(self.n_iters_training), verbose=False) #lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", #or "epoch"
                "frequency": 1,
            },
        }
        return

def calculate_mean_std(dataloader, num_channels, batch_size=32):
    """
    Calculate channel-wise mean and standard deviation for a dataset.
    Args:
        dataset (Dataset): PyTorch dataset (train dataset).
        batch_size (int): Batch size for DataLoader.
    Returns:
        tuple: mean and standard deviation (each as a list of 3 values for RGB).
    """
    mean = torch.zeros(num_channels)
    std = torch.zeros(num_channels)
    n_samples = 0

    for images, _ in dataloader:
        images = images.view(images.size(0), images.size(1), -1)
        mean += images.mean(dim=[0, 2]) * images.size(0)
        std += images.std(dim=[0, 2]) * images.size(0)
        n_samples += images.size(0)

    mean /= n_samples
    std /= n_samples

    return mean.tolist(), std.tolist()

def create_augmentation_transform():
    transform = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5)
    ])

    return transform

def create_normalization_transform(mean, std, scale=False):
    transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=scale),
        v2.Normalize(mean=mean, std=std)
    ])

    return transform

def create_scaling_transform(size):
    transform = v2.Compose([
        v2.Resize(size)
    ])

    return transform