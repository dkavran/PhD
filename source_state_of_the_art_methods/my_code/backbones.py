import os
import torch
import torch.nn as nn
import sys

from .utils import snapshot_weights, report_weight_changes

#GASSL
class MoCoResNet50Backbone(nn.Module):
    """
    A ResNet50 backbone that:
      1) Loads a MoCo checkpoint into the ResNet (freezing all except the final fc).
      2) Deletes avgpool + fc.
      3) Defines a forward(x) that only runs through layer4 and returns that feature map.
    """
    def __init__(self, arch, pretrained_path): #, freeze = False
        super().__init__()

        #imports
        import torchvision.models as models
        sys.path.append("./methods/GASSL/moco_fmow")
        import moco

        self.encoder = models.__dict__[arch]()

        if pretrained_path is not None and os.path.isfile(pretrained_path):
            orig_weights = snapshot_weights(self.encoder)

            print(f"=> loading checkpoint '{pretrained_path}'")
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            state_dict = checkpoint['state_dict']

            new_state = {}
            for k in list(state_dict.keys()):
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove the "module.encoder_q." prefix
                    new_key = k[len("module.encoder_q."):]
                    new_state[new_key] = state_dict[k]
                # whether it matched or not, delete the original key so we don't accidentally reload it:
                del state_dict[k]

            #print(new_state)
            msg = self.encoder.load_state_dict(new_state, strict=False)
            # The only missing keys should be "fc.weight" and "fc.bias"
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
            print(f"=> loaded MoCo‐pretrained weights from '{pretrained_path}'")

            report_weight_changes(self.encoder, orig_weights, atol=1e-6)

        del self.encoder.avgpool
        del self.encoder.fc

        self.out_channels = [4, 64, 256, 512, 1024, 2048]
        self.depth = 5 #out_channels - 1

        weight = self.encoder.conv1.weight.clone() #https://stackoverflow.com/questions/62629114/how-to-modify-resnet-50-with-4-channels-as-input-using-pre-trained-weights-in-py
        self.encoder.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        with torch.no_grad():
            self.encoder.conv1.weight[:, :3] = weight
            self.encoder.conv1.weight[:, 3] = self.encoder.conv1.weight[:, 0]


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute exactly the “stem + layer1 + layer2 + layer3 + layer4” portion of ResNet50,
        and then return the final conv‐feature map of shape (B, 2048, H/32, W/32).
        """
        # === Copy‐paste from torchvision’s ResNet._forward_impl up through layer4 ===
        x_r = self.encoder.conv1(x)    # conv1:     (B,   3,  H,   W) → (B,   64,  H/2,  W/2)
        x_r = self.encoder.bn1(x_r)      # bn1 + relu
        x_r = self.encoder.relu(x_r)
        x_r = self.encoder.maxpool(x_r)  # maxpool:   (B,   64,  H/2,  W/2) → (B,   64,  H/4,  W/4)

        x_1 = self.encoder.layer1(x_r)   # layer1:    (B,   64,  H/4,  W/4) → (B,  256,  H/4,  W/4)
        x_2 = self.encoder.layer2(x_1)   # layer2:    (B,  256,  H/4,  W/4) → (B,  512,  H/8,  W/8)
        x_3 = self.encoder.layer3(x_2)   # layer3:    (B,  512,  H/8,  W/8) → (B, 1024, H/16, W/16)
        x_4 = self.encoder.layer4(x_3)   # layer4:    (B, 1024, H/16, W/16) → (B, 2048, H/32, W/32)

        # That’s it. We do NOT do avgpool or fc. We simply return the final conv‐feature map:
        return [x, x_r, x_1, x_2, x_3, x_4]
    
#SeCo
class SeCo_MoCoResNet50Backbone(nn.Module):
    def __init__(self, arch, pretrained_path): #, freeze = False
        super().__init__()

        #imports
        sys.path.append("./methods/SeCo")
        from copy import deepcopy
        from models.moco2_module import MocoV2

        model = MocoV2.load_from_checkpoint(pretrained_path)
        print("Loaded weights:", pretrained_path)
        del model.encoder_k
        del model.heads_q
        del model.heads_k
        
        self.encoder = model.encoder_q
        
        del self.encoder[9]
        del self.encoder[8]
        
        if arch == "resnet18":
            self.out_channels = [4, 64, 64, 128, 256, 512]
        elif arch == "resnet50":
            self.out_channels = [4, 64, 256, 512, 1024, 2048]

        self.depth = 5 #out_channels - 1

        weight = self.encoder[0].weight.clone() #https://stackoverflow.com/questions/62629114/how-to-modify-resnet-50-with-4-channels-as-input-using-pre-trained-weights-in-py
        self.encoder[0] = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        with torch.no_grad():
            self.encoder[0].weight[:, :3] = weight
            self.encoder[0].weight[:, 3] = self.encoder[0].weight[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        outputs.append(x)
        for i in range(0, len(self.encoder)):
            x = self.encoder[i](x)
            outputs.append(x)

        #x, x_r, x_1, x_2, x_3, x_4
        return [outputs[0], outputs[4], outputs[5], outputs[6], outputs[7], outputs[8]]
    
#SatMAE
class SatMAE(nn.Module):
    def __init__(self, arch, pretrained_path, **kwargs): #, freeze = False
        #imports
        sys.path.append("./methods/SatMAE")
        from util.pos_embed import interpolate_pos_embed

        self.arch = arch

        #rest of code...
        super().__init__()

        if arch == "SatMAE_fMoW_Non_Temporal_ViT-Large":
            from models_vit import vit_large_patch16
            self.encoder = vit_large_patch16(**kwargs)

        elif arch == "SatMAE_fMoW_Temporal_ViT-Large":
            from models_vit_temporal import vit_large_patch16
            self.encoder = vit_large_patch16(**kwargs)
        
        elif arch == "SatMAE_fMoW_MultiSpectral_ViT-Base":
            from models_vit import vit_base_patch16
            self.encoder = vit_base_patch16(**kwargs)

        elif arch == "SatMAE_fMoW_MultiSpectral_ViT-Large":
            from models_vit import vit_large_patch16
            self.encoder = vit_large_patch16(**kwargs)
        
        
        checkpoint = torch.load(pretrained_path, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % pretrained_path)
        checkpoint_model = checkpoint['model']
        state_dict = self.encoder.state_dict()

        for k in ['pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(self.encoder, checkpoint_model)

        # load pre-trained model
        orig_weights = snapshot_weights(self.encoder)
        msg = self.encoder.load_state_dict(checkpoint_model, strict=False)
        report_weight_changes(self.encoder, orig_weights, atol=1e-6)


        del self.encoder.head
        del self.encoder.head_drop
        del self.encoder.fc_norm


        if arch in ["SatMAE_fMoW_Non_Temporal_ViT-Large", "SatMAE_fMoW_Temporal_ViT-Large", "SatMAE_fMoW_MultiSpectral_ViT-Large"]:
            weight = self.encoder.patch_embed.proj.weight.clone() #https://stackoverflow.com/questions/62629114/how-to-modify-resnet-50-with-4-channels-as-input-using-pre-trained-weights-in-py
            self.encoder.patch_embed.proj = nn.Conv2d(4, 1024, kernel_size=(16, 16), stride=(16, 16))
            with torch.no_grad():
                self.encoder.patch_embed.proj.weight[:, :3] = weight
                self.encoder.patch_embed.proj.weight[:, 3] = self.encoder.patch_embed.proj.weight[:, 0]
        elif arch == "SatMAE_fMoW_MultiSpectral_ViT-Base":
            weight = self.encoder.patch_embed.proj.weight.clone()
            self.encoder.patch_embed.proj = nn.Conv2d(4, 768, kernel_size=(16, 16), stride=(16, 16))
            with torch.no_grad():
                self.encoder.patch_embed.proj.weight[:, :3] = weight
                self.encoder.patch_embed.proj.weight[:, 3] = self.encoder.patch_embed.proj.weight[:, 0]
        
        

    def get_config(self):
        sys.path.append("./decoders/TransUNet")
        import networks.vit_seg_configs as seg_configs

        if self.arch == "SatMAE_fMoW_Non_Temporal_ViT-Large":
            return seg_configs.get_l16_config()
        elif self.arch == "SatMAE_fMoW_Temporal_ViT-Large":
            return seg_configs.get_l16_config()
        elif self.arch == "SatMAE_fMoW_MultiSpectral_ViT-Base":
            return seg_configs.get_b16_config()
        elif self.arch == "SatMAE_fMoW_MultiSpectral_ViT-Large":
            return seg_configs.get_l16_config()



    def forward(self, x: torch.Tensor, timestamps: torch.Tensor = None) -> torch.Tensor:
        if timestamps is not None:
            return self.encoder(x, timestamps)
        else:
            return self.encoder(x)
        
#TOV
class TOV(nn.Module):
    def __init__(self, arch, pretrained_path, **kwargs): #, freeze = False
        super().__init__()
        
        self.arch = arch

        #imports
        from argparse import ArgumentParser
        sys.path.append("./methods/TOV/G-RSIM/TOV_v1/segmentation")
        sys.path.append("./methods/TOV/G-RSIM/TOV_v1/classification")
        from models import build_model
        from utils import load_ckpt

        from argparse import Namespace

        args = Namespace(
            model_name="1012300002",
            pretrained=True,
            in_channel=3,
            load_pretrained=True,
            mode_name='finetune',
            map_keys={}
        )

        self.encoder = build_model(**vars(args))

        self.out_channels = [4, 64, 256, 512, 1024, 2048] #uporablja se ResNet50 za backbone
        self.depth = 5

        #load pretrained
        if args.load_pretrained:
            bl_layers = None
            if args.mode_name in ['train', 'finetune']:
                bl_layers = ['classifier', 'fc']

            orig_weights = snapshot_weights(self.encoder.features)

            self.encoder = load_ckpt(self.encoder.features, pretrained_path,
                            train=(args.mode_name == 'train'),
                            block_layers=bl_layers,
                            map_keys=args.map_keys,
                            verbose=True)
            
            report_weight_changes(self.encoder, orig_weights, atol=1e-6)
            
            weight = self.encoder.conv1.weight.clone()
            self.encoder.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            with torch.no_grad():
                self.encoder.conv1.weight[:, :3] = weight
                self.encoder.conv1.weight[:, 3] = self.encoder.conv1.weight[:, 0]

            self.encoder.return_layers = { #https://github.com/pytorch/vision/blob/main/torchvision/models/_utils.py
                'maxpool': 'out1',
                'layer1': 'out2',
                'layer2': 'out3',
                'layer3': 'out4',
                'layer4': 'out5',
            }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.encoder(x)
        return [x, output['out1'], output['out2'], output['out3'], output['out4'], output['out5']]