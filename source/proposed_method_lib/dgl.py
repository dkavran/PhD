import numpy as np
import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv, GATConv, GraphormerLayer, HGTConv
from dgl import function as fn
from dgl.dataloading import Sampler
from torchvision.models import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l, vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14, maxvit_t, resnet18, resnet50, resnet101, resnet152, convnext_tiny, convnext_small, convnext_base, convnext_large, shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x2_0
import torchvision
from dgl.dataloading import MultiLayerFullNeighborSampler, NeighborSampler, to_block
import matplotlib.pyplot as plt
from torchvision.transforms import v2

# Implemented basic feature extractor CNNs
class BasicFeatureExtractorCNN1(nn.Module):
    def __init__(self, in_channels):
        super(BasicFeatureExtractorCNN1, self).__init__()
        
        # Convolutional layers with BatchNorm and ELU
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Global average pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1) # Output size is (batch_size, num_features, 1, 1)

    def forward(self, x):
        # Convolutional layers with BatchNorm and ELU
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.conv2(x)))
        x = F.elu(self.bn3(self.conv3(x)))
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1) # Flatten to (batch_size, num_features)
        
        return x
    
class BasicFeatureExtractorCNN2(nn.Module):
    def __init__(self, in_channels):
        super(BasicFeatureExtractorCNN2, self).__init__()
        
        # Convolutional layers with BatchNorm and ELU
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Global average pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1) # Output size is (batch_size, num_features, 1, 1)

    def forward(self, x):
        # Convolutional layers with BatchNorm and ELU
        x = F.elu(self.bn1(self.conv1(x)))
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1) # Flatten to (batch_size, num_features)
        
        return x

def create_feature_extractor(feature_extractor_size, freeze_upper_N_layers, feat_dropout, in_channels=3):
    weights = None
    feature_extractor = None
    feature_extractor_transform = None
    feature_extractor_output_size = None

    # Efficient-Nets
    if feature_extractor_size == "EfficientNetV2-S":
        weights = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        if freeze_upper_N_layers is None:
            print("No pretrained weights used!")
            weights = None
        feature_extractor = efficientnet_v2_s(weights=weights, dropout=feat_dropout)
        feature_extractor_transform = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()
        feature_extractor_output_size = 1280
    elif feature_extractor_size == "EfficientNetV2-M":
        weights = torchvision.models.EfficientNet_V2_M_Weights.IMAGENET1K_V1
        if freeze_upper_N_layers is None:
            print("No pretrained weights used!")
            weights = None
        feature_extractor = efficientnet_v2_m(weights=weights, dropout=feat_dropout)
        feature_extractor_transform = torchvision.models.EfficientNet_V2_M_Weights.IMAGENET1K_V1.transforms()
        feature_extractor_output_size = 1280
    elif feature_extractor_size == "EfficientNetV2-L":
        weights = torchvision.models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
        if freeze_upper_N_layers is None:
            print("No pretrained weights used!")
            weights = None
        feature_extractor = efficientnet_v2_l(weights=weights, dropout=feat_dropout)
        feature_extractor_transform = torchvision.models.EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms()
        feature_extractor_output_size = 1280
        
    # ViTs
    elif feature_extractor_size == "VisionTransformer-Base-16":
        weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
        if freeze_upper_N_layers is None:
            print("No pretrained weights used!")
            weights = None
        feature_extractor = vit_b_16(weights=weights, dropout=feat_dropout)
        feature_extractor_transform = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1.transforms()
        feature_extractor_output_size = 768
    elif feature_extractor_size == "VisionTransformer-Base-32":
        weights = torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1
        if freeze_upper_N_layers is None:
            print("No pretrained weights used!")
            weights = None
        feature_extractor = vit_b_32(weights=weights, dropout=feat_dropout)
        feature_extractor_transform = torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1.transforms()
        feature_extractor_output_size = 768
    elif feature_extractor_size == "VisionTransformer-Large-16":
        weights = torchvision.models.ViT_L_16_Weights.IMAGENET1K_V1
        if freeze_upper_N_layers is None:
            print("No pretrained weights used!")
            weights = None
        feature_extractor = vit_l_16(weights=weights, dropout=feat_dropout)
        feature_extractor_transform = torchvision.models.ViT_L_16_Weights.IMAGENET1K_V1.transforms()
        feature_extractor_output_size = 1024
    elif feature_extractor_size == "VisionTransformer-Large-32":
        weights = torchvision.models.ViT_L_32_Weights.IMAGENET1K_V1
        if freeze_upper_N_layers is None:
            print("No pretrained weights used!")
            weights = None
        feature_extractor = vit_l_32(weights=weights, dropout=feat_dropout)
        feature_extractor_transform = torchvision.models.ViT_L_32_Weights.IMAGENET1K_V1.transforms()
        feature_extractor_output_size = 1024
    elif feature_extractor_size == "VisionTransformer-Huge-14":
        weights = torchvision.models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1
        if freeze_upper_N_layers is None:
            print("No pretrained weights used!")
            weights = None
        feature_extractor = vit_h_14(weights=weights, dropout=feat_dropout)
        feature_extractor_transform = torchvision.models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1.transforms()
        feature_extractor_output_size = 1280
    elif feature_extractor_size == "MaxVit":
        weights = torchvision.models.MaxVit_T_Weights.IMAGENET1K_V1
        if freeze_upper_N_layers is None:
            print("No pretrained weights used!")
            weights = None
        feature_extractor = maxvit_t(weights=weights)
        feature_extractor_transform = torchvision.models.MaxVit_T_Weights.IMAGENET1K_V1.transforms()
        feature_extractor_output_size = 512

    # ResNets
    elif feature_extractor_size == "ResNet-18":
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        if freeze_upper_N_layers is None:
            print("No pretrained weights used!")
            weights = None
        feature_extractor = resnet18(weights=weights)
        feature_extractor_transform = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
        feature_extractor_output_size = 512
    elif feature_extractor_size == "ResNet-50":
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        if freeze_upper_N_layers is None:
            print("No pretrained weights used!")
            weights = None
        feature_extractor = resnet50(weights=weights)
        feature_extractor_transform = torchvision.models.ResNet50_Weights.IMAGENET1K_V1.transforms()
        feature_extractor_output_size = 2048
    elif feature_extractor_size == "ResNet-101":
        weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V1
        if freeze_upper_N_layers is None:
            print("No pretrained weights used!")
            weights = None
        feature_extractor = resnet101(weights=weights)
        feature_extractor_transform = torchvision.models.ResNet101_Weights.IMAGENET1K_V1.transforms()
        feature_extractor_output_size = 2048
    elif feature_extractor_size == "ResNet-152":
        weights = torchvision.models.ResNet152_Weights.IMAGENET1K_V1
        if freeze_upper_N_layers is None:
            print("No pretrained weights used!")
            weights = None
        feature_extractor = resnet152(weights=weights)
        feature_extractor_transform = torchvision.models.ResNet152_Weights.IMAGENET1K_V1.transforms()
        feature_extractor_output_size = 2048

    # ConvNeXts
    elif feature_extractor_size == "ConvNeXt-tiny":
        weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        if freeze_upper_N_layers is None:
            print("No pretrained weights used!")
            weights = None
        feature_extractor = convnext_tiny(weights=weights)
        feature_extractor_transform = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()
        feature_extractor_output_size = 768
    elif feature_extractor_size == "ConvNeXt-small":
        weights = torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1
        if freeze_upper_N_layers is None:
            print("No pretrained weights used!")
            weights = None
        feature_extractor = convnext_small(weights=weights)
        feature_extractor_transform = torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1.transforms()
        feature_extractor_output_size = 768
    elif feature_extractor_size == "ConvNeXt-base":
        weights = torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1
        if freeze_upper_N_layers is None:
            print("No pretrained weights used!")
            weights = None
        feature_extractor = convnext_base(weights=weights)
        feature_extractor_transform = torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1.transforms()
        feature_extractor_output_size = 1024
    elif feature_extractor_size == "ConvNeXt-large":
        weights = torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1
        if freeze_upper_N_layers is None:
            print("No pretrained weights used!")
            weights = None
        feature_extractor = convnext_large(weights=weights)
        feature_extractor_transform = torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1.transforms()
        feature_extractor_output_size = 1536
    
    # Other efficient small networks
    elif feature_extractor_size == "ShuffleNetV2-x0.5":
        weights = torchvision.models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1
        if freeze_upper_N_layers is None:
            print("No pretrained weights used!")
            weights = None
        feature_extractor = shufflenet_v2_x0_5(weights=weights)
        feature_extractor_transform = torchvision.models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1.transforms()
        feature_extractor_output_size = 1024
    elif feature_extractor_size == "ShuffleNetV2-x1.0":
        weights = torchvision.models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
        if freeze_upper_N_layers is None:
            print("No pretrained weights used!")
            weights = None
        feature_extractor = shufflenet_v2_x1_0(weights=weights)
        feature_extractor_transform = torchvision.models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1.transforms()
        feature_extractor_output_size = 1024
    elif feature_extractor_size == "ShuffleNetV2-x2.0":
        weights = torchvision.models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1
        if freeze_upper_N_layers is None:
            print("No pretrained weights used!")
            weights = None
        feature_extractor = shufflenet_v2_x2_0(weights=weights)
        feature_extractor_transform = torchvision.models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1.transforms()
        feature_extractor_output_size = 2048
    elif feature_extractor_size == "BasicFE1":
        feature_extractor = BasicFeatureExtractorCNN1(in_channels=in_channels)
        feature_extractor_transform = None
        feature_extractor_output_size = 128
    elif feature_extractor_size == "BasicFE2":
        feature_extractor = BasicFeatureExtractorCNN2(in_channels=in_channels)
        feature_extractor_transform = None
        feature_extractor_output_size = 32
    else:
        raise Exception("Wrong feature extractor (EfficientNetV2) size")
    
    return feature_extractor, feature_extractor_transform, feature_extractor_output_size

NID = dgl.NID
EID = dgl.EID

class ProposedSampler(NeighborSampler):
    def __init__(self, fanouts, edge_sequence_stages, gat_nn=False, **kwargs):
        super(ProposedSampler, self).__init__(fanouts, **kwargs)
        
        self.edge_sequence = []
        for edge_type, num_layers in edge_sequence_stages:
            self.edge_sequence.extend([edge_type] * num_layers)
        
        self.gat_nn = gat_nn
        print("Is GAT used (edge switching sampler received info):", self.gat_nn)
                
        if len(self.fanouts) != len(self.edge_sequence):
            raise ValueError("Total layers in edge_sequence_stages should match the length of fanouts")

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []

        previous_frontier = None
        gat_frontier = None

        for i, (fanout, edge_type) in enumerate(zip(self.fanouts, self.edge_sequence)):
            if self.gat_nn is True:
                mask = (g.edata['type'] == 2) # Only get edges, which are self-loops
                mask = ~mask # Reverse ====> the edges, which are not self-loops, will be skipped
                edges_to_select = torch.nonzero(mask, as_tuple=True)[0]
                edges_to_select = edges_to_select.int()
                exclude_eids = edges_to_select

                gat_frontier = g.sample_neighbors( # Sample only self loops
                    seed_nodes,
                    -1,
                    edge_dir=self.edge_dir,
                    prob=self.prob,
                    replace=self.replace,
                    exclude_edges=exclude_eids
                )
            
            mask = (g.edata['type'] != edge_type)
            edges_to_select = torch.nonzero(mask, as_tuple=True)[0]
            edges_to_select = edges_to_select.int()
            exclude_eids = edges_to_select

            frontier = g.sample_neighbors(
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                exclude_edges=exclude_eids
            )

            if self.gat_nn is True:
                frontier = dgl.merge([gat_frontier, frontier])
                

            if previous_frontier is not None:
                # Get edges from previous frontier
                previous_frontier_edges = previous_frontier.edges()
                
                # Get indices of non present edges in current frontier, that are present in previous one
                are_edges_already_present = frontier.has_edges_between(previous_frontier_edges[0], previous_frontier_edges[1])
                indices_of_non_present_edges = torch.nonzero(~are_edges_already_present, as_tuple=True)[0]

                # Keep data/features only from edges that are in previous, but not in current frontier
                indices_of_present_edges = torch.nonzero(are_edges_already_present, as_tuple=True)[0]
                previous_frontier.remove_edges(indices_of_present_edges.int())

                # Get data from previous frontier (which is now cleaned, without data from duplicated edges)
                previous_frontier_edata = previous_frontier.edata
                previous_frontier_edata.pop('_ID', None)

                # Add edges, which are non duplicate (solved with [indices_of_non_present_edges]) from previous frontier along with the representative data
                frontier.add_edges(previous_frontier_edges[0][indices_of_non_present_edges],
                                    previous_frontier_edges[1][indices_of_non_present_edges],
                                    data=previous_frontier_edata)

            previous_frontier = frontier

            eid = frontier.edata[EID]
            block = to_block(frontier, seed_nodes)
            block.edata[EID] = eid
            seed_nodes = block.srcdata[NID]
            blocks.insert(0, block)

        return seed_nodes, output_nodes, blocks

def get_label_conversion_table(dgl_labels):
    table = {}
    
    unique_labels = np.unique(dgl_labels)
    correct_labels = np.arange(0, unique_labels.shape[0], 1)
    
    for ul_index, unique_label in enumerate(unique_labels):
        table[unique_label] = correct_labels[ul_index]
    
    return table
    

def get_correct_labels(dgl_labels, table):
    copy_dgl_labels = np.zeros(dgl_labels.shape, dtype=dgl_labels.dtype)
    unique_labels = np.unique(dgl_labels)
    
    for unique_label in unique_labels:
        copy_dgl_labels[dgl_labels == unique_label] = table[unique_label]
    
    return copy_dgl_labels

def change_last_layer_of_first_feature_extractor(feature_extractor, feature_extractor_name):
    if feature_extractor_name == "EfficientNetV2-S" or feature_extractor_name == "EfficientNetV2-M" or feature_extractor_name == "EfficientNetV2-L":
        feature_extractor.classifier[1] = torch.nn.Identity()
    elif feature_extractor_name == "VisionTransformer-Base-16" or feature_extractor_name == "VisionTransformer-Base-32" or feature_extractor_name == "VisionTransformer-Large-16" or feature_extractor_name == "VisionTransformer-Large-32" or feature_extractor_name == "VisionTransformer-Huge-14":
        feature_extractor.heads.head = torch.nn.Identity()
    elif feature_extractor_name == "MaxVit":
        feature_extractor.classifier[5] = torch.nn.Identity()
    elif feature_extractor_name == "ResNet-18" or feature_extractor_name == "ResNet-50" or feature_extractor_name == "ResNet-101" or feature_extractor_name == "ResNet-152":
        feature_extractor.fc = torch.nn.Identity()
    elif feature_extractor_name == "ConvNeXt-tiny" or feature_extractor_name == "ConvNeXt-small" or feature_extractor_name == "ConvNeXt-base" or feature_extractor_name == "ConvNeXt-large":
        feature_extractor.classifier[2] = torch.nn.Identity()
    elif feature_extractor_name == "ShuffleNetV2-x0.5" or feature_extractor_name == "ShuffleNetV2-x1.0" or feature_extractor_name == "ShuffleNetV2-x2.0":
        feature_extractor.fc = torch.nn.Identity()
    elif feature_extractor_name == "BasicFE1" or feature_extractor_name == "BasicFE2": # Basic feature extractors to use
        return feature_extractor
    else:
        raise Exception("Feature extractor is not correct when changing last layer:", feature_extractor_name)
    
    return feature_extractor

def feature_extractor_layer_freezing(feature_extractor, feature_extractor_size, freeze_upper_N_layers, max_freeze_percentage = 90.0):
    print("Performing freezing of layers....")
    total_trainable_params = sum(p.numel() for p in feature_extractor.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_trainable_params}")
    sum_trainable_parameters_till_nth_layer = 0

    ct = 0 # Freezing of layers
    if freeze_upper_N_layers is not None and freeze_upper_N_layers > 0:
        if feature_extractor_size == "EfficientNetV2-S" or feature_extractor_size == "EfficientNetV2-M" or feature_extractor_size == "EfficientNetV2-L" or feature_extractor_size == "ConvNeXt-tiny" or feature_extractor_size == "ConvNeXt-small" or feature_extractor_size == "ConvNeXt-base" or feature_extractor_size == "ConvNeXt-large":
            print("Freezing", freeze_upper_N_layers, "stages...", "out of", len(list(feature_extractor.features.children())))
            for child in feature_extractor.features.children():
                if ct < freeze_upper_N_layers: # To freeze upper N layers
                    for param in child.parameters():
                        if ((sum_trainable_parameters_till_nth_layer / total_trainable_params) * 100.0) >= max_freeze_percentage:
                            print("Exceeded maximum number of frozable parameters. Breaking...")
                            break
                        param.requires_grad = False
                        sum_trainable_parameters_till_nth_layer += param.numel()
                ct += 1
                print("Froze layer #", ct)
                if ((sum_trainable_parameters_till_nth_layer / total_trainable_params) * 100.0) >= max_freeze_percentage:
                    print("Done freezing, because exceeded the max limit of:", max_freeze_percentage, "%", "of frozen parameters!")
                    break
                if ct >= freeze_upper_N_layers:
                    print("Done freezing")
                    break

        elif feature_extractor_size == "VisionTransformer-Base-16" or feature_extractor_size == "VisionTransformer-Base-32" or feature_extractor_size == "VisionTransformer-Large-16" or feature_extractor_size == "VisionTransformer-Large-32" or feature_extractor_size == "VisionTransformer-Huge-14":
            print("Freezing", freeze_upper_N_layers, "stages...", "out of", len(list(feature_extractor.encoder.layers.children())))
            for child in feature_extractor.encoder.layers.children():
                if ct < freeze_upper_N_layers: # To freeze upper N layers
                    for param in child.parameters():
                        if ((sum_trainable_parameters_till_nth_layer / total_trainable_params) * 100.0) >= max_freeze_percentage:
                            print("Exceeded maximum number of frozable parameters. Breaking...")
                            break
                        param.requires_grad = False
                        sum_trainable_parameters_till_nth_layer += param.numel()
                ct += 1
                print("Froze layer #", ct)
                if ((sum_trainable_parameters_till_nth_layer / total_trainable_params) * 100.0) >= max_freeze_percentage:
                    print("Done freezing, because exceeded the max limit of:", max_freeze_percentage, "%", "of frozen parameters!")
                    break
                if ct >= freeze_upper_N_layers:
                    print("Done freezing")
                    break
        elif feature_extractor_size == "MaxVit":
            print("Freezing", freeze_upper_N_layers, "stages...", "out of", len(list(feature_extractor.blocks.children())))
            for child in feature_extractor.blocks.children():
                if ct < freeze_upper_N_layers: # To freeze upper N layers
                    for param in child.parameters():
                        if ((sum_trainable_parameters_till_nth_layer / total_trainable_params) * 100.0) >= max_freeze_percentage:
                            print("Exceeded maximum number of frozable parameters. Breaking...")
                            break

                        param.requires_grad = False
                        sum_trainable_parameters_till_nth_layer += param.numel()
                ct += 1
                print("Froze layer #", ct)
                if ((sum_trainable_parameters_till_nth_layer / total_trainable_params) * 100.0) >= max_freeze_percentage:
                    print("Done freezing, because exceeded the max limit of:", max_freeze_percentage, "%", "of frozen parameters!")
                    break
                if ct >= freeze_upper_N_layers:
                    print("Done freezing")
                    break
        elif feature_extractor_size == "ResNet-18" or feature_extractor_size == "ResNet-50" or feature_extractor_size == "ResNet-101" or feature_extractor_size == "ResNet-152" or feature_extractor_size == "ShuffleNetV2-x0.5" or feature_extractor_size == "ShuffleNetV2-x1.0" or feature_extractor_size == "ShuffleNetV2-x2.0" or feature_extractor_size == "MyFE1" or feature_extractor_size == "MyFE2":
            print("Freezing", freeze_upper_N_layers, "stages...", "out of", len(list(feature_extractor.children())))
            for child in feature_extractor.children():
                if ct < freeze_upper_N_layers: #freezer upper N layers
                    for param in child.parameters():
                        if ((sum_trainable_parameters_till_nth_layer / total_trainable_params) * 100.0) >= max_freeze_percentage:
                            print("Exceeded maximum number of frozable parameters. Breaking...")
                            break
                        param.requires_grad = False
                        sum_trainable_parameters_till_nth_layer += param.numel()
                ct += 1
                print("Froze layer #", ct)
                if ((sum_trainable_parameters_till_nth_layer / total_trainable_params) * 100.0) >= max_freeze_percentage:
                    print("Done freezing, because exceeded the max limit of:", max_freeze_percentage, "%", "of frozen parameters!")
                    break
                if ct >= freeze_upper_N_layers:
                    print("Done freezing")
                    break

            print("Done freezing")
        else:
            raise Exception("Incorrect feature_extractor_size passed into feature_extractor_layer_freezing(...):", feature_extractor_size)
    else:
        print("No freezing of layers...")

    percentage_trained = (sum_trainable_parameters_till_nth_layer / total_trainable_params) * 100.0
    print("Froze", sum_trainable_parameters_till_nth_layer, "/", total_trainable_params, "parameters. That is:", percentage_trained, "%")
    
    return feature_extractor

def create_augmentation_transform():
    transform = v2.Compose([
        #v2.RandomRotation(degrees=(0, 359)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5)
    ])

    return transform

def create_normalization_transform(mean, std):
    transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=False),
        v2.Normalize(mean=mean, std=std)
    ])

    return transform


class MyModelGAT(nn.Module):
    def __init__(self, h_feats: list,
                 num_classes,
                 image_size=(32, 32, 3),
                 feature_extractor_size="EfficientNetV2-S",
                 feat_dropout=0.5,
                 n_heads=8,
                 freeze_upper_N_layers = 6,
                 use_ngnn_modules = False,
                 max_freeze_percentage = 90.0,
                 train_channel_means = [0.0, 0.0, 0.0, 0.0],
                 train_channel_stds = [1.0, 1.0, 1.0, 1.0],
                 use_ELU_in_graph_layers = False,
                 aggregate_node_classifications = False):
        super(MyModelGAT, self).__init__()

        print("Created GAT!")

        print("Will perform normalization of raw data with: means:", train_channel_means, "and stds:", train_channel_stds)
        self.normalization_transform = create_normalization_transform(train_channel_means[:image_size[2]], train_channel_stds[:image_size[2]])
        self.augmentation_transform = create_augmentation_transform()

        self.image_size = image_size
        self.n_heads = n_heads
        self.freeze_upper_N_layers = freeze_upper_N_layers
        self.use_ngnn_modules = use_ngnn_modules
        self.use_ELU_in_graph_layers = use_ELU_in_graph_layers
        self.aggregate_node_classifications = aggregate_node_classifications

        if self.use_ELU_in_graph_layers is True:
            print("Using ELU activation in Graph Layers")

        if self.aggregate_node_classifications is True:
            print("Aggregating node classifications!")
        
        if self.image_size[2] != 3:
            self.conv_change_feature_channels_to_3 = nn.Conv2d(in_channels=self.image_size[2], out_channels=3, kernel_size=1, stride=1)
        else:
            print("Image depth is 3, so additional CONV layer is not needed! Just nn.Identity() used!")
            self.conv_change_feature_channels_to_3 = nn.Identity()

        self.feature_extractor, self.feature_extractor_transform, self.feature_extractor_output_size = create_feature_extractor(feature_extractor_size, self.freeze_upper_N_layers, feat_dropout, in_channels=self.image_size[2])

        if feature_extractor_size == "EfficientNetV2-S" or feature_extractor_size == "EfficientNetV2-M" or feature_extractor_size == "EfficientNetV2-L":
            self.feature_extractor_transform.resize_size = [self.image_size[0]]
            self.feature_extractor_transform.crop_size = [self.image_size[0]]

        self.feature_extractor = change_last_layer_of_first_feature_extractor(self.feature_extractor, feature_extractor_size) # Remove the classifier/final layer
        if self.aggregate_node_classifications is True:
            self.fc = nn.Linear(self.feature_extractor_output_size, num_classes)
            self.feature_extractor_output_size = num_classes

        print("Number of all parameters in feature extractor NN:", sum(p.numel() for p in self.feature_extractor.parameters()))
        self.feature_extractor = feature_extractor_layer_freezing(self.feature_extractor, feature_extractor_size, self.freeze_upper_N_layers, max_freeze_percentage)
        print("Number of TRAINABLE parameters in feature extractor NN:", sum(p.numel() for p in self.feature_extractor.parameters() if p.requires_grad))

        print("Feature extractor has output of shape:", self.feature_extractor_output_size)

        self.ngnns = []
        if len(h_feats) == 0:
            first_conv = GATConv(self.feature_extractor_output_size, num_classes, num_heads=1)
        else:
            activation = None
            if self.use_ELU_in_graph_layers is True:
                activation = nn.ELU()
            first_conv = GATConv(self.feature_extractor_output_size, h_feats[0], num_heads=n_heads, feat_drop=feat_dropout, activation=activation)
            if self.use_ngnn_modules is True:
                print("Using NGNN modules...")
                self.ngnns.append(nn.Linear(h_feats[0], h_feats[0]))
            else:
                print("Not using NGNN modules...")
        
        self.convs = [first_conv]
        for i in range(1, len(h_feats)):
            activation = None
            if self.use_ELU_in_graph_layers is True:
                activation = nn.ELU()
            self.convs.append(GATConv(h_feats[i-1] * n_heads, h_feats[i], num_heads=n_heads, feat_drop=feat_dropout, activation=activation))
            if self.use_ngnn_modules is True:
                self.ngnns.append(nn.Linear(h_feats[i], h_feats[i]))
            
        if len(h_feats) >= 1:
            self.convs.append(GATConv(h_feats[-1] * n_heads, num_classes, num_heads=1))
            if self.use_ngnn_modules is True:
                self.ngnns.append(nn.Linear(num_classes, num_classes))
        
        print("Number of Graph layers:", len(self.convs))
        self.convs = nn.ModuleList(self.convs)
        if len(self.ngnns) > 0:
            self.ngnns = nn.ModuleList(self.ngnns)

        print("Number of all parameters:", sum(p.numel() for p in self.parameters()), "/ Number of learnable parameters:", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, mfgs, x):        
        x = torch.reshape(x, (x.shape[0], self.image_size[0], self.image_size[1], -1)).permute(0,3,1,2)

        x = self.normalization_transform(x)
        x = self.conv_change_feature_channels_to_3(x)
        if self.feature_extractor_transform is not None:
            x = self.feature_extractor_transform(x)

        if self.training:
            x = self.augmentation_transform(x)

        x = self.feature_extractor(x)
        if self.aggregate_node_classifications is True:
            x = self.fc(x)
        x = x.reshape(-1, self.feature_extractor_output_size)

        data = x
        for i in range(0, len(self.convs)):
            h_dst = data[:mfgs[i].num_dst_nodes()]
            h = self.convs[i](mfgs[i], (data, h_dst))

            h = h.view(-1, h.size(1) * h.size(2)) # (in_feat, num_heads, out_dim) -> (in_feat, num_heads * out_dim) (Mentioned in: https://github.com/dmlc/dgl/issues/1887)

            if i < (len(self.convs) - 1):

                if len(self.ngnns) > 0:
                    print("Applying NGNNS....", i)
                    h = self.ngnns[i](h)
        
            data = h

        return data

class MyModelGraphTransformer(nn.Module):
    def __init__(self, h_feats: list,
                 num_classes,
                 image_size=(32, 32, 3),
                 feature_extractor_size="EfficientNetV2-S",
                 num_heads=8,
                 feat_dropout=0.5,
                 use_edge_weights=False,
                 freeze_upper_N_layers = 6,
                 use_ngnn_modules = False,
                 max_freeze_percentage = 90.0,
                 train_channel_means = [0.0, 0.0, 0.0, 0.0],
                 train_channel_stds = [1.0, 1.0, 1.0, 1.0],
                 use_normalization = False,
                 aggregate_node_classifications = False):
        super(MyModelGraphTransformer, self).__init__()

        print("Created GraphTransformer!")

        print("Will perform normalization of raw data with: means:", train_channel_means, "and stds:", train_channel_stds)
        self.normalization_transform = create_normalization_transform(train_channel_means[:image_size[2]], train_channel_stds[:image_size[2]])
        self.augmentation_transform = create_augmentation_transform()

        self.image_size = image_size
        self.use_edge_weights = use_edge_weights
        self.freeze_upper_N_layers = freeze_upper_N_layers
        self.use_ngnn_modules = use_ngnn_modules
        self.use_normalization = use_normalization
        self.aggregate_node_classifications = aggregate_node_classifications

        if self.use_normalization is True:
            print("Using layer normalization")

        if self.aggregate_node_classifications is True:
            print("Aggregating node classifications!")
        
        if self.image_size[2] != 3:
            self.conv_change_feature_channels_to_3 = nn.Conv2d(in_channels=self.image_size[2], out_channels=3, kernel_size=1, stride=1)
        else:
            print("Image depth is 3, so additional CONV layer is not needed! Just nn.Identity() used!")
            self.conv_change_feature_channels_to_3 = nn.Identity()
        
        self.feature_extractor, self.feature_extractor_transform, self.feature_extractor_output_size = create_feature_extractor(feature_extractor_size, self.freeze_upper_N_layers, feat_dropout, in_channels=self.image_size[2])

        if feature_extractor_size == "EfficientNetV2-S" or feature_extractor_size == "EfficientNetV2-M" or feature_extractor_size == "EfficientNetV2-L":
            self.feature_extractor_transform.resize_size = [self.image_size[0]]
            self.feature_extractor_transform.crop_size = [self.image_size[0]]

        self.feature_extractor = change_last_layer_of_first_feature_extractor(self.feature_extractor, feature_extractor_size) #Remove the classifier/final layer
        if self.aggregate_node_classifications is True:
            self.fc = nn.Linear(self.feature_extractor_output_size, num_classes)
            self.feature_extractor_output_size = num_classes

        print("Number of all parameters in feature extractor NN:", sum(p.numel() for p in self.feature_extractor.parameters()))
        self.feature_extractor = feature_extractor_layer_freezing(self.feature_extractor, feature_extractor_size, self.freeze_upper_N_layers, max_freeze_percentage)
        print("Number of TRAINABLE parameters in feature extractor NN:", sum(p.numel() for p in self.feature_extractor.parameters() if p.requires_grad))

        print("Input h_feats are:", h_feats)
        print("Feature extractor has output of shape:", self.feature_extractor_output_size)

        self.ngnns = []
        if len(h_feats) == 0:
            first_conv = HGTConv(self.feature_extractor_output_size, head_size=num_classes, num_heads=1, num_ntypes=1, num_etypes=2, dropout=0.0)
        else:
            first_conv = HGTConv(self.feature_extractor_output_size, head_size=h_feats[0], num_heads=num_heads, num_ntypes=1, num_etypes=2, dropout=feat_dropout, use_norm=self.use_normalization)
            if self.use_ngnn_modules is True:
                print("Using NGNN modules...")
                self.ngnns.append(nn.Linear(h_feats[0] * num_heads, h_feats[0] * num_heads))
            else:
                print("Not using NGNN modules...")
        
        self.convs = [first_conv]
        for i in range(1, len(h_feats)):
            self.convs.append(HGTConv(h_feats[i-1] * num_heads, head_size=h_feats[i], num_heads=num_heads, num_ntypes=1, num_etypes=2, dropout=feat_dropout, use_norm=self.use_normalization))
            if self.use_ngnn_modules is True:
                self.ngnns.append(nn.Linear(h_feats[i] * num_heads, h_feats[i] * num_heads))
            
        if len(h_feats) >= 1:
            self.convs.append(HGTConv(h_feats[-1] * num_heads, head_size=num_classes, num_heads=1, num_ntypes=1, num_etypes=2, dropout=0.0))
            if self.use_ngnn_modules is True:
                self.ngnns.append(nn.Linear(num_classes, num_classes))
        
        print("Number of Graph layers:", len(self.convs))
        self.convs = nn.ModuleList(self.convs)
        if len(self.ngnns) > 0:
            self.ngnns = nn.ModuleList(self.ngnns)

        print("Number of all parameters:", sum(p.numel() for p in self.parameters()), "/ Number of learnable parameters:", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, mfgs, x):
        x = torch.reshape(x, (x.shape[0], self.image_size[0], self.image_size[1], -1)).permute(0,3,1,2)
        
        x = self.normalization_transform(x)
        x = self.conv_change_feature_channels_to_3(x)
        if self.feature_extractor_transform is not None:
            x = self.feature_extractor_transform(x)

        if self.training:
            x = self.augmentation_transform(x)

        x = self.feature_extractor(x)
        if self.aggregate_node_classifications is True:
            x = self.fc(x)
        x = x.reshape(-1, self.feature_extractor_output_size)

        data = x
        for i in range(0, len(self.convs)):
            h_dst = data[:mfgs[i].num_src_nodes()]

            etype = mfgs[i].edata["type"]
            ntype = torch.zeros(mfgs[i].num_src_nodes(), dtype=torch.long, device=x.device)

            h = self.convs[i](mfgs[i], h_dst, ntype=ntype, etype=etype)
            
            if i < (len(self.convs) - 1):
                h = F.elu(h)

                if len(self.ngnns) > 0:
                    print("Applying NGNNS....", i)
                    h = self.ngnns[i](h)
        
            data = h

        return data


class MyModelGraphSAGE(nn.Module):
    def __init__(self, h_feats: list,
                 num_classes,
                 image_size=(32, 32, 3),
                 feature_extractor_size="EfficientNetV2-S",
                 aggregator_type="lstm",
                 feat_dropout=0.5,
                 use_edge_weights=False,
                 freeze_upper_N_layers = 6,
                 use_ngnn_modules = False,
                 max_freeze_percentage = 90.0,
                 train_channel_means = [0.0, 0.0, 0.0, 0.0],
                 train_channel_stds = [1.0, 1.0, 1.0, 1.0],
                 use_normalization = False,
                 use_ELU_in_graph_layers = False,
                 aggregate_node_classifications = False):
        super(MyModelGraphSAGE, self).__init__()

        print("Created GraphSAGE!")
        
        print("Will perform normalization of raw data with: means:", train_channel_means, "and stds:", train_channel_stds)
        self.normalization_transform = create_normalization_transform(train_channel_means[:image_size[2]], train_channel_stds[:image_size[2]])
        self.augmentation_transform = create_augmentation_transform()

        self.image_size = image_size
        self.use_edge_weights = use_edge_weights
        self.freeze_upper_N_layers = freeze_upper_N_layers
        self.use_ngnn_modules = use_ngnn_modules
        self.use_normalization = use_normalization
        self.use_ELU_in_graph_layers = use_ELU_in_graph_layers
        self.aggregate_node_classifications = aggregate_node_classifications

        if self.use_ELU_in_graph_layers is True:
            print("Using ELU activation in Graph Layers")

        if self.aggregate_node_classifications is True:
            print("Aggregating node classifications!")
        
        if self.use_normalization is True:
            print("Using batch normalization 1d")
        
        if self.image_size[2] != 3:
            self.conv_change_feature_channels_to_3 = nn.Conv2d(in_channels=self.image_size[2], out_channels=3, kernel_size=1, stride=1)
        else:
            print("Image depth is 3, so additional CONV layer is not needed! Just nn.Identity() used!")
            self.conv_change_feature_channels_to_3 = nn.Identity()
        
        self.feature_extractor, self.feature_extractor_transform, self.feature_extractor_output_size = create_feature_extractor(feature_extractor_size, self.freeze_upper_N_layers, feat_dropout, in_channels=self.image_size[2])

        if feature_extractor_size == "EfficientNetV2-S" or feature_extractor_size == "EfficientNetV2-M" or feature_extractor_size == "EfficientNetV2-L":
            self.feature_extractor_transform.resize_size = [self.image_size[0]]
            self.feature_extractor_transform.crop_size = [self.image_size[0]]

        self.feature_extractor = change_last_layer_of_first_feature_extractor(self.feature_extractor, feature_extractor_size) # Remove the classifier/final layer
        if self.aggregate_node_classifications is True:
            self.fc = nn.Linear(self.feature_extractor_output_size, num_classes)
            self.feature_extractor_output_size = num_classes

        print("Number of all parameters in feature extractor NN:", sum(p.numel() for p in self.feature_extractor.parameters()))
        self.feature_extractor = feature_extractor_layer_freezing(self.feature_extractor, feature_extractor_size, self.freeze_upper_N_layers, max_freeze_percentage)
        print("Number of TRAINABLE parameters in feature extractor NN:", sum(p.numel() for p in self.feature_extractor.parameters() if p.requires_grad))

        print("Using", aggregator_type, "aggregation...")
        print("Input h_feats are:", h_feats)
        print("Feature extractor has output of shape:", self.feature_extractor_output_size)

        self.ngnns = []
        if len(h_feats) == 0:
            first_conv = SAGEConv(self.feature_extractor_output_size, num_classes, aggregator_type=aggregator_type)
        else:
            norm_layer = None
            if self.use_normalization is True:
                norm_layer = nn.BatchNorm1d(num_features = h_feats[0])
            
            activation = None
            if self.use_ELU_in_graph_layers is True:
                activation = nn.ELU()

            first_conv = SAGEConv(self.feature_extractor_output_size, h_feats[0], aggregator_type=aggregator_type, feat_drop=feat_dropout, norm=norm_layer, activation=activation)
            if self.use_ngnn_modules is True:
                print("Using NGNN modules...")
                self.ngnns.append(nn.Linear(h_feats[0], h_feats[0]))
            else:
                print("Not using NGNN modules...")
        
        self.convs = [first_conv]
        for i in range(1, len(h_feats)):
            norm_layer = None
            if self.use_normalization is True:
                norm_layer = nn.BatchNorm1d(num_features = h_feats[i])

            activation = None
            if self.use_ELU_in_graph_layers is True:
                activation = nn.ELU()

            self.convs.append(SAGEConv(h_feats[i-1], h_feats[i], aggregator_type=aggregator_type, feat_drop=feat_dropout, norm=norm_layer, activation=activation))
            if self.use_ngnn_modules is True:
                self.ngnns.append(nn.Linear(h_feats[i], h_feats[i]))
            
        if len(h_feats) >= 1:
            self.convs.append(SAGEConv(h_feats[-1], num_classes, aggregator_type=aggregator_type))
            if self.use_ngnn_modules is True:
                self.ngnns.append(nn.Linear(num_classes, num_classes))
            
        print("Number of Graph layers:", len(self.convs))
        self.convs = nn.ModuleList(self.convs)
        if len(self.ngnns) > 0:
            self.ngnns = nn.ModuleList(self.ngnns)

        print("Number of all parameters:", sum(p.numel() for p in self.parameters()), "/ Number of learnable parameters:", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, mfgs, x):
        x = torch.reshape(x, (x.shape[0], self.image_size[0], self.image_size[1], -1)).permute(0,3,1,2)
        
        x = self.normalization_transform(x)
        x = self.conv_change_feature_channels_to_3(x)
        if self.feature_extractor_transform is not None:
            x = self.feature_extractor_transform(x)

        if self.training:
            x = self.augmentation_transform(x)

        x = self.feature_extractor(x)
        if self.aggregate_node_classifications is True:
            x = self.fc(x)
        x = x.reshape(-1, self.feature_extractor_output_size)

        data = x
        for i in range(0, len(self.convs)):
            h_dst = data[:mfgs[i].num_dst_nodes()]
            if self.use_edge_weights is False:
                h = self.convs[i](mfgs[i], (data, h_dst))
            else:
                edge_weights = mfgs[i].edata['weight'].float()
                h = self.convs[i](mfgs[i], (data, h_dst), edge_weights)
            
            if i < (len(self.convs) - 1):

                if len(self.ngnns) > 0:
                    print("Applying NGNNS....", i)
                    h = self.ngnns[i](h)
        
            data = h

        return data

class MyModelOnlyCNN(nn.Module):
    def __init__(self, h_feats: list,
                 num_classes,
                 image_size=(32, 32, 3),
                 feature_extractor_size="EfficientNetV2-S",
                 feat_dropout=0.5,
                 freeze_upper_N_layers = 6,
                 pooling_method='avg',
                 max_freeze_percentage = 90.0,
                 train_channel_means = [0.0, 0.0, 0.0, 0.0],
                 train_channel_stds = [1.0, 1.0, 1.0, 1.0]):
        super(MyModelOnlyCNN, self).__init__()

        print("Created NN without GNN part!")

        print("Will perform normalization of raw data with: means:", train_channel_means, "and stds:", train_channel_stds)
        self.normalization_transform = create_normalization_transform(train_channel_means[:image_size[2]], train_channel_stds[:image_size[2]]) #3 or 4 channels
        self.augmentation_transform = create_augmentation_transform()

        self.image_size = image_size
        self.freeze_upper_N_layers = freeze_upper_N_layers
        self.pooling_method = pooling_method
        
        if self.image_size[2] != 3:
            self.conv_change_feature_channels_to_3 = nn.Conv2d(in_channels=self.image_size[2], out_channels=3, kernel_size=1, stride=1)
        else:
            print("Image depth is 3, so additional CONV layer is not needed! Just nn.Identity() used!")
            self.conv_change_feature_channels_to_3 = nn.Identity()
        
        self.feature_extractor, self.feature_extractor_transform, self.feature_extractor_output_size = create_feature_extractor(feature_extractor_size, self.freeze_upper_N_layers, feat_dropout, in_channels=self.image_size[2])

        if feature_extractor_size == "EfficientNetV2-S" or feature_extractor_size == "EfficientNetV2-M" or feature_extractor_size == "EfficientNetV2-L":
            self.feature_extractor_transform.resize_size = [self.image_size[0]]
            self.feature_extractor_transform.crop_size = [self.image_size[0]]

        self.feature_extractor = change_last_layer_of_first_feature_extractor(self.feature_extractor, feature_extractor_size) # Remove the classifier/final layer

        print("Number of all parameters in feature extractor NN:", sum(p.numel() for p in self.feature_extractor.parameters()))
        self.feature_extractor = feature_extractor_layer_freezing(self.feature_extractor, feature_extractor_size, self.freeze_upper_N_layers, max_freeze_percentage)
        print("Number of TRAINABLE parameters in feature extractor NN:", sum(p.numel() for p in self.feature_extractor.parameters() if p.requires_grad))

            
        self.fc = nn.Linear(self.feature_extractor_output_size, num_classes)

        print("Number of all parameters:", sum(p.numel() for p in self.parameters()), "/ Number of learnable parameters:", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, mfgs, x):
        x = torch.reshape(x, (x.shape[0], self.image_size[0], self.image_size[1], -1)).permute(0,3,1,2)

        x = self.normalization_transform(x)
        x = self.conv_change_feature_channels_to_3(x)
        if self.feature_extractor_transform is not None:
            x = self.feature_extractor_transform(x)

        if self.training:
            x = self.augmentation_transform(x)

        x = self.feature_extractor(x)
        x = x.reshape(-1, self.feature_extractor_output_size)

        x = self.fc(x)
        x = x.squeeze(1)

        return x