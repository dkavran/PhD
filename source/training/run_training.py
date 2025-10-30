import numpy as np
import pandas as pd
import networkx as nx
import dgl
import math
import tqdm
import sklearn.metrics
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import pickle
import pathlib
import argparse
import json
import os
import torch.multiprocessing as mp
import torch.distributed as dist
import time
import gc
import torch.nn.functional as F
from torch.autograd import Variable

# Implemented imports
import sys
sys.path.append('../')
from proposed_method_lib.graph import segment_images, plot_nx_graph, save_graphs_to_file, load_item_with_pickle, get_indices_and_labels_for_time_frame_without_labels
from proposed_method_lib.dgl import MyModelGraphSAGE, MyModelGraphTransformer, MyModelGAT, MyModelOnlyCNN, get_label_conversion_table, get_correct_labels, ProposedSampler

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["NCCL_SOCKET_IFNAME"] = "lo"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = '1'
torch.backends.cudnn.benchmark = True

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

# Normalize a path to Windows long-path format (\\?\ or \\?\UNC\) (Mentioned/solved in: https://stackoverflow.com/questions/36219317/pathname-too-long-to-open)
def improve_path(path):
        path = os.path.abspath(path)
        if path.startswith(u"\\\\"):
                path=u"\\\\?\\UNC\\"+path[2:]
        else:
                path=u"\\\\?\\"+path
        return path


def main():
        parser = argparse.ArgumentParser()

        parser.add_argument('--gpus', default=1, type=int, help='Number of GPUs')
        parser.add_argument('--epochs', default=2, type=int, metavar='N', help='Total epochs')
        parser.add_argument('--runs', default=10, type=int, help="Total independent runs (with reseeding)")
        parser.add_argument('--train_places', help="Train dataset places", nargs='+', default=[])
        parser.add_argument('--test_places', help="Test dataset places", nargs='+', default=[])
        parser.add_argument('--train_pickle_file_name', help="Name of train graph pickle file", type=str, default="joined_graph_with_features.pickle")
        parser.add_argument('--test_pickle_file_name', help="Name of train graph pickle file", type=str, default="joined_graph_with_features.pickle")
        parser.add_argument('--region_bbox_size', default=48, type=int, help="Feature bounding bbox size (square)")
        parser.add_argument('--region_bbox_channels', default=3, type=int, help="Feature bounding bbox number of channels")
        parser.add_argument('--time_step_past', default=0, type=int, help="Past time step")
        parser.add_argument('--spatial_step_surrounding_region', default=0, type=int, help="How many steps in spatial neighborhood")
        parser.add_argument('--time_neighborhood_size', default=10, type=int, help="Fanout per hop in time")
        parser.add_argument('--space_neighborhood_size', default=10, type=int, help="Fanout per hop in space")
        parser.add_argument('--freeze_upper_N_layers', default=100000, type=int, help="Freeze the top N layers of the feature extractor (0 = none, but load pretrained weights; large number = freeze most of layers with loaded pretrained weights; None = none, but NO pretrained weights)")
        parser.add_argument('--aggregator_type', default=None, type=str, help="GraphSAGE aggregator ('mean', 'pool', 'lstm', …)")
        parser.add_argument('--use_edge_weights', default=False, type=bool, help="Whether to use weights on edges")
        parser.add_argument('--lr', default=0.0001, type=float, help="Learning rate")
        parser.add_argument('--feat_dropout', default=0.5, type=float, help="Feature dropout")
        parser.add_argument('--batch_size', default=30, type=int, help="Batch size")
        parser.add_argument('--test_batch_size', default=60, type=int, help="Batch size for validation/testing")
        parser.add_argument('--data_path', default="E:/PhD/methods/my_method/pipeline/2_perform_segmentation_and_data_preparation/prepared_data", type=str, help="Disk path to data")
        parser.add_argument('--use_GAT', help="Whether to use GAT neural network", action='store_true')
        parser.add_argument('--use_transformer', help="Whether to use Transformer neural network (GraphormerLayers)", action='store_true')
        parser.add_argument('--use_only_CNN', help="Use only the image feature extractor CNN + pooling for classification (no GNN)", action='store_true')
        parser.add_argument('--only_CNN_pooling_method', default=None, help="What kind of pooling to use in only CNN classification NN", choices=[None, 'avg', 'max'])
        parser.add_argument('--number_of_workers', default=0, type=int, help="Number of workers in dataloaders")
        parser.add_argument('--graph_dtype', default='int32', help="Whether graphs should be of type int32 or int64 (allowed values: 'int32', 'int64')")
        parser.add_argument('--use_custom_sampler', help="Use implemented custom time/space edge-type sampler", action='store_true')
        parser.add_argument('--h_feat_amount', default=256, type=int, help="Size of hidden layers")
        parser.add_argument("--feature_extraction_NN", default="EfficientNetV2-S", type=str, help="Which feature extraction (C)NN/Vision transformer to use")
        parser.add_argument("--num_heads", default="8", type=int, help="Number of heads in Transformer GNN")
        parser.add_argument("--use_ngnn_modules", help="Whether to use NGNN layers (just added FC layer after each GNN selected layer). Proposed in: https://arxiv.org/abs/2111.11638", action='store_true')
        parser.add_argument("--my_pretrained_CNN_weights_folder", default=None, type=str, help="folder with model of a trained model using only the CNN (e.g. ./models/my_interval_monthly_padding_16_bbox_size_32_layers_only_rgb_sc_10_sigma_1.0_min_size_50/1/final_model_run_3.pt)")
        parser.add_argument("--use_class_weights", help="Use class-balanced weights in CrossEntropy", action='store_true')
        parser.add_argument("--use_focal_loss", help="Use Focal Loss instead of CrossEntropy", action='store_true')
        parser.add_argument("--max_layer_freeze_percentage", help="Percentage of parameters that can be frozen in the feature extractor", default=80.0, type=float)
        parser.add_argument("--early_stopping_patience", default=3, type=int, help="Number of epochs trained, with no improvement, before terminating the training (of that run) based on validation loss")
        parser.add_argument("--use_uva", action='store_true', help="Whether to use Unified Virtual Addressing (UVA) to directly sample the graph and slice the features from CPU into GPU. Setting it to True will pin the graph and feature tensors into pinned memory.")
        parser.add_argument("--use_AMP", action='store_true', help="Whether to use Automatic Mixed Precision (AMP).")
        parser.add_argument("--use_normalization", action='store_true', help="Whether to use batch/layer normalization at the end of graph layers.")
        parser.add_argument("--use_ELU_in_graph_layers", action='store_true', help="Whether to use ELU activation at the end of graph layers.")
        parser.add_argument("--aggregate_node_classifications", action='store_true', help="Whether to use GNN to aggregate and classify each target node based on classifications of each individual node")
        parser.add_argument("--fl_gamma", default="2.0", type=float, help="Gamma for Focal Loss")
        
        args = parser.parse_args()
        print("Number of GPUs:", args.gpus)
        print("Epochs:", args.epochs)
        print("Runs:", args.runs)
        print("------")
        print("Train path:", args.train_places)
        print("Test path:", args.test_places)
        print("Train pickle file name:", args.train_pickle_file_name)
        print("Test pickle file name:", args.test_pickle_file_name)
        print("------")
        print("Region bbox size:", args.region_bbox_size)
        print("Region bbox number of channels:", args.region_bbox_channels)
        print("Temporal steps in history:", args.time_step_past)
        print("Spatial steps in surrounding region:", args.spatial_step_surrounding_region)
        print("Time neighborhood size:", args.time_neighborhood_size)
        print("Space neighborhood size:", args.space_neighborhood_size)
        print("------")
        print("Freeze upper N layers:", args.freeze_upper_N_layers)
        print("Aggregator type:", args.aggregator_type)
        print("Use edge weights?", args.use_edge_weights)
        print("------")
        print("Learning rate:", args.lr)
        print("Feature dropout:", args.feat_dropout)
        print("Batch size:", args.batch_size)
        print("Test batch size:", args.test_batch_size)
        print("Feature extraction NN:", args.feature_extraction_NN)
        print("Max percentage of parameters that can be frozen in feature extraction NN:", args.max_layer_freeze_percentage)
        print("Early stopping patience:", args.early_stopping_patience)
        print("------")
        print("Data path:", args.data_path)
        print("Use GAT?:", args.use_GAT)
        print("Use Graph transformer?", args.use_transformer)
        print("Attention number of heads:", args.num_heads)
        print("Using NGNN modules?", args.use_ngnn_modules)
        print("Use only CNN?", args.use_only_CNN)
        print("CNN pooling method:", args.only_CNN_pooling_method)
        print("Number of workers:", args.number_of_workers)
        print("Graph dtype:", args.graph_dtype)
        print("Custom sampler:", args.use_custom_sampler)
        print("H_feat amount:", args.h_feat_amount)
        print("Path of the pretrainde only CNN model:", args.my_pretrained_CNN_weights_folder)
        print("Use UVA:", args.use_uva)
        print("Use AMP:", args.use_AMP)
        print("Use layer/batch normalization:", args.use_normalization)
        print("Use ELU activation in graph layers:", args.use_ELU_in_graph_layers)
        print("Aggregate node classifications:", args.aggregate_node_classifications)
        
        args.segmentation_mask_folder_path = args.data_path
        
        print("Segmentation mask folder path:", args.segmentation_mask_folder_path)

        if args.use_GAT is True and args.use_only_CNN is True:
                print("You can not use GAT and only CNN neural networks!")
                return

        if args.use_transformer is True and args.use_only_CNN is True:
                print("You can not use Transformer and only CNN neural networks!")
                return

        args.devices = list(range(args.gpus))
        args.image_size = (args.region_bbox_size, args.region_bbox_size, args.region_bbox_channels)
        
        print("\n\n")
        print("Loading data...")

        # Lists to store train and test graphs
        train_dgl_graphs = []
        test_dgl_graphs = []
        
        # Map edge types to integer IDs for DGL
        edge_type_conversion = {
            'temporal': 0,
            'spatial': 1
        }

        # Get train graphs
        print("Working on train datasets...")
        for place in args.train_places:
                path = args.data_path

                # Read NetworkX graph
                graph_nx = pd.read_pickle(path + "/" + place + "/" + args.train_pickle_file_name)

                # Create DGL graphs
                for u, v, data in graph_nx.edges(data=True):
                    data['type'] = edge_type_conversion[data['type']]
                dgl_graph = dgl.from_networkx(graph_nx, node_attrs=["features", "label", "time"], edge_attrs=['type'])
                

                # Delete/free NetworkX graphs
                graph_nx = None
                del graph_nx
                gc.collect()

                if args.use_GAT is True:
                        print("Train: Add self-loops required by GAT; use edge type ID 2 for self edges")
                        dgl_graph = dgl.add_self_loop(dgl_graph, edge_feat_names=["type"], fill_data=2) #necessary for GAT neural networks, where self-loop type is marked as '2'

                # Necessary to convert DGL graphs into int64/int32 format
                if args.graph_dtype == 'int64':
                        dgl_graph = dgl_graph.long()
                elif args.graph_dtype == 'int32':
                        dgl_graph = dgl_graph.int()
                else:
                        raise Exception("Incorrect graph dtype!")
                
                train_dgl_graphs.append(dgl_graph)

        # Get test graphs
        print("Working on validation datasets...")
        for place in args.test_places:
                path = args.data_path

                graph_nx = pd.read_pickle(path + "/" + place + "/" + args.test_pickle_file_name)

                for u, v, data in graph_nx.edges(data=True):
                    data['type'] = edge_type_conversion[data['type']]
                dgl_graph = dgl.from_networkx(graph_nx, node_attrs=["features", "label", "time"], edge_attrs=['type'])

                graph_nx = None
                del graph_nx
                gc.collect()

                if args.use_GAT is True:
                        print("Test: Add self-loops required by GAT; use edge type ID 2 for self edges")
                        dgl_graph = dgl.add_self_loop(dgl_graph, edge_feat_names=["type"], fill_data=2)

                if args.graph_dtype == 'int64':
                        dgl_graph = dgl_graph.long()
                elif args.graph_dtype == 'int32':
                        dgl_graph = dgl_graph.int()
                else:
                        raise Exception("Incorrect graph dtype!")
                
                test_dgl_graphs.append(dgl_graph)

        # Join/batch train and test graphs
        train_dgl_graph = dgl.batch(train_dgl_graphs)
        test_dgl_graph = dgl.batch(test_dgl_graphs)

        # Delete both arrays of graphs
        train_dgl_graphs = None
        test_dgl_graphs = None
        del train_dgl_graphs
        del test_dgl_graphs
        gc.collect()
                
        train_node_labels = train_dgl_graph.ndata['label'].numpy()
        test_node_labels = test_dgl_graph.ndata['label'].numpy()
        labels_in_train_and_test_nodes = np.concatenate((train_node_labels, test_node_labels))

        label_table = get_label_conversion_table(labels_in_train_and_test_nodes)

        train_node_labels_corrected = get_correct_labels(train_node_labels, label_table)
        test_node_labels_corrected = get_correct_labels(test_node_labels, label_table)

        train_dgl_graph.ndata['label'] = torch.tensor(train_node_labels_corrected).int()
        test_dgl_graph.ndata['label'] = torch.tensor(test_node_labels_corrected).int()

        train_dgl_graph.ndata['time'] = train_dgl_graph.ndata['time'].int()
        test_dgl_graph.ndata['time'] = test_dgl_graph.ndata['time'].int()

        # Calculate class weights
        classes = np.unique(train_node_labels_corrected)
        class_weights_array = compute_class_weight(class_weight="balanced", classes=classes, y=train_node_labels_corrected)
        class_weights = dict(zip(classes, class_weights_array))

        # Add weight for non present classes in class weights
        for entry in list(label_table.values()):
                if entry not in list(class_weights.keys()):
                        class_weights[entry] = 1.0

        class_weights = dict(sorted(class_weights.items(), key=lambda x: x[0]))
        num_classes = len(class_weights)

        print("Number of classes:", num_classes)
        print("Class weights:", class_weights)

        # Load per-channel mean/std for normalizing per-node bounding box images
        with open(args.data_path + "/" + args.train_places[0] + 'train_set_normalization_data.json', 'r') as file:
                normalization_data = json.load(file)

        train_channel_means = np.array([float(normalization_data["means"]["R"]), float(normalization_data["means"]["G"]), float(normalization_data["means"]["B"]), float(normalization_data["means"]["infra"])])
        train_channel_stds = np.array([float(normalization_data["stds"]["R"]), float(normalization_data["stds"]["G"]), float(normalization_data["stds"]["B"]), float(normalization_data["stds"]["infra"])])
        print("Loaded normalization means:", train_channel_means, "and stds:", train_channel_stds)

        # Train and test indices of nodes in the graphs
        train_node_indices = torch.tensor(np.arange(0, train_node_labels_corrected.shape[0]))
        test_node_indices = torch.tensor(np.arange(0, test_node_labels_corrected.shape[0]))

        print("Train set has", len(train_node_indices), "nodes and test set has", len(test_node_indices), "nodes")
        print("With train labels:", np.unique(train_node_labels_corrected),"and test labels:", np.unique(test_node_labels_corrected))

        # Fanout per hop: [space fanouts per spatial hop] + [time fanouts per temporal hop].
        sampling_amount = [args.space_neighborhood_size for i in range(0, args.spatial_step_surrounding_region)] + [args.time_neighborhood_size for i in range(0, args.time_step_past)]

        # Append/Prepend a trailing 0 to align with DGL’s in-neighbor sampling when no neighborhood is included
        if args.time_step_past == 0 and args.spatial_step_surrounding_region == 0:
                sampling_amount = [0, 0] # If using GAT, set to [1], otherwise set to [0]
        elif args.time_step_past == 0 and args.spatial_step_surrounding_region > 0:
                sampling_amount = sampling_amount + [0]
        elif args.time_step_past > 0 and args.spatial_step_surrounding_region == 0:
                sampling_amount = [0] + sampling_amount

        h_feats_max_count = args.spatial_step_surrounding_region + args.time_step_past - 1
        if args.spatial_step_surrounding_region == 0 and args.time_step_past > 0:
                h_feats_max_count = args.time_step_past
        elif args.spatial_step_surrounding_region > 0 and args.time_step_past == 0:
                h_feats_max_count = args.spatial_step_surrounding_region
        elif args.use_transformer is True and args.spatial_step_surrounding_region == 0 and args.time_step_past == 0: # Edge case
                h_feats_max_count = 1



        h_feats = [args.h_feat_amount for i in range(0, h_feats_max_count, 1)]

        print("Sampling amount:", sampling_amount, ", hidden features:", h_feats, "/ time steps in spatial neighborhood:", args.spatial_step_surrounding_region, "/ time steps in the past:", args.time_step_past)
        print("Feature dropout:", args.feat_dropout, ", batch size:", args.batch_size)

        n_heads = 8

        args.train_dgl_graph = train_dgl_graph
        args.test_dgl_graph = test_dgl_graph
        args.sampling_amount = sampling_amount
        args.num_classes = num_classes
        args.h_feats = h_feats
        args.n_heads = n_heads
        args.train_node_indices = train_node_indices
        args.test_node_indices = test_node_indices
        args.class_weights = class_weights
        args.train_channel_means = train_channel_means
        args.train_channel_stds = train_channel_stds

        # Perform training on multiple GPUs (I have a single GPU, so the main process (0) is the only process)
        mp.spawn(train, nprocs=args.gpus, args=(args,))


def train(gpu, args):
        N_epochs = args.epochs
        N_runs = args.runs
        using_GAT = args.use_GAT
        using_transformer = args.use_transformer
        using_only_CNN = args.use_only_CNN
        train_dgl_graph = args.train_dgl_graph
        test_dgl_graph = args.test_dgl_graph
        sampling_amount = args.sampling_amount
        num_classes = args.num_classes
        h_feats = args.h_feats
        n_heads = args.n_heads
        train_node_indices = args.train_node_indices
        test_node_indices = args.test_node_indices
        class_weights = args.class_weights
        image_size = args.image_size
        scaling_used = args.scaling

        devices = args.devices
        dev_id = devices[gpu]
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(master_ip='127.0.0.1', master_port='12345')

        plot_progress_bar = False if dev_id == 0 else True

        device = None
        if torch.cuda.device_count() < 1:
                raise Exception("GPU is not available!")
        else:
                print("Number of available GPUs:", torch.cuda.device_count())
                torch.cuda.set_device(dev_id)
                device = torch.device('cuda:' + str(dev_id))
                print("Training on gpu", gpu, ", device:", device, "dev_id:", dev_id)

        if args.use_uva is True:
                train_node_indices = train_node_indices.to(device)
                test_node_indices = test_node_indices.to(device)
           
        spatial_step_amount = args.spatial_step_surrounding_region if args.spatial_step_surrounding_region >= 1 else 1
        time_step_amount = args.time_step_past if args.time_step_past >= 1 else 1
        
        edge_type_conversion = {
            'temporal': 0,
            'spatial': 1
        }
        
        spatial_int_value = edge_type_conversion['spatial']
        temporal_int_value = edge_type_conversion['temporal' ]
        edge_sequence_stages = [(spatial_int_value, spatial_step_amount), (temporal_int_value, time_step_amount)]
        
        if args.use_custom_sampler is True: # Implemented custom time/space edge-type sampler
                node_sampler = ProposedSampler(fanouts=sampling_amount, edge_sequence_stages=edge_sequence_stages, gat_nn=args.use_GAT, edge_dir="in")
        else:
                node_sampler = dgl.dataloading.NeighborSampler(sampling_amount, edge_dir="in") # Default sampling algorithm

        _, count_unqs = np.unique(train_node_indices.cpu().numpy(), return_counts=True)

        train_dataloader = dgl.dataloading.DataLoader(
                train_dgl_graph,                          # The graph
                train_node_indices,              # The node IDs to iterate over in minibatches
                node_sampler,                   # The neighbor sampler
                device=device,    # Put the sampled MFGs on CPU or GPU


                # The following arguments are inherited from PyTorch DataLoader.
                batch_size=args.batch_size,     # Batch size
                shuffle=True, # Whether to shuffle the nodes for every epoch
                drop_last=False,        # Whether to drop the last incomplete batch
                num_workers=args.number_of_workers,        # Number of sampler processes,
                use_ddp=False, # False, because we don't do distributed training
                use_uva=args.use_uva
        )

        test_dataloader = dgl.dataloading.DataLoader(
                test_dgl_graph,                   
                test_node_indices,               
                node_sampler,                   
                device=device,    
                batch_size=args.test_batch_size,     
                shuffle=False,     
                drop_last=False,        
                num_workers=args.number_of_workers,        
                use_ddp=False,
                use_uva=args.use_uva
        )

        run_losses = []
        run_accuracies = []
        run_weighted_precisions = []
        run_macro_precisions = []
        run_micro_precisions = []
        run_weighted_recalls = []
        run_macro_recalls = []
        run_micro_recalls = []
        run_weighted_f1s = []
        run_macro_f1s = []
        run_micro_f1s = []
        run_best_losses = []

        GNN_used = 'GraphSAGE'
        if args.use_GAT is True and args.use_transformer is False and args.use_only_CNN is False:
                GNN_used = 'GAT'
        if args.use_GAT is False and args.use_transformer is True and args.use_only_CNN is False:
                GNN_used = 'Transformer'
        elif args.use_GAT is False and args.use_transformer is False and args.use_only_CNN is True:
                GNN_used = 'NONE' # Using only CNN (without GNN) with pooling as classification model
                GNN_used += "_pooling_" + (args.only_CNN_pooling_method or '')
        
        use_ngnn = "TRUE" if args.use_ngnn_modules is True else "FALSE"
        use_my_pretrained_CNN_weights = "TRUE" if args.my_pretrained_CNN_weights_folder is not None else "FALSE"

        sampler_used = "custom"
        if args.use_custom_sampler is False:
                sampler_used = "original"

        aggregator_type_str = "NONE" if args.aggregator_type is None else args.aggregator_type

        using_class_weights = "TRUE" if args.use_class_weights is True else "FALSE"
        loss_name = "focal" + "_" + str(args.fl_gamma) if args.use_focal_loss is True else "cross_entropy"
        use_norm = "Y" if args.use_normalization is True else "N"
        use_elu_graph_layers = "Y" if args.use_ELU_in_graph_layers is True else "N"
        agg_nodes = "Y" if args.aggregate_node_classifications is True else "N"
        save_folder_path = "/" + '_'.join(args.train_places) + "/" + 'GNN_' + GNN_used + "_h_fts_" + str(args.h_feat_amount) + '_t_stp_' + str(args.time_step_past) + "_s_stp_" + str(args.spatial_step_surrounding_region) + '_nghborhood_' + 'time_' + str(args.time_neighborhood_size) + '_space_' + str(args.space_neighborhood_size) + '_batch_' + str(args.batch_size) + '_agg_' + aggregator_type_str + "_smplr_" + sampler_used + "_FE_" + args.feature_extraction_NN + "_frzn_lyrs_" + str(args.freeze_upper_N_layers) + "_max_fr_perc_" + str(int(args.max_layer_freeze_percentage)) + "_NGNN_" + use_ngnn + "_use_preTR_CNN_" + use_my_pretrained_CNN_weights + "_drpout_" + str(args.feat_dropout) + "_cls_wghts_" + using_class_weights + "_loss_" + loss_name + "_LyrNorm_" + use_norm + "_ELU_" + use_elu_graph_layers + "_agg_nds_" + agg_nodes

        if dev_id == 0:
                # Create /log and /models folders for storing logs and weights
                if not os.path.exists('./log'):
                        os.makedirs('./log')
                if not os.path.exists('./models'):
                        os.makedirs('./models')

                if not os.path.exists(improve_path('./log/' + save_folder_path)):
                        os.makedirs(improve_path('./log/' + save_folder_path))
                if not os.path.exists(improve_path('./models/' + save_folder_path)):
                        os.makedirs(improve_path('./models/' + save_folder_path))


        start_training_time = time.time()

        print("Saving argument object...")
        args_save_path = './models/' + save_folder_path + '/model_args.pickle'
        with open(improve_path(args_save_path), 'wb') as handle:
                pickle.dump(args, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        args2 = load_item_with_pickle(improve_path(args_save_path))
        os.remove(improve_path(args_save_path))
        del args2.train_dgl_graph # Release memory
        del args2.test_dgl_graph

        with open(improve_path(args_save_path), 'wb') as handle:
                pickle.dump(args2, handle, protocol=pickle.HIGHEST_PROTOCOL)

        for run in range(0, N_runs):
                if dev_id == 0:
                        print("Run number", (run + 1), "/", N_runs)

                torch.manual_seed(0)

                # Create DL model to train
                if using_GAT is True and using_transformer is False and using_only_CNN is False:
                        if dev_id == 0:
                                print("Creating GAT neural network...")
                        model = MyModelGAT(h_feats=h_feats,
                                num_classes=num_classes,
                                image_size=image_size,
                                feature_extractor_size=args.feature_extraction_NN,
                                feat_dropout=args.feat_dropout,
                                n_heads=n_heads,
                                freeze_upper_N_layers=args.freeze_upper_N_layers,
                                use_ngnn_modules=args.use_ngnn_modules,
                                max_freeze_percentage=args.max_layer_freeze_percentage,
                                train_channel_means = args.train_channel_means,
                                train_channel_stds = args.train_channel_stds,
                                use_ELU_in_graph_layers=args.use_ELU_in_graph_layers,
                                aggregate_node_classifications=args.aggregate_node_classifications).to(device)
                elif using_GAT is False and using_transformer is False and using_only_CNN is False:
                        if dev_id == 0:
                                print("Creating GraphSAGE neural network...")
                        model = MyModelGraphSAGE(h_feats=h_feats,
                                num_classes=num_classes,
                                image_size=image_size,
                                feature_extractor_size=args.feature_extraction_NN,
                                aggregator_type=args.aggregator_type,
                                feat_dropout=args.feat_dropout,
                                use_edge_weights=args.use_edge_weights,
                                freeze_upper_N_layers=args.freeze_upper_N_layers,
                                use_ngnn_modules=args.use_ngnn_modules,
                                max_freeze_percentage=args.max_layer_freeze_percentage,
                                train_channel_means = args.train_channel_means,
                                train_channel_stds = args.train_channel_stds,
                                use_normalization=args.use_normalization,
                                use_ELU_in_graph_layers=args.use_ELU_in_graph_layers,
                                aggregate_node_classifications=args.aggregate_node_classifications).to(device)
                elif using_GAT is False and using_transformer is True and using_only_CNN is False:
                        if dev_id == 0:
                                print("Creating Transformer neural network...")
                        model = MyModelGraphTransformer(h_feats=h_feats,
                                num_classes=num_classes,
                                image_size=image_size,
                                feature_extractor_size=args.feature_extraction_NN,
                                num_heads=args.num_heads,
                                feat_dropout=args.feat_dropout,
                                use_edge_weights=args.use_edge_weights,
                                freeze_upper_N_layers=args.freeze_upper_N_layers,
                                use_ngnn_modules=args.use_ngnn_modules,
                                max_freeze_percentage=args.max_layer_freeze_percentage,
                                train_channel_means = args.train_channel_means,
                                train_channel_stds = args.train_channel_stds,
                                use_normalization=args.use_normalization,
                                aggregate_node_classifications=args.aggregate_node_classifications).to(device)
                else:
                        if dev_id == 0:
                                print("Creating only CNN neural network...")

                        model = MyModelOnlyCNN(
                                h_feats=h_feats,
                                num_classes=num_classes,
                                image_size=image_size,
                                feature_extractor_size=args.feature_extraction_NN,
                                feat_dropout=args.feat_dropout,
                                freeze_upper_N_layers=args.freeze_upper_N_layers,
                                pooling_method=args.only_CNN_pooling_method,
                                max_freeze_percentage=args.max_layer_freeze_percentage,
                                train_channel_means = args.train_channel_means,
                                train_channel_stds = args.train_channel_stds
                        ).to(device)

                # Load pretrained weights from the trained only CNN
                if args.my_pretrained_CNN_weights_folder is not None and using_only_CNN is False:
                        pretrained_only_CNN_weights_full_path = args.my_pretrained_CNN_weights_folder
                        print("Loading pretrained only CNN_weights from:", pretrained_only_CNN_weights_full_path)
                        pretrained_dict = torch.load(pretrained_only_CNN_weights_full_path)
                        print("Loaded pretrained only CNN weights!")
                        
                        if hasattr(model, 'fc'):
                                print("Weights in .fc layer before loading the pretrained weights:", model.state_dict()["fc.weight"])
                        
                        # Load weights for specific layers
                        model_dict = model.state_dict()
                        # 1. Filter out unnecessary keys
                        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                        # 2. Overwrite entries in the existing state dict
                        model_dict.update(pretrained_dict) 
                        # 3. Load the new state dict
                        model.load_state_dict(model_dict)
                        
                        print("Freezing all layers from feature extractor!")
                        print("Loaded pretrained state dict keys to the model:")
                        for k, v in pretrained_dict.items():
                                print("key:", k)

                        # Freeze the selected layers
                        for param in model.conv_change_feature_channels_to_3.parameters():
                                param.requires_grad = False

                        for param in model.feature_extractor.parameters():
                                param.requires_grad = False

                        if hasattr(model, 'fc'):
                                print("model has FC layer and we are freezing it now!")
                                for param in model.fc.parameters():
                                        param.requires_grad = False
                                print("Weights in .fc layer AFTER loading the pretrained weights:", model.state_dict()["fc.weight"])

                        # Count the total number of parameters (weights) in the model
                        total_params = sum(p.numel() for p in model.parameters())

                        # Count the number of trainable parameters (weights) in the model
                        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

                        # Print the results of loading
                        print(f"Total number of NN parameters after loading the FE network: {total_params}")
                        print(f"Number of NN trainable parameters after loading the FE network: {trainable_params}")

                opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.25)

                # Create loss criterion - either Cross Entropy loss, possibly weighted by class weights, OR Focal Loss
                if args.use_focal_loss is True:
                        print("Using Focal loss with gamma:", args.fl_gamma)
                        loss_criterion_weighted = FocalLoss(gamma=args.fl_gamma)
                else:
                        print("Using Cross Entropy loss!")
                        if args.use_class_weights:
                                loss_criterion_weighted = nn.CrossEntropyLoss(weight=torch.tensor(list(class_weights.values())).float().to(device))
                                print("With class weights...")
                                print(class_weights)
                        else:
                                loss_criterion_weighted = nn.CrossEntropyLoss()
                                print("Without class weights...")

                # Perform training of the DL model
                softmax_predictions_all_epochs = []
                predictions_all_epochs = []
                labels_all_epochs = []
                losses_all_epochs = []

                final_loss = 0.0
                final_accuracy = 0.0
                final_weighted_precision = 0.0
                final_micro_precision = 0.0
                final_macro_precision = 0.0
                final_weighted_recall = 0.0
                final_micro_recall = 0.0
                final_macro_recall = 0.0
                final_weighted_f1 = 0.0
                final_micro_f1 = 0.0
                final_macro_f1 = 0.0

                early_stopping_patience_counter = 0
                best_val_loss = float('inf')
                best_epoch_index = -1

                scaler = None
                if args.use_AMP is True:
                        scaler = torch.cuda.amp.GradScaler()

                # Iteration through the epochs
                for epoch in range(N_epochs):
                        if early_stopping_patience_counter >= args.early_stopping_patience:
                                with open(improve_path('./log/' + save_folder_path + '/training_progress.txt'), 'a') as file:
                                        file.write("Early stopping triggered. Ending training in run #" + str(run + 1) + "! Best loss at early stopping: " + str(best_val_loss) + ".\n")
                                break

                        start_epoch_time = time.time()
                        model.train()

                        train_epoch_avg_loss = 0.0
                        train_epoch_avg_accuracy = 0.0
                        train_epoch_avg_weighted_f1 = 0.0

                        with tqdm.tqdm(train_dataloader, disable=plot_progress_bar) as tq:
                                epoch_avg_accuracy = 0.0
                                epoch_avg_loss = 0.0
                                epoch_avg_weighted_precision = 0.0
                                epoch_avg_macro_precision = 0.0
                                epoch_avg_micro_precision = 0.0
                                epoch_avg_weighted_recall = 0.0
                                epoch_avg_macro_recall = 0.0
                                epoch_avg_micro_recall = 0.0
                                epoch_avg_weighted_f1 = 0.0
                                epoch_avg_macro_f1 = 0.0
                                epoch_avg_micro_f1 = 0.0

                                for step, x in enumerate(tq):
                                        mfgs = x[2]
                                        inputs = mfgs[0].srcdata['features'].float()
                                        labels = mfgs[-1].dstdata['label'].long()

                                        if args.use_AMP is False:
                                                predictions = model(mfgs, inputs)
                                                loss = loss_criterion_weighted(predictions, labels)
                                                opt.zero_grad(set_to_none=True)
                                                loss.backward()
                                                opt.step()
                                        else:
                                                opt.zero_grad(set_to_none=True)
                                                with torch.cuda.amp.autocast():
                                                        predictions = model(mfgs, inputs)
                                                        loss = loss_criterion_weighted(predictions, labels)

                                                scaler.scale(loss).backward()
                                                scaler.step(opt)
                                                scaler.update()
                                                

                                        accuracy = sklearn.metrics.accuracy_score(labels.cpu().numpy(), predictions.argmax(1).detach().cpu().numpy())
                                        weighted_f1 = sklearn.metrics.f1_score(labels.cpu().numpy(), predictions.argmax(1).detach().cpu().numpy(), average='weighted', zero_division=0)
                                        macro_f1 = sklearn.metrics.f1_score(labels.cpu().numpy(), predictions.argmax(1).detach().cpu().numpy(), average='macro', zero_division=0)
                                        micro_f1 = sklearn.metrics.f1_score(labels.cpu().numpy(), predictions.argmax(1).detach().cpu().numpy(), average='micro', zero_division=0)
                                        weighted_precision = sklearn.metrics.precision_score(labels.cpu().numpy(), predictions.argmax(1).detach().cpu().numpy(), average='weighted', zero_division=0)
                                        macro_precision = sklearn.metrics.precision_score(labels.cpu().numpy(), predictions.argmax(1).detach().cpu().numpy(), average='macro', zero_division=0)
                                        micro_precision = sklearn.metrics.precision_score(labels.cpu().numpy(), predictions.argmax(1).detach().cpu().numpy(), average='micro', zero_division=0)
                                        weighted_recall = sklearn.metrics.recall_score(labels.cpu().numpy(), predictions.argmax(1).detach().cpu().numpy(), average='weighted', zero_division=0)
                                        macro_recall = sklearn.metrics.recall_score(labels.cpu().numpy(), predictions.argmax(1).detach().cpu().numpy(), average='macro', zero_division=0)
                                        micro_recall = sklearn.metrics.recall_score(labels.cpu().numpy(), predictions.argmax(1).detach().cpu().numpy(), average='micro', zero_division=0)

                                        if dev_id == 0:
                                                tq.set_postfix({'loss': '%.03f' % loss.item(), 'acc': '%.03f' % accuracy, 'weighted f1': '%.03f' % weighted_f1, 'macro f1': '%.03f' % macro_f1}, refresh=True)

                                        epoch_avg_loss += loss.item()
                                        epoch_avg_accuracy += accuracy
                                        epoch_avg_weighted_precision += weighted_precision
                                        epoch_avg_macro_precision += macro_precision
                                        epoch_avg_micro_precision += micro_precision
                                        epoch_avg_weighted_recall += weighted_recall
                                        epoch_avg_macro_recall += macro_recall
                                        epoch_avg_micro_recall += micro_recall
                                        epoch_avg_weighted_f1 += weighted_f1
                                        epoch_avg_macro_f1 += macro_f1
                                        epoch_avg_micro_f1 += micro_f1

                                epoch_avg_loss /= len(tq)
                                epoch_avg_accuracy /= len(tq)
                                epoch_avg_weighted_precision /= len(tq)
                                epoch_avg_macro_precision /= len(tq)
                                epoch_avg_micro_precision /= len(tq)
                                epoch_avg_weighted_recall /= len(tq)
                                epoch_avg_macro_recall /= len(tq)
                                epoch_avg_micro_recall /= len(tq)
                                epoch_avg_weighted_f1 /= len(tq)
                                epoch_avg_macro_f1 /= len(tq)
                                epoch_avg_micro_f1 /= len(tq)

                                if dev_id == 0:
                                        print("Epoch {} Train avg. loss: {}, avg. accuracy: {}, avg. weighted F1: {}, avg. macro F1: {}, avg. micro F1: {}, avg. weighted precision: {}, avg. macro precision: {}, avg. micro precision: {}, avg. weighted recall: {}, avg. macro recall: {}, avg. micro recall: {}".format(epoch, epoch_avg_loss, epoch_avg_accuracy, epoch_avg_weighted_f1, epoch_avg_macro_f1, epoch_avg_micro_f1, epoch_avg_weighted_precision, epoch_avg_macro_precision, epoch_avg_micro_precision, epoch_avg_weighted_recall, epoch_avg_macro_recall, epoch_avg_micro_recall))
                                
                                train_epoch_avg_loss = epoch_avg_loss
                                train_epoch_avg_accuracy = epoch_avg_accuracy
                                train_epoch_avg_weighted_f1 = epoch_avg_weighted_f1

                        model.eval()

                        softmax_predictions = []
                        predictions = []
                        labels = []
                        with tqdm.tqdm(test_dataloader, disable=plot_progress_bar) as tq_test, torch.no_grad():
                                epoch_avg_loss = 0.0

                                for x in tq_test:
                                        mfgs = x[2]
                                        inputs = mfgs[0].srcdata['features'].float().to(device)
                                        label = mfgs[-1].dstdata['label'].long().to(device)

                                        labels.append(label.cpu().numpy())

                                        if args.use_AMP is False:
                                                test_prediction = model(mfgs, inputs).to(device)
                                                test_loss = loss_criterion_weighted(test_prediction, label)
                                        else:
                                                with torch.cuda.amp.autocast():
                                                        test_prediction = model(mfgs, inputs).to(device)
                                                        test_loss = loss_criterion_weighted(test_prediction, label)

                                        softmax_predictions.append(test_prediction.cpu().numpy())
                                        predictions.append(test_prediction.argmax(1).cpu().numpy())
                                        epoch_avg_loss += test_loss.item()

                                if dev_id == 0: # Main process calculates the loss and metrics
                                        epoch_avg_loss /= len(tq_test)
                                        softmax_predictions = np.concatenate(softmax_predictions)
                                        predictions = np.concatenate(predictions)
                                        labels = np.concatenate(labels)

                                        accuracy = sklearn.metrics.accuracy_score(labels, predictions)
                                        weighted_precision = sklearn.metrics.precision_score(labels, predictions, average='weighted', zero_division=0)
                                        macro_precision = sklearn.metrics.precision_score(labels, predictions, average='macro', zero_division=0)
                                        micro_precision = sklearn.metrics.precision_score(labels, predictions, average='micro', zero_division=0)
                                        weighted_recall = sklearn.metrics.recall_score(labels, predictions, average='weighted', zero_division=0)
                                        macro_recall = sklearn.metrics.recall_score(labels, predictions, average='macro', zero_division=0)
                                        micro_recall = sklearn.metrics.recall_score(labels, predictions, average='micro', zero_division=0)
                                        weighted_f1 = sklearn.metrics.f1_score(labels, predictions, average='weighted', zero_division=0)
                                        macro_f1 = sklearn.metrics.f1_score(labels, predictions, average='macro', zero_division=0) #https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f#33e1
                                        micro_f1 = sklearn.metrics.f1_score(labels, predictions, average='micro', zero_division=0)

                                        print('Epoch {} Test avg. loss: {}, accuracy: {}, weighted F1: {}, macro F1: {}, micro F1: {}, weighted precision: {}, macro precision: {}, micro precision: {}, weighted recall: {}, macro recall: {}, micro recall: {}'.format(epoch, epoch_avg_loss, accuracy, weighted_f1, macro_f1, micro_f1, weighted_precision, macro_precision, micro_precision, weighted_recall, macro_recall, micro_recall))

                                        softmax_predictions_all_epochs.append(softmax_predictions)
                                        predictions_all_epochs.append(predictions)
                                        labels_all_epochs.append(labels)
                                        losses_all_epochs.append(epoch_avg_loss)

                                        # Save log data (softmax predictions, labels and losses)
                                        with open(improve_path('./log/' + save_folder_path + '/test_log_softmax_preds_per_epoch_run_' + str(run + 1) + '.pickle'), 'wb') as handle: #previous: 'test_log_softmax_predictions_per_epoch_run_'
                                                pickle.dump(softmax_predictions_all_epochs, handle, protocol=pickle.HIGHEST_PROTOCOL)

                                        with open(improve_path('./log/' + save_folder_path + '/test_log_preds_per_epoch_run_' + str(run + 1) + '.pickle'), 'wb') as handle:
                                                pickle.dump(predictions_all_epochs, handle, protocol=pickle.HIGHEST_PROTOCOL)

                                        with open(improve_path('./log/' + save_folder_path + '/test_log_labels_per_epoch_run_' + str(run + 1) + '.pickle'), 'wb') as handle:
                                                pickle.dump(labels_all_epochs, handle, protocol=pickle.HIGHEST_PROTOCOL)

                                        with open(improve_path('./log/' + save_folder_path + '/test_log_losses_per_epoch_run_' + str(run + 1) + '.pickle'), 'wb') as handle:
                                                pickle.dump(losses_all_epochs, handle, protocol=pickle.HIGHEST_PROTOCOL)

                                        torch.save(model.state_dict(), improve_path('./models/' + save_folder_path + '/final_model_run_' + str(run + 1) + '.pt'))

                                        final_loss = epoch_avg_loss
                                        final_accuracy = accuracy
                                        final_weighted_precision = weighted_precision
                                        final_macro_precision = macro_precision
                                        final_micro_precision = micro_precision
                                        final_weighted_recall = weighted_recall
                                        final_macro_recall = macro_recall
                                        final_micro_recall = micro_recall
                                        final_weighted_f1 = weighted_f1
                                        final_macro_f1 = macro_f1
                                        final_micro_f1 = micro_f1

                                        with open(improve_path('./log/' + save_folder_path + '/training_progress.txt'), 'a') as file:
                                                file.write("Run: %s" % (run + 1))
                                                file.write(", epoch: %s " % (epoch + 1))

                                                total_epoch_time = time.time() - start_epoch_time

                                                epoch_seconds = total_epoch_time
                                                epoch_minutes = epoch_seconds // 60
                                                epoch_hours = epoch_minutes // 60

                                                file.write(", epoch elapsed time: %02d:%02d:%02d" % (epoch_hours, epoch_minutes % 60, epoch_seconds % 60))

                                                total_training_time = time.time() - start_training_time

                                                total_training_seconds = total_training_time
                                                total_training_minutes = total_training_seconds // 60
                                                total_training_hours = total_training_minutes // 60

                                                file.write(", total training elapsed time: %02d:%02d:%02d" % (total_training_hours, total_training_minutes % 60, total_training_seconds % 60))
                                                file.write(" | TRAIN loss: " + str(train_epoch_avg_loss) + ", accuracy: " + str(train_epoch_avg_accuracy) + ", weighted F1: " + str(train_epoch_avg_weighted_f1))
                                                file.write(" | VALIDATION loss: " + str(final_loss) + ", accuracy: " + str(final_accuracy) + ", weighted F1: " + str(final_weighted_f1))
                                                file.write("\n")
                                        
                                                if final_loss < best_val_loss:
                                                        best_val_loss = final_loss
                                                        early_stopping_patience_counter = 0
                                                        best_epoch_index = epoch + 1
                                                        torch.save(model.state_dict(), improve_path('./models/' + save_folder_path + '/es_best_model_run_' + str(run + 1) + '.pt'))
                                                        file.write("Validation loss improved at epoch #" + str(best_epoch_index) + "! Validation loss of " + str(best_val_loss) + " achieved! Best model saved.\n")
                                                else:
                                                        early_stopping_patience_counter += 1
                                                        file.write("No improvement in validation loss. Patience counter:" + str(early_stopping_patience_counter) + "/" + str(args.early_stopping_patience) + "\n")

                run_losses.append(final_loss)
                run_accuracies.append(final_accuracy)
                run_weighted_precisions.append(final_weighted_precision)
                run_macro_precisions.append(final_macro_precision)
                run_micro_precisions.append(final_micro_precision)
                run_weighted_recalls.append(final_weighted_recall)
                run_macro_recalls.append(final_macro_recall)
                run_micro_recalls.append(final_micro_recall)
                run_weighted_f1s.append(final_weighted_f1)
                run_macro_f1s.append(final_macro_f1)
                run_micro_f1s.append(final_micro_f1)
                run_best_losses.append(best_val_loss)


        if dev_id == 0:
                run_losses = np.array(run_losses)
                run_accuracies = np.array(run_accuracies)
                run_weighted_precisions = np.array(run_weighted_precisions)
                run_macro_precisions = np.array(run_macro_precisions)
                run_micro_precisions = np.array(run_micro_precisions)
                run_weighted_recalls = np.array(run_weighted_recalls)
                run_macro_recalls = np.array(run_macro_recalls)
                run_micro_recalls = np.array(run_micro_recalls)
                run_weighted_f1s = np.array(run_weighted_f1s)
                run_macro_f1s = np.array(run_macro_f1s)
                run_micro_f1s = np.array(run_micro_f1s)
                run_best_losses = np.array(run_best_losses)

                average_run_losses = np.mean(run_losses)
                average_run_accuracies = np.mean(run_accuracies)
                average_run_weighted_precisions = np.mean(run_weighted_precisions)
                average_run_macro_precisions = np.mean(run_macro_precisions)
                average_run_micro_precisions = np.mean(run_micro_precisions)
                average_run_weighted_recalls = np.mean(run_weighted_recalls)
                average_run_macro_recalls = np.mean(run_macro_recalls)
                average_run_micro_recalls = np.mean(run_micro_recalls)
                average_run_weighted_f1s = np.mean(run_weighted_f1s)
                average_run_macro_f1s = np.mean(run_macro_f1s)
                average_run_micro_f1s = np.mean(run_micro_f1s)
                average_run_best_losses = np.mean(run_best_losses)

                std_run_losses = np.std(run_losses)
                std_run_accuracies = np.std(run_accuracies)
                std_run_weighted_precisions = np.std(run_weighted_precisions)
                std_run_macro_precisions = np.std(run_macro_precisions)
                std_run_micro_precisions = np.std(run_micro_precisions)
                std_run_weighted_recalls = np.std(run_weighted_recalls)
                std_run_macro_recalls = np.std(run_macro_recalls)
                std_run_micro_recalls = np.std(run_micro_recalls)
                std_run_weighted_f1s = np.std(run_weighted_f1s)
                std_run_macro_f1s = np.std(run_macro_f1s)
                std_run_micro_f1s = np.std(run_micro_f1s)
                std_run_best_losses = np.std(run_best_losses)

                # Summarize metrics across runs (mean ± std) and save
                with open(improve_path('./log/' + save_folder_path + '/testing_report.txt'), 'a') as file:

                        file.write("Losses by runs: [")
                        for x in run_losses:
                                file.write("%s " % x)
                        file.write("]\n")

                        file.write("Accuracies by runs: [")
                        for x in run_accuracies:
                                file.write("%s " % x)
                        file.write("]\n")

                        file.write("Weighted precisions by runs: [")
                        for x in run_weighted_precisions:
                                file.write("%s " % x)
                        file.write("]\n")

                        file.write("Macro precisions by runs: [")
                        for x in run_macro_precisions:
                                file.write("%s " % x)
                        file.write("]\n")

                        file.write("Micro precisions by runs: [")
                        for x in run_micro_precisions:
                                file.write("%s " % x)
                        file.write("]\n")

                        file.write("Weighted recalls by runs: [")
                        for x in run_weighted_recalls:
                                file.write("%s " % x)
                        file.write("]\n")

                        file.write("Macro recalls by runs: [")
                        for x in run_macro_recalls:
                                file.write("%s " % x)
                        file.write("]\n")

                        file.write("Micro recalls by runs: [")
                        for x in run_micro_recalls:
                                file.write("%s " % x)
                        file.write("]\n")

                        file.write("Weighted F1s by runs: [")
                        for x in run_weighted_f1s:
                                file.write("%s " % x)
                        file.write("]\n")

                        file.write("Macro F1s by runs: [")
                        for x in run_macro_f1s:
                                file.write("%s " % x)
                        file.write("]\n")

                        file.write("Micro F1s by runs: [")
                        for x in run_micro_f1s:
                                file.write("%s " % x)
                        file.write("]\n")

                        file.write("Best losses (at early stopping) by runs: [")
                        for x in run_best_losses:
                                file.write("%s " % x)
                        file.write("]\n")

                        file.write("\n\n\n\n")
                        file.write("Average loss: " + str(average_run_losses) + " with standard deviation of: " + str(std_run_losses))
                        file.write("\nAverage accuracy: " + str(average_run_accuracies) + " with standard deviation of: " + str(std_run_accuracies))
                        file.write("\nAverage weighted precision: " + str(average_run_weighted_precisions) + " with standard deviation of: " + str(std_run_weighted_precisions))
                        file.write("\nAverage macro precision: " + str(average_run_macro_precisions) + " with standard deviation of: " + str(std_run_macro_precisions))
                        file.write("\nAverage micro precision: " + str(average_run_micro_precisions) + " with standard deviation of: " + str(std_run_micro_precisions))
                        file.write("\nAverage weighted recall: " + str(average_run_weighted_recalls) + " with standard deviation of: " + str(std_run_weighted_recalls))
                        file.write("\nAverage macro recall: " + str(average_run_macro_recalls) + " with standard deviation of: " + str(std_run_macro_recalls))
                        file.write("\nAverage micro recall: " + str(average_run_micro_recalls) + " with standard deviation of: " + str(std_run_micro_recalls))
                        file.write("\nAverage weighted F1: " + str(average_run_weighted_f1s) + " with standard deviation of: " + str(std_run_weighted_f1s))
                        file.write("\nAverage macro F1: " + str(average_run_macro_f1s) + " with standard deviation of: " + str(std_run_macro_f1s))
                        file.write("\nAverage micro F1: " + str(average_run_micro_f1s) + " with standard deviation of: " + str(std_run_micro_f1s))
                        file.write("\nAverage best loss (at early stopping): " + str(average_run_best_losses) + " with standard deviation of: " + str(std_run_best_losses))
                        file.write("\n")

        print("Process number", dev_id, "done!")

if __name__ == '__main__':
        print("Starting training...")
        main()