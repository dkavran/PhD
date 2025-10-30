import cv2
import numpy as np
from skimage.segmentation import quickshift, felzenszwalb, slic
import math
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
from skimage.feature import hog
import traceback

import pickle
import copy

from skimage.segmentation import mark_boundaries
from scipy.spatial import distance

import gc
import multiprocessing

# Implemented imports
from .cld import divide_img_blocks, calculate_avg_value_in_block, dct2, idct2, array2d_to_zigzag

def get_indices_and_labels_for_time_frame_without_labels(t_indices, t_frame_index_start, t_frame_index_end, n_segments, notification_counter = 5000):
    indices = []
    for i, index in enumerate(t_indices):
        if i % notification_counter == 0:
            print("Analysing index #", i, "/", len(t_indices))
        image_index, _ = get_image_index_and_real_segment_index(index, n_segments)
        if t_frame_index_start <= image_index and image_index <= t_frame_index_end:
            indices.append(index)

    return np.array(indices)


def save_item_with_pickle(item, path):
    with open(path, 'wb') as handle:
        pickle.dump(item, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_item_with_pickle(path):
    data = None
    with open(path, "rb") as input_file:
        data = pickle.load(input_file)
    return data

def segment_images(img, ratio = 1.0, kernel_size = 3, max_dist = 100, return_tree = False, sigma = 0.5, convert2lab = False, rng = 42, longest_image_dimension = 1024, k=70, beta=0.85, epsilon=0, use_quickshift_plus_plus = False, use_felzenszwalb = False, use_SLIC = False, scale=1, min_size=20, n_segments=None, compactness=None, max_num_iter=None):
    segmentation_masks = None
    segmentation_masks_num_of_segments = []

    if use_quickshift_plus_plus is False and use_felzenszwalb is False and use_SLIC is False:
        print("Using algorithm Quickshift (regular)")
    elif use_felzenszwalb is True:
        print("Using Felzenszwalb algorithm")
    elif use_SLIC is True:
        print("Using algorithm SLIC")
    else:
        print("Using algorithm Quickshift++")

    for i in range(0, img.shape[0]): # Img shape: (time, X, Y, channels)
        if i % 25 == 0:
            print("Segmenting image #", (i+1))

        segmentation_mask = None
        if use_quickshift_plus_plus is False and use_felzenszwalb is False and use_SLIC is False:
            segmentation_mask = quickshift(img[i,:,:,:].astype(np.double), ratio=ratio, kernel_size=kernel_size, max_dist=max_dist, return_tree=return_tree, sigma=sigma, convert2lab=convert2lab, random_seed=rng)
        elif use_felzenszwalb is True:
            segmentation_mask = felzenszwalb(img[i,:,:,:].astype(np.double), scale=scale, sigma=sigma, min_size=min_size, channel_axis=2)
        elif use_SLIC is True:
            segmentation_mask = slic(img[i,:,:,:].astype(np.double), convert2lab=convert2lab, sigma=sigma, n_segments=n_segments, compactness=compactness, max_num_iter=max_num_iter)
        else:
            segmentation_mask = quickshift_plus_plus(img[i,:,:,:], longest_image_dimension, k=k, beta=beta, epsilon=epsilon)

            print("Segmentation mask has", np.unique(segmentation_mask).shape[0], "segments", "-", "min label:", segmentation_mask.min(), ", max label:", segmentation_mask.max())

            # Because Quickshift can have labels with empty space between (in at least 2 segments), we further dissect those regions with cv2.connectedComponents algorithm
            segmentation_mask_labels = np.unique(segmentation_mask)
            segmentation_mask_corrected = np.zeros(segmentation_mask.shape, dtype=segmentation_mask.dtype)
            
            new_label = 0
            segm_mask_with_only_label = None
            num_labels = None
            labels_im = None

            for l in segmentation_mask_labels:
                segm_mask_with_only_label = np.zeros(segmentation_mask.shape, dtype=np.uint8)
                segm_mask_with_only_label[segmentation_mask == l] = 1

                num_labels, labels_im = cv2.connectedComponents(segm_mask_with_only_label)
                connected_components_labels = np.unique(labels_im)[1:] # Remove the background on index 0
                for cc_l in connected_components_labels:
                    segmentation_mask_corrected[labels_im == cc_l] = new_label
                    new_label += 1

            segmentation_mask = segmentation_mask_corrected

            print("Corrected segmentation mask has", np.unique(segmentation_mask).shape[0], "segments", "-", "min label:", segmentation_mask.min(), ", max label:", segmentation_mask.max())

        if segmentation_masks is None:
            segmentation_masks = np.zeros((img.shape[0], segmentation_mask.shape[0], segmentation_mask.shape[1]), dtype=np.uint16)

        segmentation_masks[i,:,:] = segmentation_mask
    
        unique= np.unique(segmentation_mask) # Number of segments
        segmentation_masks_num_of_segments.append(unique.shape[0])
    
    segmentation_masks = np.array(segmentation_masks)
    segmentation_masks_num_of_segments = np.array(segmentation_masks_num_of_segments)

    return segmentation_masks, segmentation_masks_num_of_segments

def plot_nx_graph(G, node_color_map = None):
    pos = graphviz_layout(G, prog="dot")

    node_labels = list(nx.get_node_attributes(G, 'label').values())
    node_changes = list(nx.get_node_attributes(G, 'change').values())
    node_indices = list(G.nodes)

    if len(node_labels) and node_labels[0] is not None and node_color_map is not None:
        node_colors = []
        for node_label in node_labels:
            node_colors.append(node_color_map[node_label])

        no_change_node_indices = []
        no_change_node_colors = []
        change_node_indices = []
        change_node_colors = []
        for i, node_change in enumerate(node_changes):
            if node_change == True:
                change_node_indices.append(node_indices[i])
                change_node_colors.append(node_colors[i])
            else:
                no_change_node_indices.append(node_indices[i])
                no_change_node_colors.append(node_colors[i])

        nx.draw_networkx_nodes(G, pos, nodelist=no_change_node_indices, node_size=650, node_color=no_change_node_colors, edgecolors='black', node_shape='o')
        nx.draw_networkx_nodes(G, pos, nodelist=change_node_indices, node_size=650, node_color=change_node_colors, edgecolors='black', node_shape='s')
    else:
        nx.draw_networkx_nodes(G, pos, node_size=500)

    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)


    index_labels = nx.get_node_attributes(G, 'index')
    for p in pos:
        pos_copy = list(pos[p])
        pos_copy[0] += 8
        pos[p] = tuple(pos_copy)
    nx.draw_networkx_labels(G, pos, labels=index_labels)

    # Plot edge labels with time information; Specifically, the information of time index 't'
    edge_labels = nx.get_edge_attributes(G, 't')
    nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)

    plt.show()

def save_graphs_to_file(graphs, root_path, prefix, labels = None):
    import pandas as pd

    paths = []
    for i, graph in enumerate(graphs):
        path = root_path + "/" + prefix + "_" + str(i) + '.pickle'
        with open(path, 'wb') as handle:
            pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        paths.append(path)
    
    pandas_data = {}
    pandas_data['graph_path'] = paths
    
    if labels is not None:
        pandas_data['label'] = labels
            
    df = pd.DataFrame(data=pandas_data)
    df.to_csv(root_path + "/" + prefix + "_data.csv", index=False, sep=";")