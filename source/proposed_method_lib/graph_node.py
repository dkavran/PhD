import cv2
import numpy as np
from skimage.segmentation import quickshift, felzenszwalb
import math
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
from skimage.feature import hog
import traceback
import pandas as pd

import pickle
import copy

from skimage.segmentation import mark_boundaries
from scipy.spatial import distance

import gc
import multiprocessing

import tensorflow as tf

# Implemented imports
from .cld import divide_img_blocks, calculate_avg_value_in_block, dct2, idct2, array2d_to_zigzag
from .graph import get_image_index_and_real_segment_index

def find_adjacent_segments(mask, min_node_value):
    """Returns a dictionary where keys are segment values, and values are sets of adjacent segment values."""
    height, width = mask.shape
    adjacent = {}

    # Directions for neighbors: top, right, bottom, left, top-left, top-right, bottom-right, bottom-left
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, -1), (-1, 1), (1, 1), (1, -1)]

    for i in range(height):
        for j in range(width):
            curr_segment = mask[i, j] + min_node_value
            if curr_segment not in adjacent:
                adjacent[curr_segment] = set()

            for dx, dy in directions:
                ni, nj = i + dx, j + dy
                if 0 <= ni < height and 0 <= nj < width:  # Check if the neighbor is inside the image
                    neighbor_segment = mask[ni, nj] + min_node_value
                    if curr_segment != neighbor_segment:
                        adjacent[curr_segment].add(neighbor_segment)

    # Convert sets to lists for better readability
    for key, value in adjacent.items():
        adjacent[key] = list(value)

    return adjacent

# Add spatial connections with type: 'spatial'
def add_spatial_connections_to_graphs(graphs, segmentation_masks, segmentation_masks_num_of_segments):
    print("Working on calculating adjacent segments in all segmentation masks...")
    segmentation_mask_adjacent_segments_ind = []
    min_segment_value = 0
    for i, segmentation_mask in enumerate(segmentation_masks):
        print("Working on segmentation mask #", (i+1), "/", len(segmentation_masks))
        segmentation_mask_adjacent_segments_ind.append(find_adjacent_segments(segmentation_mask, min_segment_value))
        min_segment_value += segmentation_masks_num_of_segments[i]
        
        
    # Join all dictionaries
    segmentation_mask_adjacent_segments = {}
    for d in segmentation_mask_adjacent_segments_ind:
        segmentation_mask_adjacent_segments.update({k: v for k, v in d.items() if k not in segmentation_mask_adjacent_segments})
    
    print("Working on adding spatial edges to graphs...")
    graph_nodes = []
    for graph in graphs:
        nodes = list(graph.nodes())
        graph_nodes.append(nodes)
    
    for g_index, nodes in enumerate(graph_nodes):
        for node in nodes:
            adjacent_segments = segmentation_mask_adjacent_segments[node]
            spatial_edges = [(node, adj_segment, {'type': 'spatial'}) for adj_segment in adjacent_segments]
            graphs[g_index].add_edges_from(spatial_edges)
    
    return graphs

def create_time_series_graphs(segmentation_masks, segmentation_masks_num_of_segments, split_percantage=0.0, classification_masks = None):
    connections = []
    weights = []
    print("Number of time frames:", segmentation_masks.shape[0])

    for i in range(0, segmentation_masks.shape[0] - 1): # Iterate over time
        print("Working on time frame #", (i+1))

        segmentation_mask = segmentation_masks[i]
        binary_j_mask = np.array(segmentation_mask, dtype=segmentation_mask.dtype)

        connection_array = []
        weight_array = []
        
        for j in range(1, segmentation_masks_num_of_segments[i] + 1):
            conn_array = []
            wei_array = []

            binary_j_mask[:,:] = 0
            binary_j_mask[segmentation_mask == j] = 1
            
            binary_j_mask = binary_j_mask * segmentation_masks[i+1]
            
            unique, counts = np.unique(binary_j_mask, return_counts=True)
            index_of_zero = np.where(unique==0)[0][0]
            unique = np.delete(unique, index_of_zero)
            counts = np.delete(counts, index_of_zero)

            num_uniqs = []
            for uniq in unique:
                num_uniq = len(segmentation_masks[i+1][segmentation_masks[i+1]==uniq].ravel())
                num_uniqs.append(num_uniq)
            num_uniqs = np.array(num_uniqs, dtype=np.float32)


            counts = counts / num_uniqs
            indices_of_connections = np.argwhere(counts >= split_percantage)

            j_single_class_classification = True
            j_class = None
            if classification_masks is not None:
                j_single_class_classification, j_class = is_segment_classified_with_only_one_class(classification_masks[i], segmentation_masks[i], j)
                if j_single_class_classification is False:
                    continue


            for k, index in enumerate(indices_of_connections):
                label = unique[index][0]
                conn_array.append(label - 1)

                weight = counts[index][0]
                wei_array.append(weight)


            connection_array.append(conn_array)
            weight_array.append(wei_array)

        connections.append(connection_array)
        weights.append(weight_array)

    # Append connections for nodes/regions in last image
    last_time_frame_connection_array = []
    last_time_frame_weight_array = []
    for j in range(1, segmentation_masks_num_of_segments[segmentation_masks.shape[0] - 1] + 1):
        last_time_frame_connection_array.append([])
        last_time_frame_weight_array.append([])

    connections.append(last_time_frame_connection_array)
    weights.append(last_time_frame_weight_array)
    
    return connections, weights

def is_segment_classified_with_only_one_class(classification_mask, segmentation_mask, x_label):
    j_classification_mask = np.array(classification_mask)
    j_classification_mask[segmentation_mask != x_label] = -20

    unique = np.unique(j_classification_mask)

    index_of_zero = np.where(unique==-20)[0][0]
    unique = np.delete(unique, index_of_zero)

    is_one_class = (unique.shape[0] == 1)
    one_class = unique[0]

    return (unique.shape[0] == 1), one_class


def is_segment_classified_with_only_one_class_fast(classification_mask, segmentation_mask, x_label):    
    all_labels = classification_mask[segmentation_mask == x_label]
    unique = pd.unique(all_labels.ravel())
    is_one_class = (unique.shape[0] == 1)
    one_class = unique[0]

    return (unique.shape[0] == 1), one_class

def get_segment_classification(classification_mask, segmentation_mask, x_label):
    all_labels = classification_mask[segmentation_mask == x_label]
    is_one_class = True
    one_class = all_labels[0]
    return is_one_class, one_class

def get_segment_raw_image(segmentation_mask, raw_image, segment_index, desired_size=64, mask_segment_with_zeros = True, around_region_padding_amount = 0):
    from .image import resize_image

    raw_image_only_segment = np.array(raw_image, dtype=raw_image.dtype)
    if mask_segment_with_zeros is True:
        raw_image_only_segment[segmentation_mask != segment_index, :] = 0

    if around_region_padding_amount > 0:
        paddings = tf.constant([[around_region_padding_amount, around_region_padding_amount], [around_region_padding_amount, around_region_padding_amount], [0,0]])
        raw_image_only_segment = tf.pad(raw_image_only_segment, paddings, "CONSTANT", constant_values=0).numpy()

    where_segment_indices = np.where(segmentation_mask == segment_index)
    x_min = where_segment_indices[0].min()
    x_max = where_segment_indices[0].max()
    y_min = where_segment_indices[1].min()
    y_max = where_segment_indices[1].max()

    if around_region_padding_amount > 0:
        # Shift for padding for not going out of bounds
        x_min += around_region_padding_amount
        x_max += around_region_padding_amount
        y_min += around_region_padding_amount
        y_max += around_region_padding_amount

        # Strething for padding
        x_min -= around_region_padding_amount
        x_max += around_region_padding_amount
        y_min -= around_region_padding_amount
        y_max += around_region_padding_amount

    raw_segment_image = raw_image_only_segment[x_min:x_max+1,y_min:y_max+1,:]

    if desired_size is not None:
        raw_segment_image_resized, _ = resize_image(raw_segment_image, desired_size=desired_size)
        return raw_segment_image_resized
    else:
        return raw_segment_image

# Calculate features of each bounding box (if feature_type == 'raw', the bounding box is just raveled)
def get_features(img, feature_type, segmentation_mask, raw_image, segment_index, bow_dictionaries = None, dictionary_size = 32, dtype=None):
    feature_types = ["hog", "cld", "histogram", "channel_avg", "sq_channel_avg", "channel_median", "channel_mode", "bow", "raw"]
    if feature_type not in feature_types:
        raise Exception("Incorrect feature type")

    node_features = None
    if feature_type == "hog": # Histogram of Oriented Gradients features
        node_features = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=False, multichannel=True, feature_vector=True)

    elif feature_type == "cld": # Color layout descriptor using DCT from: https://en.wikipedia.org/wiki/Color_layout_descriptor
        block_mesh = (8, 8)

        blocks = divide_img_blocks(img, block_mesh)

        # Calculate avg. pixel value in block
        tiny_image = np.zeros((block_mesh[0], block_mesh[1], img.shape[2]), dtype=img.dtype)
        for i in range(blocks.shape[0]):
            for j in range(blocks.shape[1]):
                block = blocks[i, j]
                tiny_image[i,j] = calculate_avg_value_in_block(block, axis=(0,1))

        # Run 2D dct
        dct_tiny_image = np.zeros(tiny_image.shape, dtype=tiny_image.dtype)
        for ch in range(0, tiny_image.shape[2]):
            dct_tiny_image[:,:,ch] = dct2(tiny_image[:,:,ch])

        # Zig-zag read all DCT channels
        zig_zags = []
        for ch in range(0, dct_tiny_image.shape[2]):
            zig_zags.append(array2d_to_zigzag(dct_tiny_image[:,:,ch]))

        zig_zags = np.array(zig_zags).ravel()

        # Create feature vector
        node_features = zig_zags

    elif feature_type == "histogram": # Get histogram of value for each channel
        channel_histograms = []
        for ch in range(0, raw_image.shape[2]):
            raw_data = raw_image[:,:,ch][segmentation_mask == segment_index]

            hist, _ = np.histogram(raw_data, bins=100, range=(0.0, 1.0), density=True)
            hist /= hist.sum()
            channel_histograms.append(hist)

        node_features = np.array(channel_histograms).ravel()

    elif feature_type == "channel_avg": # Get average pixel value of each channel
        channel_avgs = []
        for ch in range(0, raw_image.shape[2]):
            raw_data = raw_image[:,:,ch][segmentation_mask == segment_index]
            channel_avgs.append(np.mean(raw_data))

        node_features = np.array(channel_avgs).ravel()

    elif feature_type == "sq_channel_avg": # RMS for each channel
        channel_avgs = []
        for ch in range(0, raw_image.shape[2]):
            raw_data = raw_image[:,:,ch][segmentation_mask == segment_index]
            channel_avgs.append(np.sqrt(np.mean(np.power(raw_data, 2))).real)

        node_features = np.array(channel_avgs).ravel()

    elif feature_type == "channel_median": # Get median pixel value for each channel
        channel_medians = []
        for ch in range(0, raw_image.shape[2]):
            raw_data = raw_image[:,:,ch][segmentation_mask == segment_index]
            channel_medians.append(np.median(raw_data))

        node_features = np.array(channel_medians).ravel()

    elif feature_type == "channel_mode": # Get most frequent value for each channel
        channel_modes = []
        for ch in range(0, raw_image.shape[2]):
            raw_data = raw_image[:,:,ch][segmentation_mask == segment_index]
            raw_data = np.round(raw_data, 2)

            vals, counts = np.unique(raw_data, return_counts=True)
            index_most_common = np.argmax(counts)

            channel_modes.append(vals[index_most_common])

        node_features = np.array(channel_modes).ravel()

    elif feature_type == "bow": # Bag of words
        ch_bow_features = []

        sift = cv2.SIFT_create(contrastThreshold=0.001)

        for ch in bow_dictionaries.keys():
            ch_histogram = np.zeros(dictionary_size, dtype=np.int32)
            gray = (img[:,:,ch] * 255.0).astype(np.uint8)
            keypoints, descriptors = sift.detectAndCompute(gray, None)

            if descriptors is not None:
                bin_indices = bow_dictionaries[ch].predict(descriptors)
                for index in bin_indices:
                    ch_histogram[index] += 1

            ch_bow_features.append(ch_histogram)

        node_features = np.array(ch_bow_features).ravel()

        for ch in range(0, raw_image.shape[2]): # Also add averages for each channel
            raw_data = (raw_image[:,:,ch][segmentation_mask == segment_index] * 255.0).astype(np.uint8)
            node_features = np.append(node_features, np.mean(raw_data))

    elif feature_type == "raw":
        node_features = img.ravel()

    if dtype is not None:
        node_features = node_features.astype(dtype)

    return node_features

def get_all_possible_graphs_fast(connections, weights, segmentation_masks, segmentation_masks_num_of_segments, raw = None, classification_masks = None, feature_type = "hog", mask_segments_with_zeros = True, around_region_padding_amount = 0):

    bow_dictionaries = None
    
    if feature_type == 'bow':
        bow_dictionaries = get_bow_dictionaries(segmentation_masks, raw)

    print("Graphs will have", feature_type, "features")

    mask_connections = []
    for i in range(0, len(connections)):
        mask_connections.append(np.ones(len(connections[i])))
    
    graphs = []
    labels = []
    for i in range(0, len(mask_connections)):
        print("Working on time frame #", i)
        for j in range(0, mask_connections[i].shape[0]):
            if j % 200 == 0:
                print("Working on segment #", j, "/", mask_connections[i].shape[0])
            if mask_connections[i][j] == 1: # If connection was not used yet
                try:
                    if classification_masks is not None:
                        G = get_nx_graph_fast(connections, weights, mask_connections, segmentation_masks, segmentation_masks_num_of_segments, j, i, raw=raw, classification_masks=classification_masks, feature_type=feature_type, bow_dictionaries=bow_dictionaries, mask_segments_with_zeros = mask_segments_with_zeros, around_region_padding_amount = around_region_padding_amount)
                    else:
                        G = get_nx_graph_fast(connections, weights, mask_connections, segmentation_masks, segmentation_masks_num_of_segments, j, i, raw=raw, feature_type=feature_type, bow_dictionaries=bow_dictionaries, mask_segments_with_zeros = mask_segments_with_zeros, around_region_padding_amount = around_region_padding_amount)
                    graphs.append(G)
                except Exception as e:
                    print(e)
                    continue
                
                # Mark all graph nodes as used
                node_names = nx.get_node_attributes(G, 'time').keys()
                
                time_frames = nx.get_node_attributes(G, 'time')
                segmentation_indices = nx.get_node_attributes(G, 'index')
                for name in node_names:
                    t = time_frames[name]
                    index = segmentation_indices[name]
                    mask_connections[t][index] = 0
        
        print("Step", i, "- number of graphs created:", len(graphs))

    are_all_nodes_used = True
    for i in range(0, len(mask_connections)):
        are_all_nodes_used_in_time_frame_i = np.unique(mask_connections[i])
        if are_all_nodes_used_in_time_frame_i.shape[0] != 1:
            are_all_nodes_used = False
            break

        if are_all_nodes_used_in_time_frame_i.shape[0] == 1 and are_all_nodes_used_in_time_frame_i[0] != 0:
            are_all_nodes_used = False
            break
    if are_all_nodes_used is True:
        print("All nodes are used! Number of graphs created:", len(graphs))
        
        
    return graphs

def update_nx_graph_features(nx_graph, segmentation_masks, segmentation_masks_num_of_segments, raw = None, feature_type = "hog", bow_dictionaries = None, mask_segments_with_zeros = True, around_region_padding_amount = 0, desired_img_size=32, update_counter = 2000, dtype=None):
    node_indices = list(nx_graph.nodes)
    for i, node in enumerate(node_indices):
        if i % update_counter == 0:
            print("Updating node #", i, "/", len(node_indices))
        image_index, segment_index = get_image_index_and_real_segment_index(node, segmentation_masks_num_of_segments)

        segment_raw_image = get_segment_raw_image(segmentation_masks[image_index], raw[image_index,:,:,:], segment_index, desired_size=desired_img_size, mask_segment_with_zeros = mask_segments_with_zeros, around_region_padding_amount = around_region_padding_amount)
        features = get_features(segment_raw_image, feature_type, segmentation_masks[image_index], raw[image_index,:,:,:], segment_index, bow_dictionaries=bow_dictionaries, dtype=dtype)
        nx_graph.nodes[node].update(features=features)

    return nx_graph

def get_nx_graph_fast(connections, weights, mask_connections, segmentation_masks, segmentation_masks_num_of_segments, starting_index, starting_time, raw = None, classification_masks = None, feature_type = "hog", bow_dictionaries = None, mask_segments_with_zeros = True, around_region_padding_amount = 0):

    index = np.array([[starting_index]])

    # Create graph
    G = nx.DiGraph()

    number_of_all_nodes = 0
    for i in range(0, starting_time):
        number_of_all_nodes += segmentation_masks_num_of_segments[i]

    number_of_all_nodes_in_next_time_frame = number_of_all_nodes

    for i in range(starting_time, len(connections)):
        number_of_all_nodes_in_next_time_frame += segmentation_masks_num_of_segments[i]
        segmentation_mask = segmentation_masks[i]
        segmentation_mask_plot = np.zeros(segmentation_masks[i].shape, dtype=np.int32)
        new_index = []
        found_successor_in_time_series = False

        for idx_pair in index:
            for idx in idx_pair:
                if idx != -1:
                    # If any children are NaN -> that means they can not form further connections; in that case, this node is invalid and we try with next child node
                    if np.any(np.isnan(connections[i][int(idx)])):
                        # Remove this incorrect node
                        current_node_index = int(idx) + number_of_all_nodes
                        if G.has_node(current_node_index):
                            print("Removing node with index:", current_node_index)
                            G.remove_node(current_node_index)
                        continue

                    found_successor_in_time_series = True
                    segmentation_mask_plot[segmentation_mask == int(idx)] = int(idx) + 1
                
                    
                    # Calculate features for root node
                    root_features = None
                    root_label = None
                    if classification_masks is not None:
                        is_one_class, one_class = get_segment_classification(classification_masks[i], segmentation_masks[i], int(idx))
                        if is_one_class:
                            root_label = one_class
                        else:
                            continue

                    if raw is not None:
                        desired_img_size = 64
                        if feature_type == "cld":
                            desired_img_size = 128
                        elif feature_type == "raw":
                            desired_img_size = 32

                        segment_raw_image = get_segment_raw_image(segmentation_masks[i], raw[i,:,:,:], int(idx), desired_size=desired_img_size, mask_segment_with_zeros = mask_segments_with_zeros, around_region_padding_amount = around_region_padding_amount)
                        root_features = get_features(segment_raw_image, feature_type, segmentation_masks[i], raw[i,:,:,:], int(idx), bow_dictionaries=bow_dictionaries)

                    new_next_connections = []
                    for new_possible_index in connections[i][int(idx)]:
                        if mask_connections[i+1][new_possible_index] == 1:
                            new_next_connections.append(new_possible_index)
                            mask_connections[i+1][new_possible_index] = 0
                        
                    new_index.append(new_next_connections)
                        
                    # Add root node to graph
                    G.add_node(int(idx) + number_of_all_nodes, index=int(idx), features=root_features, label = root_label, time=i)

                    new_nodes = list(connections[i][int(idx)])
                    new_edge_weights = list(weights[i][int(idx)])

                    # Calculate features for child nodes
                    new_node_features = []
                    new_node_labels = []
                    new_nodes_correct = []
                    new_edge_weights_correct = []

                    for nd_i, node_index in enumerate(new_nodes):
                        if int(node_index) != -1:
                            node_features = None
                            node_label = None

                            if classification_masks is not None:
                                is_one_class, one_class = get_segment_classification(classification_masks[i+1], segmentation_masks[i+1], int(node_index))
                                if is_one_class:
                                    node_label = one_class
                                    new_nodes_correct.append(node_index)
                                    new_edge_weights_correct.append(new_edge_weights[nd_i])
                                else:
                                    continue
                            else:
                                new_nodes_correct.append(node_index)
                                new_edge_weights_correct.append(new_edge_weights[nd_i])

                            if raw is not None:
                                desired_img_size = 64
                                if feature_type == "cld":
                                    desired_img_size = 128
                                elif feature_type == "raw":
                                    desired_img_size = 32

                                segment_raw_image = get_segment_raw_image(segmentation_masks[i+1], raw[i+1,:,:,:], int(node_index), desired_size=desired_img_size, mask_segment_with_zeros = mask_segments_with_zeros, around_region_padding_amount = around_region_padding_amount)
                                node_features = get_features(segment_raw_image, feature_type, segmentation_masks[i+1], raw[i+1,:,:,:], int(node_index), bow_dictionaries=bow_dictionaries)

                            new_node_features.append(node_features)
                            new_node_labels.append(node_label)

                    new_nodes = [[int(n) + number_of_all_nodes_in_next_time_frame, {'index': int(n), 'features': new_node_features[n_index], 'label': new_node_labels[n_index], 'time': (i+1)}] for n_index, n in enumerate(new_nodes_correct) if n != -1]
                
                    if len(new_nodes) > 0: # Add child nodes to graph
                        G.add_nodes_from(new_nodes)
                        # Connect root with child nodes
                        new_edges = list(zip([idx + number_of_all_nodes] * len(new_nodes), [node[0] for node in new_nodes], [{'t': i, 'weight': new_edge_weights_correct[nd_i], 'type': 'temporal' } for nd_i, node in enumerate(new_nodes)]))
                        G.add_edges_from(new_edges)
       
        if found_successor_in_time_series is False:
            break
    
        index = np.array(new_index)
        number_of_all_nodes += segmentation_masks_num_of_segments[i]

    return G

def get_connected_nodes_indices(l):
    out = []
    while len(l)>0:
        first, *rest = l
        first = set(first)

        lf = -1
        while len(first)>lf:
            lf = len(first)

            rest2 = []
            for r in rest:
                if len(first.intersection(set(r)))>0:
                    first |= set(r)
                else:
                    rest2.append(r)     
            rest = rest2

        out.append(first)
        l = rest

    result = []
    for l in out:
        result.append(list(l))
    return result


def join_graphs_if_joint_nodes(graphs, check_interval = 200, graphs_checked_interval = 500):
    
    graph_nodes = []
    for i, graph in enumerate(graphs):
        tmp_graph_nodes = list(graph.nodes)
        graph_nodes.append(tmp_graph_nodes)
        
    connected_nodes_indices = get_connected_nodes_indices(graph_nodes)

    if len(connected_nodes_indices) == 1:
        master_graph = nx.DiGraph()

        for G in graphs:
            # Deep copy to ensure original graph is not modified
            G_copy = copy.deepcopy(G)

            # Add nodes and edges from the copied graph to the master graph
            master_graph.add_nodes_from(G_copy.nodes(data=True))
            master_graph.add_edges_from(G_copy.edges(data=True))

        return [master_graph]
    

    joined_graphs = []

    already_checked_graphs = np.zeros(len(graphs), dtype=np.int32)
    
    for index, indices in enumerate(connected_nodes_indices):
        if index % check_interval == 0:
            print("Joining", index, "-th / ", len(connected_nodes_indices), "connected indices")
        
        n_graph = None

        for i, graph in enumerate(graphs):
            if already_checked_graphs[i] == 1:
                continue
                
            if i % graphs_checked_interval == 0:
                print("Checking graph #", i,"/", len(graphs))
            common_elements = list(set(graph_nodes[i]).intersection(indices))
            if len(common_elements) > 0:
                
                already_checked_graphs[i] = 1
                
                if n_graph == None: # Concatenate graphs
                    n_graph = copy.deepcopy(graph)
                else:
                    n_graph = nx.compose(n_graph, graph)
                    
        
        if n_graph is not None:
            joined_graphs.append(n_graph)


    return joined_graphs