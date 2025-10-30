# Submitted PhD dissertation
**Title:** Method for spatiotemporal semantic segmentation of land use on satellite imagery using graph neural networks

---

This repository includes:
- The abstract of the submitted PhD dissertation
- An illustration of the basic workflow of the proposed semantic segmentation method
- The algorithm for subgraph sampling and an image demonstrating results of the proposed method
- A partial codebase representing the work conducted during the PhD research

---

### Information regarding the PhD submission

- **Submission date:** October 20, 2025 (***currently under committee review***)  
- **University and faculty:** University of Maribor, Faculty of Electrical Engineering and Computer Science (UM FERI), Slovenia  
- **Language:** Slovenian  
- **Expected defence and completion:** May–June 2026  

### Abstract

In the doctoral dissertation, spatiotemporal semantic segmentation of land use on satellite imagery is presented. Current state-of-the-art methods exhibit limitations in usage of spatial context and temporal information. The proposed method overcomes these challenges by utilizing graph neural networks on time series of multispectral images of the Earth’s surface. In the first step, individual images are segmented into regions, followed by the construction of a graph with spatial and temporally directed edges between the regions. For each classified region, or target node, a directed subgraph is created, which includes neighboring nodes with established spatial and temporal connections, leading to the target node. These neighboring nodes thus represent the spatial and temporal neighborhood. The subgraph is then passed into the target node classification pipeline, where a convolutional neural network first extracts high-level features over the bounding boxes of the regions of all subgraph nodes. Subsequently, the subgraph enriched with these features, is processed by a graph neural network, which performs the classification of the target node. This procedure is carried out for each node or region. The result of the proposed method is a predicted time series of semantically segmented input area in the form of segmentation maps with land use labels. The method was evaluated on the DynamicEarthNet dataset and compared against state-of-the-art methods for training remote sensing foundation models, namely GASSL, SeCo, SatMAE, and TOV. The proposed method achieved the best average results, with an average mIoU 0.4145, when using spatial neighborhood in the subgraphs, and an mF1 0.5202, when using temporal neighborhood. The best-trained model of the proposed method achieved a wF1 score of 0.6905, when using temporal neighborhood in the subgraphs. By contrast, the best-performing state-of-the-art method, GASSL, yielded inferior average results, with an average mIoU 0.3823 and mF1 0.4908. Statistical analysis confirmed that the proposed method provides statistically significantly better results compared to existing state-of-the-art methods.

### Workflow

The image below illustrates the overall workflow of the proposed method.

![The workflow of the proposed method consisting of four main steps](images/general_workflow_proposed_method.png?raw=true "The workflow of the proposed method consisting of four main steps")

The following is the algorithm for subgraph sampling used in the proposed method (3rd step of the workflow shown in the image above):

![Algorithm for subgraph sampling](images/subgraph_sampling_algorithm.png?raw=true "Algorithm for subgraph sampling")

### Results

A comparison of the proposed semantic segmentation method was conducted against several state-of-the-art remote sensing foundation models:
- [**GASSL**](https://github.com/sustainlab-group/geography-aware-ssl)
- [**SeCo**](https://github.com/ServiceNow/seasonal-contrast)
- [**SatMAE**](https://github.com/sustainlab-group/SatMAE)
- [**TOV**](https://github.com/GeoX-Lab/G-RSIM/tree/main/TOV_v1)

Visualizations of the results obtained using the proposed method and comparable state-of-the-art remote sensing foundation models are presented in the image below.

![Results](images/vizualization_results_proposed_method_vs_state_of_the_art.png?raw=true "Results")

### Source code

***Note:*** The current version of the code is *partial* — it does **not include the dataset** and the **full codebase**. It is provided **solely as a showcase** of the source code developed for the PhD work. Future updates will include the complete codebase.

Requirements:
- Python and Anaconda  
- PyTorch  
- Deep Graph Library (DGL)
- NetworkX
- NumPy & SciPy
- scikit-image & OpenCV

Code Structure (`source/`):
- `proposed_method_lib/`: core library implementing the main logic of the proposed method:
    - **Graph construction:**
      Implemented in `graph_node.py` with the following key functions:
      - `create_time_series_graphs()` → **creates directed temporal connections** between segments/nodes.  
      - `get_all_possible_graphs_fast()` → **generates graphs** based on temporal connections.  
      - `add_spatial_connections_to_graphs()` → **adds spatial connections** between segments/nodes to the graphs.  

    - **GNN architectures:**  
      Implemented in `dgl.py` within the following classes:
      - `MyModelGAT` – **Graph Attention Network (GAT)**  
      - `MyModelGraphTransformer` – **Heterogeneous Graph Transformer (HGT)**  
      - `MyModelGraphSAGE` – **GraphSAGE**

    - **Graph sampling:**  
      - `ProposedSampler` class in `dgl.py`  
        Performs subgraph sampling, as described in the algorithm above.

- `data_preparation/`: Jupyter notebooks for data preprocessing and graph generation:
    - **`1_prepare_graphs.ipynb`**  
      Creates graphs (1st and 2nd steps in the workflow shown above).  

    - **`2_postprocessing_join_all_separate_graphs_into_one.ipynb`**  
      Merges all separate graphs into a single large graph (post-processing after 2nd step of the workflow). 

     - **`3_calculate_normalization_parameters.ipynb`**  
       Calculates normalization parameters (means and standard deviations) across all training images and saves them to `train_set_normalization_data.json`. The resulting parameters are applied to standardize input data during both model training and inference.

- `training/`: scripts for model training and evaluation:
  - **`run_training.py`**  
    Executes the training and validation (3rd and 4th steps in the workflow shown above).  

    **Example usage:**
    `nohup python -u ./run_training.py --epochs 30 --runs 5 --train_places train_data_path/ --test_places test_data_path/ --train_pickle_file_name train/combined_graph.pickle --test_pickle_file_name test/combined_graph.pickle --region_bbox_size 32 --region_bbox_channels 4 --spatial_step_surrounding_region 0 --space_neighborhood_size 0 --time_step_past 1 --time_neighborhood_size 5 --h_feat_amount 16 --data_path C://...//prepared_data --number_of_workers 0 --graph_dtype int32 --gpus 1 --batch_size 128 --test_batch_size 768 --freeze_upper_N_layers 0 --max_layer_freeze_percentage 0.0 --use_custom_sampler --aggregator_type mean --feature_extraction_NN ShuffleNetV2-x0.5  --use_class_weights --early_stopping_patience 3 --use_uva --use_AMP --use_normalization --use_focal_loss --fl_gamma 2.0 --use_ELU_in_graph_layers &`