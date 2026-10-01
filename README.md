# PhD dissertation
**Title:** Method for Spatiotemporal Semantic Segmentation of Land Use on Satellite Imagery Using Graph Neural Networks

---

This repository includes:
- The abstract of the submitted PhD dissertation
- An illustration of the basic workflow of the proposed semantic segmentation method
- The algorithm for subgraph sampling and an image demonstrating results of the proposed method
- A partial codebase representing the work conducted during the PhD research

---

### Information regarding the PhD dissertation

- **University and faculty:** University of Maribor, Faculty of Electrical Engineering and Computer Science (UM FERI), Maribor, Slovenia (EU)
- **Language:** Slovenian  
- **Completion:** March 2026
- **Available through the Digital Library of University of Maribor:** [Link](https://dk.um.si/IzpisGradiva.php?lang=slv&id=95633)

### Abstract

In the doctoral dissertation, spatiotemporal semantic segmentation of land use on satellite imagery is presented. Current state-of-the-art methods exhibit limitations in the usage of spatial context and temporal information and additionally lack adaptability to changes in the structure of input data. The proposed method overcomes these challenges by utilizing graph neural networks on time series of multispectral images of the Earth’s surface.

In the first step of the proposed method, individual images are segmented into regions, followed by the construction of a graph with spatial and temporally directed edges between the regions. For each classified region, or target node, a directed subgraph is created, which includes neighboring nodes with established spatial and temporal connections leading to the target node. These neighboring nodes thus represent the spatial and temporal neighborhood. The subgraph is then passed into the target node classification pipeline, where a convolutional neural network first extracts high-level features from the bounding boxes of the regions of all subgraph nodes. Subsequently, the subgraph, enriched with these features, is processed by a graph neural network, which performs the classification of the target node. This procedure is carried out for each node or region. The result of the proposed method is a predicted time series of the semantically segmented input area in the form of segmentation maps with land use labels.

The proposed semantic segmentation method was evaluated on the DynamicEarthNet dataset, which contains time series of daily satellite imagery from 75 geographic regions worldwide, and compared against state-of-the-art methods for training remote sensing foundation models, namely GASSL, SeCo, SatMAE, and TOV. The proposed method achieved the best average results compared to the state-of-the-art methods, with statistically significant differences, reaching an average mIoU 0.4145 ± 0.0051 (p-values: 0.0111, 0.0087, 2 × 10−6, and 9 × 10−5) when using spatial neighborhood in the subgraphs, and an mF1 0.5202 ± 0.0103 (p-values: 0.0287, 0.0274, 6 × 10−8, and 9 × 10−5) when using temporal neighborhood. By contrast, the best-performing state-of-the-art method GASSL yielded inferior average results, with an average mIoU 0.3823 ± 0.0156 and mF1 0.4908 ± 0.0198. The results thus demonstrate that the proposed semantic segmentation method achieves a better average performance, compared to the best-performing state-of-the-art method, with improvements of +0.0322 in mIoU and +0.0294 in mF1. An extensive statistical analysis of the results confirmed that the proposed method achieves statistically significantly better results, compared to existing state-of-the-art methods in terms of the mF1 and mIoU metrics. The best-trained model of the proposed method achieved a wF1 score of 0.6905 on the DynamicEarthNet dataset when using temporal neighborhood in the subgraphs. The highest classification performance was obtained for the class »Water«, with an F1 score of 0.9159 and an IoU of 0.8449. Among the remaining five classification classes, the »Forest and other vegetation« class stood out with an F1 score of 0.7890 and an IoU of 0.6515, while the »Soil« class was classified with an F1 score of 0.6561 and an IoU of 0.4882.

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

Requirements:
- Python and Anaconda  
- PyTorch  
- Deep Graph Library (DGL)
- NetworkX
- NumPy & SciPy
- scikit-image & OpenCV

Proposed method code structure (`source/`):
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

 <br />
 <br />
 <br />
State-of-the-art foundation models that the proposed method was compared against (`source_state_of_the_art_methods/`):

* `methods/`: original repositories of the compared foundation models (self-supervised pretrained encoders), with minor adaptations so they can be used as segmentation backbones:
   * `GASSL/` – Geography-Aware Self-Supervised Learning (MoCo-v2 ResNet-50 pretrained on fMoW). Variants: `GASSL-basic`, `GASSL-TP`, `GASSL-GEO`, `GASSL-GEO+TP`.
   * `SeCo/` – Seasonal Contrast (MoCo-v2 ResNet-18/50 pretrained on SeCo-100K/1M). Variants: `SeCo-100K-ResNet-18`, `SeCo-1M-ResNet-18`, `SeCo-100K-ResNet-50`, `SeCo-1M-ResNet-50`.
   * `SatMAE/` – Satellite Masked Autoencoder (ViT pretrained on fMoW). Variants: `SatMAE_fMoW_Non_Temporal_ViT-Large`, `SatMAE_fMoW_Temporal_ViT-Large`, `SatMAE_fMoW_MultiSpectral_ViT-Base`, `SatMAE_fMoW_MultiSpectral_ViT-Large`.
      * `models_vit.py` and `models_vit_temporal.py` were modified so that `forward()` returns patch-token features (CLS token removed) instead of classification outputs. In the temporal variant, patch features of the three input images are additionally averaged across time.
   * `TOV/` – The Original Vision model (ResNet-50 pretrained on TOV-RS-balanced), from the G-RSIM repository.
   * Each method contains a `weights/` folder into which the official pretrained checkpoints must be downloaded before training.
* `decoders/TransUNet/`: original TransUNet repository. Its `DecoderCup` (in `networks/vit_seg_modeling.py`) is used as the segmentation decoder for ViT-based encoders (SatMAE), configured without skip connections.
* `my_code/`: core library that wraps the foundation models into a unified segmentation pipeline:
   * Backbones: Implemented in `backbones.py` within the following classes:
      * `MoCoResNet50Backbone` – GASSL encoder
      * `SeCo_MoCoResNet50Backbone` – SeCo encoder
      * `SatMAE` – SatMAE encoder (also provides the matching TransUNet decoder config via `get_config()`)
      * `TOV` – TOV encoder

      Each foundation model loads the pretrained checkpoint, removes the classification head, and expands the first convolution/patch embedding layer from 3 (RGB) to 4 input channels (RGB + NIR), initializing the NIR channel with the pretrained weights of the red channel. CNN backbones return multi-scale feature maps from all ResNet stages.
   * Model: Implemented in `model.py`:
      * `create_state_of_the_art_model()` → builds the encoder–decoder model for the selected architecture: ResNet-based encoders (GASSL, SeCo, TOV) are combined with a UPerNet decoder, ViT-based encoders (SatMAE) with the TransUNet decoder, and a ResNet-34 U-Net (ImageNet weights) is included as a baseline.
      * `MyModel` – PyTorch Lightning module handling input resizing, normalization, random horizontal/vertical flip augmentation, loss (cross-entropy or focal), AdamW optimization with polynomial learning-rate decay, and logging of mIoU.
   * Data loading: Implemented in `dataset.py`:
      * `Dataset` class → loads monthly image/label pairs. For the temporal SatMAE variant, it also returns the two preceding images of the time series together with their timestamps.
      * `create_dataloader()` → creates dataloaders for the selected train/val/test subsets.
   * Evaluation: Implemented in `utils.py`:
      * `run_metrics()` → computes mIoU and weighted/macro F1 per subset and across the whole dataset.
* `default_train_set_normalization_data.json`: per-channel (R, G, B, NIR) means and standard deviations of the training set, used to standardize inputs during training and inference.
* **`training.ipynb`**: Jupyter notebook for training and evaluating the selected model (`model_arch`) over multiple independent runs (with early stopping and best-checkpoint saving), and writing the averaged test metrics to a text file. Set `relative_path_to_data` to the location of the prepared data before running.

 <br />
 <br />
 <br />
Demo of the proposed method with pretrained weights (`run_proposed_method.ipynb`):

* **`run_proposed_method.ipynb`**: Jupyter notebook that creates the proposed model, loads the best trained weights, runs inference on a single demo region and plots the predicted segmentation maps next to the ground truth.
   * **Model configuration** (matching the `run_training.py` arguments of the best-trained model):
      * GNN architecture: Heterogeneous Graph Transformer (HGT), `MyModelGraphTransformer` with 8 attention heads and 256 hidden features (`--use_transformer --num_heads 8 --h_feat_amount 256`)
      * Feature extractor: EfficientNetV2-S (`--feature_extraction_NN EfficientNetV2-S`)
      * Region bounding boxes: 32 × 32 px with 4 channels, RGB + NIR (`--region_bbox_size 32 --region_bbox_channels 4`)
      * Subgraph sampling: `ProposedSampler` with temporal neighborhood only, using 0 spatial steps and 1 temporal step into the past with a fanout of 2 (`--use_custom_sampler --spatial_step_surrounding_region 0 --space_neighborhood_size 0 --time_step_past 1 --time_neighborhood_size 2`)
   * **Required files:**
      * `proposed_method_weights/best_model.pt` contains the trained model weights.
      * `source/train_set_normalization_data.json` contains the normalization parameters of the training set.
      * The demo region folder must contain `graph.pickle` and `segmentation_masks.pickle`, as produced by `data_preparation/1_prepare_graphs.ipynb`.
   * **Usage:** set `DEMO_REGION_PATH` to the demo region folder, including the trailing `/`, and run all cells in order. The notebook performs the following steps:
      1. Builds the model and loads the weights. The number of classes is read from the checkpoint.
      2. Converts the NetworkX graph of the region into a DGL graph.
      3. Classifies every node (region) with subgraphs sampled by `ProposedSampler`.
      4. Maps the node predictions back to pixels using the segmentation masks.
      5. Plots the predicted segmentation maps and the ground truth for the selected months (`time_indices_to_plot`), using the DynamicEarthNet class colors.
