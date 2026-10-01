import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from os import listdir
from os.path import isfile, join
import torch
from dataloader_wrapper import DataloaderWrapper
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torchvision.transforms import v2
import re

class Dataset(torch.utils.data.Dataset): #https://stackoverflow.com/questions/67406731/pytorch-import-dataset-with-images-as-labels
    def __init__(self, raw_image_paths, labels_paths, image_index_from_start_of_set, n_classes, RGB_or_RGB_plus_near_INFRARED, verbose = False, temporal = False):
        self.raw_image_paths = raw_image_paths
        self.labels_paths = labels_paths
        self.image_index_from_start_of_set = image_index_from_start_of_set
        self.n_classes = n_classes
        self.RGB_or_RGB_plus_near_INFRARED = RGB_or_RGB_plus_near_INFRARED
        self.verbose = verbose
        self.temporal = temporal

        if len(self.raw_image_paths) != len(self.labels_paths):
            raise Exception("Non-equal number of raw images and/or labels in Dataset!")
        
    def prepare_image(self, img):
        if self.RGB_or_RGB_plus_near_INFRARED == "only_rgb":
            img = img[:,:,[0,1,2]]
            if self.verbose is True:
                print("Keeping only 3 channels for RGB!")

        img = swap_dimension(img, 1, 2) #swap height with channels
        img = swap_dimension(img, 0, 1) #swap width with channels
        return img
    
    def prepare_timestamp_temporal(self, image_path):
        min_year = 2018 #hardcoded for dynamic earth net

        match = re.search(r'(\d{4})_(\d{2})_(\d{2})', image_path)
        if match:
            year = int(match.group(1))  # Convert to integer
            month = int(match.group(2))  # Convert to integer
            day = int(match.group(3))  # Convert to integer

            return np.array([year - min_year, month - 1, 0]) #year, month, hour
        else:
            raise Exception("Error when converting image_path to year, month and day")
        

    def __getitem__(self, idx):
        raw_image_path = self.raw_image_paths[idx]
        labels_path = self.labels_paths[idx]

        raw = np.load(raw_image_path)
        labels = np.load(labels_path)

        if self.verbose is True:
            print("Maximum value in raw:", raw.max(), "/ minimum value in raw:", raw.min())

        raw = self.prepare_image(raw)

        if self.verbose is True:
            print("Read raw shape:", raw.shape, "/ read labels shape:", labels.shape)
            print("Classes in labels", np.unique(labels))

        if self.verbose is True:
            print("Example of raw data:")
            plt.imshow(raw[0, :, :])
            plt.show()
        
            plt.imshow(labels)
            plt.show()

        if self.temporal is True:
            img1 = raw

            current_index_in_ts = self.image_index_from_start_of_set[idx]
            img2 = None
            img3 = None
            img2_path = None
            img3_path = None

            img2_path = self.raw_image_paths[idx - 1]
            if current_index_in_ts >= 2:
                img2_path = self.raw_image_paths[idx - 1]
                img3_path = self.raw_image_paths[idx - 2]
            elif current_index_in_ts == 1:
                img2_path = self.raw_image_paths[idx - 1]
                img3_path = self.raw_image_paths[idx - 1] #read same as img2
            else: #current_index_in_ts == 0, so no previous images are available in ts, so just duplicate the original first image twice for img2 and img3
                img2_path = raw_image_path
                img3_path = raw_image_path

            #read
            img2 = np.load(img2_path)
            img3 = np.load(img3_path)

            #prepare img2 and img3
            img2 = self.prepare_image(img2)
            img3 = self.prepare_image(img3)

            raw = torch.stack([torch.from_numpy(img1), torch.from_numpy(img2), torch.from_numpy(img3)], dim=0)
                

        if self.verbose is True:
            print("Creating dataset with shape:", raw.shape, "and", labels.shape)


        if self.temporal is False:
            return raw, labels #one_hot_labels
        else:
            ts1 = self.prepare_timestamp_temporal(raw_image_path)
            ts2 = self.prepare_timestamp_temporal(img2_path)
            ts3 = self.prepare_timestamp_temporal(img3_path)

            ts = np.stack([ts1, ts2, ts3], axis=0)

            raw = raw.float()

            return raw, ts, labels

    def __len__(self):
        return len(self.raw_image_paths)
    
dates_with_labels = [
    "2018_01_01",
    "2018_02_01",
    "2018_03_01",
    "2018_04_01",
    "2018_05_01",
    "2018_06_01",
    "2018_07_01",
    "2018_08_01",
    "2018_09_01",
    "2018_10_01",
    "2018_11_01",
    "2018_12_01",
    "2019_01_01",
    "2019_02_01",
    "2019_03_01",
    "2019_04_01",
    "2019_05_01",
    "2019_06_01",
    "2019_07_01",
    "2019_08_01",
    "2019_09_01",
    "2019_10_01",
    "2019_11_01",
    "2019_12_01"
]

def create_dataloader(sets, set_split, n_classes, RGB_or_RGB_plus_near_INFRARED, verbose, batch_size, shuffle, num_workers, relative_path_to_data = "../../../prepared_dynamic_earth_net_data/", keep_interval = "monthly", temporal = False):
    all_raw_paths = []
    all_labels_paths = []
    image_index_from_start_of_set = []

    for i_set, _set in enumerate(sets):
        print("Working on ", _set, "(#", (i_set + 1), "/", len(sets), ")")
    
        raw = []
        labels = []
    
        #read all raw and labels data
        raw_data_path = relative_path_to_data + set_split + "/" + _set + "/image"
        labels_data_path = relative_path_to_data + set_split +"/" + _set + "/label"
            
        raw_file_names = natsorted([f for f in listdir(raw_data_path) if isfile(join(raw_data_path, f))])
        labels_file_names = natsorted([f for f in listdir(labels_data_path) if isfile(join(labels_data_path, f))])
    
        label_indices_with_actual_labels = []
        if keep_interval == "monthly":
            monthly_label_image_index = 0
        for i_row_index, _ in enumerate(labels_file_names):
            if keep_interval == "monthly":
                if _.split(".")[0] in dates_with_labels:
                    label_indices_with_actual_labels.append(monthly_label_image_index)
                    image_index_from_start_of_set.append(monthly_label_image_index)
                    monthly_label_image_index += 1
                
                    raw_image_path = raw_data_path + "/" + raw_file_names[i_row_index]
                    label_image_path = labels_data_path + "/" + labels_file_names[i_row_index]

                    all_raw_paths.append(raw_image_path)
                    all_labels_paths.append(label_image_path)

    print("Number of all paths:", len(all_raw_paths))
                    
    ds = Dataset(all_raw_paths, all_labels_paths, image_index_from_start_of_set, n_classes, RGB_or_RGB_plus_near_INFRARED, verbose, temporal)

    return DataloaderWrapper(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers).dataloader

def swap_dimension(img, axis_a, axis_b):
    return np.swapaxes(img,axis_a,axis_b)