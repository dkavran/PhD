import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn import metrics

from .dataset import create_dataloader

def get_predictions_to_plot_later(model, selected_set_to_evaluate, set, n_classes, RGB_or_RGB_plus_near_INFRARED, num_workers, all_runs_subset_mious, all_runs_subset_weighted_f1s, all_runs_subset_macro_f1s, show_images, relative_path_to_data = "../../../prepared_dynamic_earth_net_data/", keep_interval = "monthly", temporal = False, device = None):    
    pred_images = []
    test_dataloader = create_dataloader([set], selected_set_to_evaluate, n_classes, RGB_or_RGB_plus_near_INFRARED, verbose=False, batch_size=1, shuffle=False, num_workers=num_workers, relative_path_to_data=relative_path_to_data, keep_interval = keep_interval, temporal = temporal)
        
    if temporal is False:
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_dataloader):
                output = model(inputs.to(device))
                output_image = output.detach().cpu().numpy()[0].argmax(axis=0)
                
                pred_images.append(output_image)
    else:
        with torch.no_grad():
            for batch_idx, (inputs, timesteps, labels) in enumerate(test_dataloader):
                output = model(inputs.to(device), timesteps.to(device))
                output_image = output.detach().cpu().numpy()[0].argmax(axis=0)
                
                pred_images.append(output_image)
    
    return pred_images


def run_metrics(model, selected_set_to_evaluate, sets, n_classes, RGB_or_RGB_plus_near_INFRARED, num_workers, all_runs_subset_mious, all_runs_subset_weighted_f1s, all_runs_subset_macro_f1s, show_images, relative_path_to_data = "../../../prepared_dynamic_earth_net_data/", keep_interval = "monthly", temporal = False, device = None):
    #metrics storage
    dataset_labels = []
    dataset_pred_labels = []
    subset_mious = []
    subset_weighted_f1s = []
    subset_macro_f1s = []
    
    #run commands
    set_to_run = None
    if selected_set_to_evaluate in ["train", "val", "test"]:
        set_to_run = sets
    else:
        raise Exception("selected_set_to_evalute can be only 'train', 'val' or 'test'")
    
    for test_index, test_set in enumerate(set_to_run):
        test_dataloader = create_dataloader([test_set], selected_set_to_evaluate, n_classes, RGB_or_RGB_plus_near_INFRARED, verbose=False, batch_size=1, shuffle=False, num_workers=num_workers, relative_path_to_data=relative_path_to_data, keep_interval = keep_interval, temporal = temporal)
        
        subset_labels = []
        subset_pred_labels = []

        
        if temporal is False:
            with torch.no_grad():
                #with torch.autocast(device_type="cuda", dtype=torch.float16):
                for batch_idx, (inputs, labels) in enumerate(test_dataloader):
                    output = model(inputs.to(device))
                    output_image = output.detach().cpu().numpy()[0].argmax(axis=0)
                    label_image = labels.detach().cpu().numpy()[0]
                
                    subset_pred_labels.append(output_image)
                    subset_labels.append(label_image)
        else:
            with torch.no_grad():
                #with torch.autocast(device_type="cuda", dtype=torch.float16):
                for batch_idx, (inputs, timesteps, labels) in enumerate(test_dataloader):
                    output = model(inputs.to(device), timesteps.to(device))
                    output_image = output.detach().cpu().numpy()[0].argmax(axis=0)
                    label_image = labels.detach().cpu().numpy()[0]
                
                    subset_pred_labels.append(output_image)
                    subset_labels.append(label_image)
    
        if show_images is True:
            plt.imshow(subset_pred_labels[0])
            plt.show()
            
            plt.imshow(subset_labels[0])
            plt.show()
    
        subset_miou = mIOU(torch.from_numpy(np.array(subset_labels)), torch.from_numpy(np.array(subset_pred_labels)), n_classes = n_classes)

        mask = np.array(subset_pred_labels) != 6 #don't include snow & ice in calculation of F1 metric (mIoU one row above automatically skips this with check: true_label.long().sum().item() == 0, so it's solved )
        subset_weighted_f1 = metrics.f1_score(np.array(subset_labels)[mask].ravel(), np.array(subset_pred_labels)[mask].ravel(), average='weighted')
        subset_macro_f1 = metrics.f1_score(np.array(subset_labels)[mask].ravel(), np.array(subset_pred_labels)[mask].ravel(), average='macro')
    
        subset_mious.append(subset_miou)
        subset_weighted_f1s.append(subset_weighted_f1)
        subset_macro_f1s.append(subset_macro_f1)

        all_runs_subset_mious.append(subset_miou)
        all_runs_subset_weighted_f1s.append(subset_weighted_f1)
        all_runs_subset_macro_f1s.append(subset_macro_f1)
    
        for image in subset_labels:
            dataset_labels.append(image)
    
        for image in subset_pred_labels:
            dataset_pred_labels.append(image)
    
    #complete dataset calculation
    dataset_miou = mIOU(torch.from_numpy(np.array(dataset_labels)), torch.from_numpy(np.array(dataset_pred_labels)), n_classes = n_classes)

    mask = np.array(dataset_pred_labels) != 6 #don't include snow & ice in calculation of F1 metric (mIoU one row above automatically skips this with check: true_label.long().sum().item() == 0, so it's solved )
    dataset_weighted_f1 = metrics.f1_score(np.array(dataset_labels)[mask].ravel(), np.array(dataset_pred_labels)[mask].ravel(), average='weighted')
    dataset_macro_f1 = metrics.f1_score(np.array(dataset_labels)[mask].ravel(), np.array(dataset_pred_labels)[mask].ravel(), average='macro')

    dataset_weighted_f1_correct = metrics.f1_score(np.array(dataset_labels).ravel(), np.array(dataset_pred_labels).ravel(), labels=[0,1,2,3,4,5], average='weighted')
    dataset_macro_f1_correct = metrics.f1_score(np.array(dataset_labels).ravel(), np.array(dataset_pred_labels).ravel(), labels=[0,1,2,3,4,5], average='macro')
    
    return subset_mious, subset_weighted_f1s, subset_macro_f1s, dataset_miou, dataset_weighted_f1, dataset_macro_f1, dataset_weighted_f1_correct, dataset_macro_f1_correct

def mIOU(mask, pred_mask, smooth=1e-10, n_classes=7):
    iou_per_class = []
    for classes in range(0, n_classes):
        true_class = (pred_mask == classes)
        true_label = (mask == classes)
            
        if true_label.long().sum().item() == 0:
            iou_per_class.append(np.nan)
        else:
            intersect = torch.logical_and(true_class, true_label).sum().float().item()
            union = torch.logical_or(true_class, true_label).sum().float().item()
                
            iou = (intersect + smooth) / (union + smooth)
            iou_per_class.append(iou)

    return np.nanmean(iou_per_class)

def snapshot_weights(model):
    """
    Take a clone of every tensor in model.state_dict().
    Returns a dict mapping name → tensor clone.
    """
    return {name: tensor.clone() for name, tensor in model.state_dict().items()}

def report_weight_changes(model, orig_snapshot, atol=0.0):
    """
    Compare current model weights to an orig_snapshot (from snapshot_weights),
    and print which tensors have changed.

    Args:
        model (torch.nn.Module): your model.
        orig_snapshot (dict): name→tensor clone from snapshot_weights().
        atol (float): tolerance for torch.allclose; 0.0 means exact equality.
    """
    changed = []
    for name, new_tensor in model.state_dict().items():
        old_tensor = orig_snapshot[name]
        same = torch.equal(new_tensor, old_tensor) if atol == 0.0 else torch.allclose(new_tensor, old_tensor, atol=atol)
        if not same:
            changed.append(name)

    if changed:
        print(f"{len(changed)} tensor(s) updated after loading weights...")
    else:
        print("No parameters changed.")