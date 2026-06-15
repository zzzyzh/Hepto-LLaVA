"""
Dataset and Data Loading Utilities

Components:
- MAEPretrainDataset: MAE pretraining dataset
- ClassificationDataset: Classification dataset
- Data loading and processing functions
"""

import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from typing import Dict, Tuple, List


class MAEPretrainDataset(Dataset):
    
    def __init__(self, feature_dir: str, return_pack_coords: bool = False):
        self.feature_dir = feature_dir
        self.file_list = [f for f in os.listdir(feature_dir) if f.endswith('.pt')]
        self.return_pack_coords = return_pack_coords
        print(f"Found {len(self.file_list)} feature files for pretraining")
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        filepath = os.path.join(self.feature_dir, self.file_list[idx])
        data = torch.load(filepath, map_location='cpu', weights_only=False)
        features = data['features']
        if self.return_pack_coords:
            pack_coords = data.get('pack_coords', [])
            return features, pack_coords
        return features


class ClassificationDataset(Dataset):
    
    def __init__(
        self, 
        features: List[torch.Tensor], 
        labels: Dict[str, List], 
        task_names: List[str], 
        task_info: Dict[str, Dict]
    ):
        self.features = features
        self.labels = labels
        self.task_names = task_names
        self.task_info = task_info
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feat = self.features[idx]
        labels_dict = {}
        for task in self.task_names:
            label_val = self.labels[task][idx]
            dtype = self.task_info[task]['dtype']
            if dtype == torch.float32:
                labels_dict[task] = torch.tensor(label_val, dtype=dtype)
            else:
                labels_dict[task] = torch.tensor(int(label_val), dtype=dtype)
        return feat, labels_dict


def collate_fn_mae(batch):
    if isinstance(batch[0], tuple):
        features, pack_coords = batch[0]
        return features.unsqueeze(0), pack_coords
    else:
        features = batch[0].unsqueeze(0)
        return features, None


def collate_fn_classification(batch):
    features, labels_dict_list = zip(*batch)
    features = features[0].unsqueeze(0)
    labels_dict = labels_dict_list[0]
    batch_labels_dict = {}
    for task, label_tensor in labels_dict.items():
        if label_tensor.dim() == 0:
            batch_labels_dict[task] = label_tensor.unsqueeze(0)
        else:
            batch_labels_dict[task] = label_tensor
    return features, batch_labels_dict


def load_features_only(feature_dir: str) -> List[torch.Tensor]:
    print(f"Loading features from {feature_dir}...")
    all_features = []
    
    for filename in os.listdir(feature_dir):
        if not filename.endswith('.pt'):
            continue
        filepath = os.path.join(feature_dir, filename)
        data = torch.load(filepath, map_location='cpu', weights_only=False)
        features = data['features']
        all_features.append(features)
    
    print(f"Loaded {len(all_features)} samples for pretraining")
    return all_features


def load_features_and_labels(
    feature_dir: str,
    label_csv: str,
    id_col: str = 'id'
) -> Tuple[List[torch.Tensor], Dict[str, List], Dict[str, Dict], List[str]]:
    print("=" * 60)
    print("Loading features and labels...")
    print("=" * 60)
    
    df = pd.read_csv(label_csv)
    df[id_col] = df[id_col].astype(str)
    
    target_cols = [col for col in df.columns if col != id_col]
    print(f"Found {len(target_cols)} target columns: {target_cols}")
    
    labels_dict = {}
    task_info = {}
    
    for col in target_cols:
        valid_df = df[[id_col, col]].dropna()
        unique_values = sorted(valid_df[col].unique())
        n_classes = len(unique_values)
        
        if n_classes == 2:
            task_type = 'binary'
            output_dim = 1
            dtype = torch.float32
            label_map = {val: float(i) for i, val in enumerate(unique_values)}
        else:
            task_type = 'multi'
            output_dim = n_classes
            dtype = torch.long
            label_map = {val: int(i) for i, val in enumerate(unique_values)}
        
        print(f"  {col}: {task_type}, classes={n_classes}")
        
        id_to_label = {str(row[id_col]): label_map[row[col]] for _, row in valid_df.iterrows()}
        labels_dict[col] = id_to_label
        task_info[col] = {
            'output_dim': output_dim,
            'task_type': task_type,
            'classes': unique_values,
            'dtype': dtype,
            'label_map': label_map
        }
    
    all_features = []
    all_ids = []
    matched_labels = {task: [] for task in target_cols}
    
    for filename in os.listdir(feature_dir):
        if not filename.endswith('.pt'):
            continue
        
        slide_id = os.path.splitext(filename)[0]
        has_all_labels = all(slide_id in labels_dict[task] for task in target_cols)
        if not has_all_labels:
            continue
        
        filepath = os.path.join(feature_dir, filename)
        data = torch.load(filepath, map_location='cpu', weights_only=False)
        features = data['features']
        
        all_features.append(features)
        all_ids.append(slide_id)
        
        for task in target_cols:
            matched_labels[task].append(labels_dict[task][slide_id])
    
    print(f"Loaded {len(all_features)} samples with labels")
    return all_features, matched_labels, task_info, all_ids

