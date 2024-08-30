import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from argparse import ArgumentParser

from voxmol.dataset.dataset import DatasetVoxMol

DATASET_DIR = "input/"
TRAIN_SUB_GRIDS = "train_sub_grids"
VALID_SUB_GRIDS = "valid_sub_grids"

file = open(os.path.join(DATASET_DIR, 'train_splits.txt'))
train = file.readlines()
print("Training Data file found and the number of protein graph splits are:", len(train))

file = open(os.path.join(DATASET_DIR, 'valid_splits.txt'))
valid = file.readlines()
print("Valid Data file found and the number of protein graph splits are:", len(valid))

class CryoData(Dataset):
    def __init__(self, root, sub_grid_dir, transform=None, target_transform=None):
        self.root = root
        self.sub_grid_dir = sub_grid_dir
        self.transform = transform
        self.trarget_transform = target_transform
        self.data_files = []
        
        # 构建完整的目录路径
        full_dir_path = os.path.join(self.root, self.sub_grid_dir)

        # 检查目录是否存在
        if not os.path.exists(full_dir_path):
            raise FileNotFoundError(f"The directory does not exist: {full_dir_path}")

        # 读取目录下所有的.npz文件
        for file in os.listdir(full_dir_path):
            if file.endswith('.npz'):
                self.data_files.append(file)

        print(f"Loaded {len(self.data_files)} files from {full_dir_path}")

    def __len__(self):
        return len(self.data_files)

    # def __getitem__(self, idx):
    #     cryodata = train[idx]
    #     cryodata = cryodata.strip("\n")
    #     # loaded_data = np.load(f"{DATASET_DIR}/{TRAIN_SUB_GRIDS}/{cryodata}")
    #     file_path = os.path.join(self.root, TRAIN_SUB_GRIDS, cryodata)
    #     # print(f"Loading file: {file_path}")
    #     if not os.path.exists(file_path):
    #         raise FileNotFoundError(f"File not found: {file_path}")
    #     loaded_data = np.load(file_path)
        
    #     protein_manifest = loaded_data['protein_grid']
    #     protein_torch = torch.from_numpy(protein_manifest).type(torch.FloatTensor)
    #     atom_manifest = loaded_data['atom_grid']
    #     atom_torch = torch.from_numpy(atom_manifest).type(torch.FloatTensor)
    #     # esm_embeds = loaded_data['embeds']
    #     # esm_embeds_torch = torch.from_numpy(esm_embeds).type(torch.FloatTensor)

    #     return [protein_torch, atom_torch]
    
    def __getitem__(self, idx):
        cryodata = train[idx].strip("\n")
        file_path = os.path.join(self.root, TRAIN_SUB_GRIDS, cryodata)
        # print("Attempting to load:", file_path)
        # 检查文件扩展名，以确定如何加载数据
        # file_name = self.data_files[idx]
        # file_path = os.path.join(self.root, TRAIN_SUB_GRIDS, file_name)
        loaded_data = np.load(file_path, allow_pickle=True)
        
        protein_manifest = loaded_data['protein_grid']
        protein_torch = torch.from_numpy(protein_manifest).type(torch.FloatTensor)
        atom_manifest = loaded_data['atom_grid']
        atom_torch = torch.from_numpy(atom_manifest).type(torch.FloatTensor)

        combined_torch = torch.cat([protein_torch.unsqueeze(0), atom_torch.unsqueeze(0)], dim=0)
        return protein_torch, atom_torch

class CryoData_valid(Dataset):
    def __init__(self, root, sub_grid_dir, transform=None, target_transform=None):
        self.root = root
        self.sub_grid_dir = sub_grid_dir
        self.transform = transform
        self.trarget_transform = target_transform
        self.data_files = []
        
        # 构建完整的目录路径
        full_dir_path = os.path.join(self.root, self.sub_grid_dir)

        # 检查目录是否存在
        if not os.path.exists(full_dir_path):
            raise FileNotFoundError(f"The directory does not exist: {full_dir_path}")

        # 读取目录下所有的.npz文件
        for file in os.listdir(full_dir_path):
            if file.endswith('.npz'):
                self.data_files.append(file)

        print(f"Loaded {len(self.data_files)} files from {full_dir_path}")

    def __len__(self):
        return len(self.data_files)

    # def __getitem__(self, idx):
    #     cryodata = valid[idx]
    #     cryodata = cryodata.strip("\n")
    #     # loaded_data = np.load(f"{DATASET_DIR}/{VALID_SUB_GRIDS}/{cryodata}")
    #     file_path = os.path.join(DATASET_DIR, VALID_SUB_GRIDS, cryodata)
    #     # print(f"Loading file: {file_path}")
    #     loaded_data = np.load(file_path)
        
    #     protein_manifest = loaded_data['protein_grid']
    #     protein_torch = torch.from_numpy(protein_manifest).type(torch.FloatTensor)
    #     atom_manifest = loaded_data['atom_grid']
    #     atom_torch = torch.from_numpy(atom_manifest).type(torch.FloatTensor)
    #     return [protein_torch, atom_torch]
    
    def __getitem__(self, idx):
        cryodata = train[idx].strip("\n")
        file_path = os.path.join(self.root, TRAIN_SUB_GRIDS, cryodata)
        
        # file_name = self.data_files[idx]
        # file_path = os.path.join(self.root, TRAIN_SUB_GRIDS, file_name)
        loaded_data = np.load(file_path, allow_pickle=True)
        
        protein_manifest = loaded_data['protein_grid']
        protein_torch = torch.from_numpy(protein_manifest).type(torch.FloatTensor)
        atom_manifest = loaded_data['atom_grid']
        atom_torch = torch.from_numpy(atom_manifest).type(torch.FloatTensor)

        combined_torch = torch.cat([protein_torch.unsqueeze(0), atom_torch.unsqueeze(0)], dim=0)
        return protein_torch, atom_torch
        
def create_loader(config: dict):
    """
    Create data loaders for training and validation sets.

    Args:
        config (dict): Configuration parameters for the data loaders.

    Returns:
        tuple: A tuple containing the training and validation data loaders.
    """
    # dset_train = DatasetVoxMol(
    #     dset_name=config["dset_name"],
    #     data_dir=config["data_dir"],
    #     elements=config["elements"],
    #     split="train",
    #     small=config["debug"],
    # )
    # loader_train = torch.utils.data.DataLoader(
    #     dset_train,
    #     batch_size=config["batch_size"],
    #     num_workers=config["num_workers"],
    #     shuffle=True,
    #     pin_memory=True,
    #     drop_last=True,
    # )

    # dset_val = DatasetVoxMol(
    #     dset_name=config["dset_name"],
    #     data_dir=config["data_dir"],
    #     elements=config["elements"],
    #     split="val",
    #     small=config["debug"],
    # )
    # loader_val = torch.utils.data.DataLoader(
    #     dset_val,
    #     batch_size=config["batch_size"],
    #     num_workers=config["num_workers"],
    #     shuffle=False,
    #     pin_memory=True,
    #     drop_last=False,
    # )
    BATCH_SIZE = 4 * 4 * 1
    DATALOADERS = 6
    
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help="effective_batch_size = batch_size * num_gpus * num_nodes")
    parser.add_argument('--num_dataloader_workers', type=int, default=DATALOADERS, help="Number of workers for data loading")
    
    args = parser.parse_args()
    
    
    # 加载数据集
    dataset = CryoData(root="input", sub_grid_dir="train_sub_grids")
    dataset_valid = CryoData_valid(root="input", sub_grid_dir="valid_sub_grids")
    loader_train = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    loader_val = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"  | train loader: {len(loader_train)} batches")
    print(f"  | val loader: {len(loader_val)} batches")
    return loader_train, loader_val
