from datetime import datetime
import numpy as np
import os
import random
import torch
import yaml

import plotly.graph_objects as go
import seaborn as sns
import sys

from voxmol.constants import COLORS


def makedir(path: str):
    """
    Create a directory at the specified path if it does not already exist.

    Args:
        path (str): The path of the directory to be created.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def seed_everything(seed: int):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.

    Returns:
        None
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def create_exp_dir(config: dict):
    """
    Create an experiment directory based on the provided configuration.

    Args:
        config (dict): The configuration dictionary containing the experiment details.

    Returns:
        None
    """
    if config['exp_name'] is None:
        exp_name = datetime.now().strftime('%Y%m%d_%H%M%S').replace("'", '')
    else:
        exp_name = config['exp_name']
    output_dir = os.path.join(config['exp_dir'], exp_name)
    config['output_dir'] = output_dir
    makedir(output_dir)
    print('>> saving experiments in:', output_dir)
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

def get_task_type():
    script_name = os.path.basename(sys.argv[0])
    if 'atom' in script_name:
        return 'atom'
    elif 'amino' in script_name:
        return 'amino'
    else:
        return 'default'

def save_checkpoint(
    state: dict,
    config: dict,
    epoch: int,
    chkp_name: str = "checkpoint.ckpt",
    is_best: bool = False
):
    task_type = get_task_type()
    chkp_name = f"checkpoint_epoch_{epoch}_{task_type}.ckpt"
    
    filename = os.path.join(config['output_dir'], chkp_name)
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")
    
    if is_best:
        best_path = os.path.join(config['output_dir'], f"best_checkpoint_{task_type}.ckpt")
        torch.save(state, best_path)
        print(f"Best checkpoint saved to {best_path}")


def load_checkpoint(
    model: torch.nn.Module,
    pretrained_path: str,
    optimizer: torch.optim.Optimizer = None,
    best_model: bool = True
):
    """
    Loads a checkpoint file and restores the model and optimizer states.

    Args:
        model (torch.nn.Module): The model to load the checkpoint into.
        pretrained_path (str): The path to the directory containing the checkpoint file.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load the checkpoint into. Defaults to None.
        best_model (bool, optional): Whether to load the best model checkpoint or the regular checkpoint.
            Defaults to True.

    Returns:
        tuple: A tuple containing the loaded model, optimizer (if provided), and the number of epochs trained.
    """
    task_type = get_task_type()
    
    chck_name = f"best_checkpoint_{task_type}.ckpt" if best_model else f"checkpoint_epoch_{epoch}_{task_type}.ckpt"
    checkpoint_path = os.path.join(pretrained_path, chck_name)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found")
    
    checkpoint = torch.load(checkpoint_path)
    n_epochs = checkpoint['epoch']

    # little hack to cope with torch.compile and multi-GPU training
    sd = "state_dict_ema" if "state_dict_ema" in checkpoint else "state_dict"
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint[sd].items()}
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint[sd].items()}

    print(f">> Model from {pretrained_path} trained for {n_epochs} epochs. Weights have been loaded.")
    
    model.load_state_dict(state_dict)
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        return model, optimizer, checkpoint["epoch"]
    else:
        return model, checkpoint["epoch"]


def mol2xyz(sample: dict):
    """
    Convert a molecular sample dictionary to XYZ format.

    Args:
        sample (dict): A dictionary containing the molecular sample data.

    Returns:
        str: The molecular sample data in XYZ format.
    """
    sample = remove_atoms_too_close(sample)
    n_atoms = sample['atoms_channel'].shape[-1]
    atom_elements = ["C", "H", "O", "N", "F", "S", "Cl", "Br", "P", "I", "B"]
    xyz_str = str(n_atoms) + "\n\n"
    for i in range(n_atoms):
        element = sample['atoms_channel'][0, i]
        element = atom_elements[int(element.item())]

        coords = sample['coords'][0, i, :]

        line = element + "\t" + str(coords[0].item()) + "\t" + str(coords[1].item()) + "\t" + str(coords[2].item())
        xyz_str += line + "\n"
    return xyz_str


def remove_atoms_too_close(mol: dict, dist_thr=.8):
    # this can be done in a much better way
    idxs = [None]
    while len(idxs) > 0:
        dists = torch.cdist(mol["coords"], mol["coords"], compute_mode="donot_use_mm_for_euclid_dist")[0]
        idxs = np.where((dists > 0) & (dists < dist_thr))
        idxs = list(set(np.concatenate(idxs)))
        if len(idxs) == 0:
            break
        n_atoms = mol["atoms_channel"].shape[-1]
        rows = torch.BoolTensor(n_atoms).fill_(True)
        rows[idxs[0]] = False
        mol = {
            "coords": mol["coords"][:, rows, :],
            "atoms_channel": mol["atoms_channel"][:, rows],
            "radius": mol["radius"][:, rows],
        }

    return mol


def save_molecules_xyz(molecules_xyz: list, out_dir: str, obabel_postprocess: bool = True):
    """
    Save a list of molecules in XYZ format to individual files in the specified output directory.

    Args:
        molecules_xyz (list): A list of strings representing the XYZ format of each molecule.
        out_dir (str): The path to the output directory where the files will be saved.
        obabel_postprocess (bool, optional): Whether to postprocess the saved files using Open Babel.
            Defaults to True.
    """
    for idx, mol_xyz in enumerate(molecules_xyz):
        with open(os.path.join(out_dir, f"sample_{idx:05d}.xyz"), "w") as f:
            f.write(mol_xyz)

    # postprocess them (ie, convert all the .xyz into a single .sdf)
    if obabel_postprocess:
        print(">> process .xyz files and save in .sdf on same folder")
        cmd = f"cd {out_dir}; obabel *xyz -osdf -O molecules_obabel.sdf --title  end"
        os.system(cmd)


########################################################################################
# Visualization
def visualize_voxel_grid(
    voxel: torch.Tensor,
    fname: str = "figures/temp.png",
    threshold: float = 0.1,
    to_png: bool = True,
    to_html: bool = False
):
    """
    Visualizes a voxel grid using Plotly.

    Args:
        voxel (torch.Tensor): The voxel grid to visualize.
        fname (str, optional): The filename to save the visualization. Defaults to "figures/temp.png".
        threshold (float, optional): The threshold value to remove voxels below. Defaults to 0.1.
        to_png (bool, optional): Whether to save the visualization as a PNG image. Defaults to True.
        to_html (bool, optional): Whether to save the visualization as an HTML file. Defaults to False.
    """
    sns.set_theme()

    voxel = voxel.squeeze().cpu()
    assert len(voxel.shape) == 4, "voxel grid need to be of the form CxLxLxL"
    voxel[voxel < threshold] = 0
    X, Y, Z = np.mgrid[:voxel.shape[-3], :voxel.shape[-2], :voxel.shape[-1]]

    fig = go.Figure()
    for channel in range(voxel.shape[0]):
        voxel_channel = voxel[channel:channel+1]
        if voxel_channel.sum().item() == 0:
            continue
        fig.add_volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=voxel_channel.flatten(),
            isomin=0.1,
            isomax=.3,  # 0.8,
            opacity=0.1,  # 0.075, # needs to be small to see through all surfaces
            surface_count=17,  # needs to be a large number for good volume rendering
            colorscale=COLORS[channel],
            showscale=False
        )
    if to_html:
        fig.write_html(fname.replace("png", "html"))
    if to_png:
        fig.write_image(fname)
