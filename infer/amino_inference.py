import os
import time
import getpass as gt
import torch
import torch.nn as nn
import math

from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, FBetaScore

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from voxmol.options import parse_args
from voxmol.metrics import create_metrics, MetricsDenoise
from voxmol.models.adamw import AdamW
from voxmol.utils import seed_everything, save_checkpoint, load_checkpoint, makedir, save_molecules_xyz

import sys
import numpy as np
import mrcfile
import torch.nn.functional as F
from copy import deepcopy

box_size = 32  # Expected Dimensions to pass to Transformer Unet
core_size = 20  # core of the image where we dnt have to worry about boundary issues

class CryoData_infer(Dataset):
    def __init__(self, root, sub_grid_dir, transform=None, target_transform=None):
        self.root = root
        self.sub_grid_dir = sub_grid_dir
        self.transform = transform
        self.target_transform = target_transform
        self.data_splits = []
        
        # 构建完整的目录路径
        full_dir_path = os.path.join(self.root, self.sub_grid_dir)

        # 检查目录是否存在
        if not os.path.exists(full_dir_path):
            raise FileNotFoundError(f"The directory does not exist: {full_dir_path}")

        # 读取目录下所有的.npz文件
        for file in os.listdir(full_dir_path):
            if file.endswith('.npz'):
                self.data_splits.append(file)

        print(f"Loaded {len(self.data_splits)} files from {full_dir_path}")

    def __len__(self):
        return len(self.data_splits)

    def __getitem__(self, idx):
        # 获取数据文件的路径
        file_name = self.data_splits[idx]
        file_path = os.path.join(self.root, self.sub_grid_dir, file_name)

        # 加载数据文件
        loaded_data = np.load(file_path, allow_pickle=True)
        
        # 只加载 protein_grid 用于推理
        protein_manifest = loaded_data['protein_grid']
        protein_torch = torch.from_numpy(protein_manifest).type(torch.FloatTensor)
        
        return protein_torch
    
def loader_infer(input_data_dir, sub_grid_dir, batch_size=1, num_workers=4):
    """
    创建用于推理的数据加载器

    Args:
        input_data_dir (str): 输入数据的根目录
        density_map_name (str): 密度图的名称
        batch_size (int): 每个批次的大小
        num_workers (int): 数据加载器使用的线程数

    Returns:
        DataLoader: 验证数据加载器
    """
    # 创建验证数据集
    dataset_infer = CryoData_infer(root=input_data_dir, sub_grid_dir=sub_grid_dir)
    
    # 创建数据加载器
    loader_infer = DataLoader(dataset_infer, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"  | infer loader: {len(loader_infer)} batches")
    return loader_infer

class ConditionalDiffusion(nn.Module):
    def __init__(self, timesteps: int = 1000):
        super(ConditionalDiffusion, self).__init__()
        self.timesteps = timesteps
        
        betas = torch.linspace(0.0001, 0.02, timesteps, device='cuda')
        
        # 计算 alpha 和 alphas_cumprod
        alphas = 1.0 - betas  # 每个时间步的 alpha 值
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)  # 累积乘积
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device='cuda'), self.alphas_cumprod[:-1]])  # 前一个累乘值

        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv3d(66, 16, kernel_size=3, padding=1),  # 现在接受3个通道 (2 for protein + atom + 1 for time embedding)
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        
        # 时间步嵌入
        self.time_embedding = nn.Sequential(
            nn.Linear(1, 64),  # t 是标量，所以输入尺寸为 1
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 4, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
    def forward(self, x, t, condition):
        # 时间嵌入
        t_emb = self.time_embedding(t.view(-1, 1))
        t_emb = t_emb.view(t_emb.size(0), t_emb.size(1), 1, 1, 1)  # [batch_size, 64, 1, 1, 1]
        
        t_emb = t_emb.expand(-1, -1, x.size(2), x.size(3), x.size(4))  # [batch_size, 64, 32, 32, 32]
        
        # 将条件 (protein) 与输入拼接
        x = torch.cat([x, condition, t_emb], dim=1)  # 在通道维度拼接
        
        # 编码过程
        encoded = self.encoder(x)
        
        # 解码过程
        decoded = self.decoder(encoded)
        return decoded
    
    def post_process_output(self, output):
        # 对输出应用 softmax 和 argmax 获取预测类别
        probs = torch.softmax(output, dim=1)  # 应用 softmax
        preds = torch.argmax(probs, dim=1)    # 选择最高概率的类别
        
        # 选择第一个样本，使其最终大小为 (32, 32, 32)
        preds = preds[0] #[32,32,32]
        
        return preds


    def add_noise(self, x, t):
        """向输入 x 添加噪声，根据时间步 t 调整噪声的比例"""
        # 生成与 x 形状相同的随机噪声
        noise = torch.randn_like(x)

        # 确保 t 是整数类型
        t = t.long()  # 转换 t 为长整型

        # 获取 alpha_t 的值并调整形状以进行广播
        alpha_t = self.alphas_cumprod[t]  # 累乘的 alpha 值
        sqrt_alpha_t = torch.sqrt(alpha_t).view(-1, 1, 1, 1, 1)  # 调整形状为 (batch_size, 1, 1, 1, 1)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t).view(-1, 1, 1, 1, 1)  # 调整形状为 (batch_size, 1, 1, 1, 1)

        # 加噪声
        noisy_x = sqrt_alpha_t * x + sqrt_one_minus_alpha_t * noise
        return noisy_x

    def remove_noise(self, x, t, condition):
        """去除噪声，生成干净的输出"""
        return self.forward(x, t, condition)

    def diffusion_step(self, x, t, condition):
        """扩散步骤：在每个时间步 t 执行加噪声和去噪声"""
        # 加噪声
        noisy_x = self.add_noise(x, t)
        # 去噪声
        denoised_x = self.remove_noise(noisy_x, t, condition)
        return denoised_x
    
data_splits = list() 
collect_pred_probs = dict()

def prepare_data(dataset_dir, density_map_name):
    data_splits_old = [splits for splits in os.listdir(dataset_dir) if splits.endswith('.npz')]
    for filename in data_splits_old:
        data_splits.append(filename)

def infer(density_map_splits_dir, input_data_dir, density_map_name, amino_checkpoint, infer_run_on, infer_on_gpu):
    # 设置设备
    device = torch.device(f"cuda:{infer_on_gpu}" if infer_run_on == 'gpu' and torch.cuda.is_available() else 'cpu')
    
    prepare_data(density_map_splits_dir, density_map_name)   
    loader_val = loader_infer(input_data_dir, density_map_splits_dir)
    
    # 加载模型
    model = ConditionalDiffusion().to(device)
    model, _ = load_checkpoint(model, amino_checkpoint, None)
    model.eval()
    
    predicts = []
    idx_val_list = []
    # collect_pred_probs = {}
    
    with torch.no_grad():
        for i, protein in enumerate(loader_val):
            # 生成随机的时间步
            t = torch.randint(0, model.timesteps, (protein.size(0),), device='cuda').float()
            protein = protein.unsqueeze(1)  # [batch_size, 1, 32, 32, 32]

            # 将 protein 数据加载到 GPU 上
            protein = protein.to('cuda')

            # 执行推理，使用扩散步骤
            output = model.diffusion_step(torch.zeros_like(protein), t, protein)  # 使用随机噪声作为初始输入
            # print(output.shape) torch.Size([16, 4, 32, 32, 32])
            
            preds = model.post_process_output(output)
            probs = torch.softmax(output, dim=1)
            # print(preds.shape) torch.Size([16, 20, 20, 20])
            # print(probs.shape) torch.Size([16, 4, 32, 32, 32])

            # 存储预测结果
            predicts.append(preds.cpu().numpy())

            # 生成空的索引值用于保存结果
            idx_val_np = np.empty((32, 32, 32), dtype='S30')  # 占位符
            idx_val_list.append(idx_val_np)

            # 保存每个体素的预测概率
            for i in range(probs.size(2)):
                for j in range(probs.size(3)):
                    for k in range(probs.size(4)):
                        val_prob = probs[0, :, i, j, k].cpu().numpy()
                        collect_pred_probs[f'{i}_{j}_{k}'] = val_prob

    org_map = f"{input_data_dir}/{density_map_name}/emd_normalized_map.mrc"
    with mrcfile.open(org_map, mode='r') as org_map_file:
        recon, idx_val_mat = reconstruct_map(predicts, idx_val_list, org_map_file.data.shape)
        outfilename = f"{input_data_dir}/{density_map_name}/amino_predicted.mrc"
        with mrcfile.new(outfilename, overwrite=True) as mrc:
            mrc.set_data(recon.astype(np.float32))
            mrc.voxel_size = org_map_file.voxel_size
            mrc.header.origin = org_map_file.header.origin
        
    # 保存预测的概率
    file_prob = f"{input_data_dir}/{density_map_name}/{density_map_name}_probabilities_amino.txt"
    save_probs(outfilename, idx_val_mat, file_prob)

    print(f"Inference completed. Results saved in {outfilename}")

def get_xyz(idx, voxel, origin):
    return (idx * voxel) + origin
    
def save_probs(mrc_file, idx_file, file_prob):
    mrc_map = mrcfile.open(mrc_file, mode='r')
    x_origin = mrc_map.header.origin['x']
    y_origin = mrc_map.header.origin['y']
    z_origin = mrc_map.header.origin['z']
    x_voxel = mrc_map.voxel_size['x']
    y_voxel = mrc_map.voxel_size['y']
    z_voxel = mrc_map.voxel_size['z']
    mrc_data = deepcopy(mrc_map.data)
    with open(file_prob, "w") as f:
        for k in range(len(mrc_data[2])):
            for j in range(len(mrc_data[1])):
                for i in range(len(mrc_data[0])):
                    try:
                        if mrc_data[i][j][k] > 0:
                            ids = idx_file[i][j][k]
                            if ids != b'' and ids.strip():  # 仅处理非空的 id
                                ids = ids.decode()
                                value = collect_pred_probs.get(ids, None)
                                if value is not None:
                                    x = round(get_xyz(k, x_voxel, x_origin), 3)
                                    y = round(get_xyz(j, y_voxel, y_origin), 3)
                                    z = round(get_xyz(i, z_voxel, z_origin), 3)
                                    lst = value.tolist()
                                    lst.insert(0, [x, y, z])
                                    json_dump = json.dumps(lst)
                                    final = json_dump[1:-1]
                                    f.writelines(final)
                                    f.writelines('\n')
                                else:
                                    print(f"Warning: No prediction probabilities for ids {ids}")
                            else:
                                print(f"Skipping empty id at position ({i}, {j}, {k})")
                    except UnicodeDecodeError:
                        print("Error", i, j, k)
                        pass
                    except IndexError:
                        pass

def get_manifest_dimensions(image_shape):
    dimensions = [0, 0, 0]
    dimensions[0] = math.ceil(image_shape[0] / core_size) * core_size
    dimensions[1] = math.ceil(image_shape[1] / core_size) * core_size
    dimensions[2] = math.ceil(image_shape[2] / core_size) * core_size
    return dimensions

def reconstruct_map(manifest, idx_val_np, image_shape):
    print(f"Total manifest length: {len(manifest)}")
    
    extract_start = int((box_size - core_size) / 2)
    extract_end = int((box_size - core_size) / 2) + core_size
    dimentions = get_manifest_dimensions(image_shape)

    reconstruct_image = np.zeros((dimentions[0], dimentions[1], dimentions[2]))

    idx_val_mat = np.empty(shape=(dimentions[0], dimentions[1], dimentions[2]), dtype='S30')

    counter = 0
    for z_steps in range(int(dimentions[2] / core_size)):
        for y_steps in range(int(dimentions[1] / core_size)):
            for x_steps in range(int(dimentions[0] / core_size)):
                reconstruct_image[x_steps * core_size:(x_steps + 1) * core_size,
                y_steps * core_size:(y_steps + 1) * core_size, z_steps * core_size:(z_steps + 1) * core_size] = \
                    manifest[counter][extract_start:extract_end, extract_start:extract_end,
                    extract_start:extract_end]

                idx_val_mat[x_steps * core_size:(x_steps + 1) * core_size,
                y_steps * core_size:(y_steps + 1) * core_size, z_steps * core_size:(z_steps + 1) * core_size] = \
                    idx_val_np[counter][extract_start:extract_end, extract_start:extract_end,
                    extract_start:extract_end]

                counter += 1
      
    float_reconstruct_image = np.array(reconstruct_image, dtype=np.float32)
    float_reconstruct_image = float_reconstruct_image[:image_shape[0], :image_shape[1], :image_shape[2]]
    idx_val_np_mat = idx_val_mat[:image_shape[0], :image_shape[1], :image_shape[2]]
    return float_reconstruct_image, idx_val_np_mat

# def sample(
#     model: torch.nn.Module,
#     config: dict,
#     epoch: int = -1
# ):
#     """
#     Generate samples using the given model.

#     Args:
#         model (torch.nn.Module): The model used for sampling.
#         config (dict): Configuration parameters for sampling.
#         epoch (int, optional): The epoch number. Defaults to -1.
#     """
#     if torch.cuda.device_count() > 1:
#         model = model.module
#     model.eval()

#     # sample
#     molecules_xyz = model.sample(
#         grid_dim=config["grid_dim"],
#         n_batch_chains=config["n_chains"],
#         n_repeats=config["repeats_wjs"],
#         n_steps=config["steps_wjs"],
#         max_steps=config["max_steps_wjs"],
#         warmup_steps=config["warmup_wjs"],
#         refine=True,
#     )

#     # save molecules on xyz format
#     dirname_out = os.path.join(config["output_dir"], "samples/", f"epoch={epoch}/")
#     print(f">> saving samples in {dirname_out}")
#     makedir(dirname_out)
#     save_molecules_xyz(molecules_xyz, dirname_out)


if __name__ == "__main__":
    density_map_splits_dir = sys.argv[1]
    input_data_dir = sys.argv[2]
    density_map_name = sys.argv[3]
    amino_checkpoint = sys.argv[4]
    infer_run_on = sys.argv[5]
    output_dir = '/gpfs/share/home/2201111701/szs/szs1/voxmol/output_infer'
    # infer_run_gpu = int(sys.argv[6])
    
    # 定义 infer_on_gpu，确保从命令行参数 sys.argv[6] 获取并转换为整数
    if len(sys.argv) > 6:
        infer_on_gpu = int(sys.argv[6])
    else:
        infer_on_gpu = 2  # 如果没有传入 infer_on_gpu 参数，则默认使用 0（即 GPU 0）
    
    infer(density_map_splits_dir, input_data_dir, density_map_name, amino_checkpoint, infer_run_on, infer_on_gpu)