import torch
import torch.nn as nn  # 确保导入的路径正确
from voxmol.dataset import create_loader  # 确保导入的路径正确
from voxmol.options import parse_args  # 确保导入的路径正确
import matplotlib.pyplot as plt

class ConditionalDiffusion(nn.Module):
    def __init__(self, timesteps: int = 1000):
        super(ConditionalDiffusion, self).__init__()
        self.timesteps = timesteps
        
        # 初始化 beta 序列，通常是从较小值到较大值的线性或平方等变化
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
            nn.ConvTranspose3d(16, 1, kernel_size=3, padding=1),
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
        probs = torch.softmax(output, dim=1)  # 应用 softmax
        preds = torch.argmax(probs, dim=1)  # 选择最高概率的类别
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


def run_diffusion_process_1000_steps(config):
    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalDiffusion().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    # 创建数据加载器并加载单个数据样本
    loader, _ = create_loader(config)
    protein, atom = next(iter(loader))
    
    # 模型验证逻辑
    model.eval()

    with torch.no_grad():
        # 初始化时间步 t
        t = torch.randint(0, model.timesteps, (atom.size(0),), device='cuda').float()

        # 数据处理
        protein = protein.unsqueeze(1).to(device)  # [batch_size, 1, 32, 32, 32]
        atom = atom.unsqueeze(1).to(device)  # [batch_size, 1, 32, 32, 32]

        # 初始化噪声样本（随机噪声输入）
        noisy_sample = torch.randn_like(atom).to(device)

        # 执行 1000 次扩散步骤
        for step in range(1000):
            t_step = torch.full((atom.size(0),), step, device=device).float()
            noisy_sample = model.diffusion_step(noisy_sample, t_step, protein)

        # 后处理以获取类别标签
        output = model.diffusion_step(atom, t, protein)
        
        # 计算恢复的准确率
        accuracy = (output == atom.squeeze(1)).float().mean().item()
        print(f"Recovery Accuracy after 1000 steps: {accuracy * 100:.2f}%")
        
        # 可视化原始标签与恢复后的标签
        original_volume = atom.squeeze(1)[0]  # 可视化第一个 batch
        recovered_volume = output.squeeze(1)[0]  # 可视化恢复的第一个 batch

        visualize_results_3d_slices(original_volume, title="Original Volume Slices")
        plt.savefig("original_slices.png")  # 保存到当前文件夹

        visualize_results_3d_slices(recovered_volume, title="Recovered Volume Slices")
        plt.savefig("recovered_slices.png")  # 保存到当前文件夹

        
def visualize_results_3d_slices(volume, title="Volume Slices", num_slices=5):
    """
    可视化 3D 张量（深度、高度、宽度）的切片。

    Args:
        volume (torch.Tensor or np.ndarray): 形状为 [depth, height, width] 的 3D 张量。
        title (str): 图的标题。
        num_slices (int): 要显示的切片数量（从每个维度取切片）。
    """
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().numpy()

    depth, height, width = volume.shape

    # 在每个维度中均匀取几个切片
    slice_indices = [depth // num_slices * i for i in range(num_slices)]

    fig, axes = plt.subplots(3, num_slices, figsize=(15, 10))
    fig.suptitle(title)

    # 显示深度方向的切片
    for i, idx in enumerate(slice_indices):
        axes[0, i].imshow(volume[idx, :, :], cmap='viridis')
        axes[0, i].set_title(f"Depth Slice {idx}")
        axes[0, i].axis('off')

    # 显示高度方向的切片
    for i, idx in enumerate(slice_indices):
        axes[1, i].imshow(volume[:, idx, :], cmap='viridis')
        axes[1, i].set_title(f"Height Slice {idx}")
        axes[1, i].axis('off')

    # 显示宽度方向的切片
    for i, idx in enumerate(slice_indices):
        axes[2, i].imshow(volume[:, :, idx], cmap='viridis')
        axes[2, i].set_title(f"Width Slice {idx}")
        axes[2, i].axis('off')

    plt.tight_layout()

if __name__ == "__main__":
    # 配置
    config = parse_args()
    
    # 运行扩散过程 1000 次
    run_diffusion_process_1000_steps(config)