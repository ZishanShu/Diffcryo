import os
import time
import getpass as gt
import wandb
import torch
import torch.nn as nn

from voxmol.options import parse_args
from voxmol.models import create_model
from voxmol.dataset import create_loader
from voxmol.metrics import create_metrics, MetricsDenoise
from voxmol.models.ema import ModelEma
from voxmol.models.adamw import AdamW
from voxmol.utils import seed_everything, save_checkpoint, load_checkpoint, makedir, save_molecules_xyz

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
    
def main():
    # ----------------------
    # basic inits
    config = parse_args()
    print(">> n gpus available:", torch.cuda.device_count())
    torch.set_default_dtype(torch.float32)
    seed_everything(config["seed"])
    if config["wandb"] > 0:
        wandb.init(
            project="voxmol",
            entity=gt.getuser(),
            config=config,
            name=config["exp_name"],
            dir=config["output_dir"],
            settings=wandb.Settings(code_dir=".")
        )

    # ----------------------
    # data loaders
    start_epoch = 0
    loader_train, loader_val = create_loader(config)

    # ----------------------
    # voxelizer, model, criterion, optimizer, scheduler
    device = torch.device("cuda")
    # voxelizer = Voxelizer(
    #     grid_dim=config["grid_dim"],
    #     num_channels=len(config["elements"]),
    #     device=device,
    # )
    model = ConditionalDiffusion().to('cuda')
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["wd"],
        betas=[0.99, 0.999],
    )
    optimizer.zero_grad()

    # optionally resume training
    if config["resume"] is not None:
        model, optimizer, start_epoch = load_checkpoint(
            model, config["output_dir"], optimizer, best_model=False
        )
        
        os.system(f"cp {os.path.join(config['output_dir'], 'checkpoint.pth.tar')} " +
                  f"{os.path.join(config['output_dir'], f'checkpoint_{start_epoch}.pth.tar')}")

    # ema (exponential moving average)
    # model_ema = ModelEma(model, decay=.999)

    # ----------------------
    # metrics
    metrics = create_metrics()

    # ----------------------
    # start training
    print(">> start training...")
    for epoch in range(start_epoch, start_epoch + config["num_epochs"]):
        t0 = time.time()

        # train
        train_metrics = train(
            loader_train, model, criterion, optimizer, metrics, config
        )

        # val
        val_metrics = val(
            loader_val, model, criterion, metrics, config
        )

        # sample
        if epoch > 0 and epoch % 50 == 0:
            print(f"| sampling at epoch {epoch}")
            sample(model, config, epoch)

        # print metrics, log wandb
        print_metrics(epoch, time.time()-t0, train_metrics, val_metrics)
        if config["wandb"] > 0:
            wandb.log({"train": train_metrics, "val": val_metrics, "sampling": None})

        # save model
        save_checkpoint({
            "epoch": epoch + 1,
            "config": config,
            # "state_dict_ema": model_ema.module.state_dict(),
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, config=config)


def train(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    # model_ema: ModelEma,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: MetricsDenoise,
    config: dict,
):
    """
    Trains the model using the given data loader, voxelizer, model, criterion,
    optimizer, and metrics.

    Args:
        loader (torch.utils.data.DataLoader): The data loader for loading the training data.
        voxelizer (Voxelizer): The voxelizer for converting input data into voxels.
        model (torch.nn.Module): The model to be trained.
        model_ema (ModelEma): The exponential moving average of the model parameters.
        criterion (torch.nn.Module): The loss function for calculating the training loss.
        optimizer (torch.optim.Optimizer): The optimizer for updating the model parameters.
        metrics (MetricsDenoise): The metrics object for tracking the training metrics.
        config (dict): The configuration dictionary containing training settings.

    Returns:
        dict: The computed metrics for the training process.
    """
    metrics.reset()
    model.train()
    
    
    for i, (protein, atom) in enumerate(loader):
        t = torch.randint(0, model.module.timesteps, (atom.size(0),), device='cuda').float()
        protein = protein.unsqueeze(1)
        atom = atom.unsqueeze(1)
        
        protein = protein.to('cuda')
        atom = atom.to('cuda')
        
        # forward/backward
        output = model.module.diffusion_step(atom, t, protein) # (batch_size, 4, 32, 32, 32)
        # print(f"Output shape: {output.shape}, range: ({output.min().item()}, {output.max().item()})")
        
        preds = model.module.post_process_output(output) # (batch_size, 32, 32, 32)
        
        loss = criterion(output, atom.squeeze(1).long())  # 直接使用atom作为真实值
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # model_ema.update(model)
        accuracy = (preds == atom).float().mean()  # 计算准确率

        # update metrics
        metrics.update(loss, output, atom.squeeze(1))

        if i*config["batch_size"] >= 100_000:
            break
        if config["debug"] and i == 10:
            break

    return metrics.compute()


def val(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    metrics: MetricsDenoise,
    config: dict,
):
    """
    Perform validation on the given data loader using the provided model and criterion.

    Args:
        loader (torch.utils.data.DataLoader): Data loader for validation data.
        voxelizer (Voxelizer): Voxelizer object for converting input data to voxels.
        model (torch.nn.Module): Model to be used for prediction.
        criterion (torch.nn.Module): Loss criterion for calculating the loss.
        metrics (MetricsDenoise): Metrics object for tracking evaluation metrics.
        config (dict): Configuration dictionary containing various settings.

    Returns:
        float: Computed metrics for the validation data.
    """
    metrics.reset()
    model.eval()

    with torch.no_grad():
        for i, (protein, atom) in enumerate(loader):
            t = torch.randint(0, model.module.timesteps, (atom.size(0),), device='cuda').float()
            # voxelize
            # voxels = voxelizer(batch)
            protein = protein.unsqueeze(1) # [batch_size, 1, 32, 32, 32]
            atom = atom.unsqueeze(1) # [batch_size, 1, 32, 32, 32]
            
            protein = protein.to('cuda')
            atom = atom.to('cuda')
            
            # forward
            output = model.module.diffusion_step(atom, t, protein)
            preds = model.module.post_process_output(output) # (batch_size, 32, 32, 32)

            loss = criterion(output, atom.squeeze(1).long())  # 直接使用atom作为真实值

            # update metrics
            metrics.update(loss, output, atom.squeeze(1))
            
            if config["debug"] and i == 10:
                break
    return metrics.compute()


def sample(
    model: torch.nn.Module,
    config: dict,
    epoch: int = -1
):
    """
    Generate samples using the given model.

    Args:
        model (torch.nn.Module): The model used for sampling.
        config (dict): Configuration parameters for sampling.
        epoch (int, optional): The epoch number. Defaults to -1.
    """
    if torch.cuda.device_count() > 1:
        model = model.module
    model.eval()

    # sample
    molecules_xyz = model.sample(
        grid_dim=config["grid_dim"],
        n_batch_chains=config["n_chains"],
        n_repeats=config["repeats_wjs"],
        n_steps=config["steps_wjs"],
        max_steps=config["max_steps_wjs"],
        warmup_steps=config["warmup_wjs"],
        refine=True,
    )

    # save molecules on xyz format
    dirname_out = os.path.join(config["output_dir"], "samples/", f"epoch={epoch}/")
    print(f">> saving samples in {dirname_out}")
    makedir(dirname_out)
    save_molecules_xyz(molecules_xyz, dirname_out)


def print_metrics(
    epoch: int,
    time: float,
    train_metrics: list,
    val_metrics: list,
    sampling_metrics: dict = None,
):
    """
    Print the metrics for each epoch.

    Args:
        epoch (int): The current epoch number.
        time (float): The time taken for the epoch.
        train_metrics (list): The metrics for the training set.
        val_metrics (list): The metrics for the validation set.
        sampling_metrics (dict, optional):The metrics for the sampling.Defaults to None.
    """
    all_metrics = [train_metrics, val_metrics, sampling_metrics]
    metrics_names = ["train", "val", "sampling"]

    str_ = f">> epoch: {epoch} ({time:.2f}s)"
    for (split, metric) in zip(metrics_names, all_metrics):
        if metric is None:
            continue
        str_ += "\n"
        str_ += f"[{split}]"
        for k, v in metric.items():
            if k == "loss":
                str_ += f" | {k}: {v:.4f}"
            else:
                str_ += f" | {k}: {v:.4f}"
    print(str_)


# def add_noise_voxels(voxels: torch.Tensor, sigma: float):
#     """
#     Adds Gaussian noise to the input voxels.

#     Args:
#         voxels (torch.Tensor): Input tensor representing the voxels.
#         sigma (float): Standard deviation of the Gaussian noise.

#     Returns:
#         torch.Tensor: Tensor with Gaussian noise added to the input voxels.
#     """
#     if sigma > 0:
#         noise = torch.randn_like(voxels) * sigma
#         return voxels + noise
#     return voxels

if __name__ == "__main__":
    main()
