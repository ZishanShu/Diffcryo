# Diffcryo

# Install

```jsx
conda env create -f diffcryo_env.yml

conda activate diffcryo
```

# Download dataset

download_data/fetch_data_xxx.sh（EMDB下载map的meta data）

download_data/download_pdb.py（根据map id搜pdbid在PDBbank里下map对应的PDB）

download_data/download_fasta.py（根据PDB id搜pdbid在PDBbank里下PDB对应的fasta）

# Data processing

## intro

data_processing.sh

```bash
python3 "$script_dir/get_resample_map.py" "$dir"
python3 "$script_dir/get_normalize_map.py" "$dir"
python3 "$script_dir/get_atoms_label.py" "$dir"
python3 "$script_dir/get_amino_labels.py" "$dir"
```

1、resample

download_data/get_resample_map.py（通过`vol resample #1 spacing 1.0`，对体积数据集（#1）重采样，并且将体素之间的间距设置为 1.0，修改分辨率。）

2、normalize

`percentile = np.percentile(map_data[np.nonzero(map_data)], 95)`

- 使用 `numpy` 的 `percentile` 函数，计算体积数据中非零值的第95百分位数，用于后续的归一化处理。

`map_data /= percentile`

- 将 `map_data` 中的所有数据值除以95百分位数，实现归一化操作。

然后：

- `map_data[map_data < 0] = 0`
    - 将 `map_data` 中所有小于 0 的值设置为 0。
- `map_data[map_data > 1] = 1`
    - 将 `map_data` 中所有大于 1 的值设置为 1。❌cryo2 struct有误

我的操作：

对于后续分布  用了min-max归一化  再映射到【0-1】范围内

![image.png](Diffcryo%20Repo%2011f55c26dd9980c5b98df3c6c25d41d4/image.png)

3、分别label atom和amino

最后再download_data/grid_division.py（把任意【x*y*z】的map切分为很多个【32，32，32】的grids）便于输入网络

```jsx
ls train_sub_grids > train_splits.txt
ls valid_sub_grids > valid_splits.txt
```

生成后面index的.txt

## code

下载map数据集的代码：

download_data/fetch_data.sh

下载PDB的代码：
python download_data/download_pdb.py dataset/train

下载fasta的代码：

python3 get_fasta_from_rcsb.py dataset/train

conda activate vox_mol

再跑数据处理：

download_data/data_processing.sh

./data_processing.sh dataset

emd_0004.map：原始冷冻电镜密度图。

emd_normalized_map.mrc：归一化的冷冻电镜密度图。 

atom_emd_normalized_map.mrc：标记了原子位置的冷冻电镜密度图。 

atom_ca_emd_normalized_map.mrc：仅标记了碳-α（Cα）原子的冷冻电镜密度图。

amino_emd_normalized_map.mrc：标记了氨基酸的冷冻电镜密度图。

sec_struc_emd_normalized_map.mrc：标记了二级结构的冷冻电镜密度图。 （目前🈚️）

6giq.pdb：冷冻电镜密度的PDB文件。

6giq_helix.pdb：从PDB文件中提取的螺旋结构。 （目前🈚️）

6giq_coil.pdb：从PDB文件中提取的无规则卷曲结构。 （目前🈚️）

6giq_strand.pdb：从PDB文件中提取的链状结构。 （目前🈚️）

6giq.fasta：FASTA格式的原始蛋白质序列。 

6giq_all_chain_combined.fasta：原始序列中所有链的组合序列，FASTA格式。 （目前🈚️）

atomic.fasta：从PDB结构中提取的序列，FASTA格式。 （目前🈚️）

dealign_clustal_input.fasta：用于Clustal Omega工具的输入文件。 （目前🈚️）

dealign_clustal_output.fasta：由Clustal Omega工具生成的原始序列与从蛋白质结构中提取的序列之间的比对结果，FASTA格式。（目前🈚️）

目前二级结构的data processing还未涉及

# Train

```jsx
cd ~/szs/szs1/voxmol

source activate vox_mol

CUDA_VISIBLE_DEVICES=0,1,2 python voxmol/cryo_training_amino.py
CUDA_VISIBLE_DEVICES=3,4,5 python voxmol/cryo_training_atom.py
```

大概300s一个epoch

改数据集路径在

dataset/**init**.py

我们使用分布式数据并行 (DDP) 技术在每个配备了 6 个 NVIDIA V100 GPU（每个 32GB 内存）的 24 个计算节点上训练模型。

```bash
cd ~/szs/szs1/voxmol

source activate vox_mol

CUDA_VISIBLE_DEVICES=3 python voxmol/cryo_training.py
```

# Inference：

单独跑infer：

```jsx
python3 infer/atom_inference.py dataset_test/valid_sub_grids dataset_test 26801 exps/voxmol_experiment gpu 0
python3 infer/amino_inference.py dataset_test/valid_sub_grids dataset_test 26801 exps/voxmol_experiment gpu 0
```

# Sampling：

```jsx
python3 inference_sampling.py
```
