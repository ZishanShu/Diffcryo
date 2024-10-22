#!/usr/bin/env bash

echo "Script started for data preparation"

# 获取数据集目录作为第一个参数
relative_path=$1

# 确定数据集目录的绝对路径
density_map_dir=$(readlink -f "$relative_path")

# 确定脚本目录
script_dir="/gpfs/share/home/2201111701/szs/szs1/voxmol/download_data"

for dir in "$density_map_dir"/*/; do
  echo "Running for density maps present in: $dir"
  
  # 对每个子目录运行 Python 脚本，而不是具体的 .map 文件
  python3 "$script_dir/get_resample_map.py" "$dir"
  python3 "$script_dir/get_normalize_map.py" "$dir"
  python3 "$script_dir/get_atoms_label.py" "$dir"
  python3 "$script_dir/get_amino_labels.py" "$dir"
  # python3 "$script_dir/grid_division.py" "$dir"
  
  echo "Done with maps in directory: $dir"
done

echo "ALL DONE!"