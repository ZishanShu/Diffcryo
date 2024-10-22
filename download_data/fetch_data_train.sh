#!/usr/bin/env bash

# 定义文件列表
file_list="27220"

# 定义目标目录
base_dir="/gpfs/share/home/2201111701/szs/szs1/voxmol/dataset/train"

# 遍历文件列表中的每个ID
for i in $file_list; do
  # 创建一个新的子目录，例如 /gpfs/share/home/2201111701/szs/szs1/voxmol/dataset/27661
  target_dir="$base_dir/$i"
  mkdir -p $target_dir
  
  # 定义下载URL
  URL="ftp://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-$i/map/emd_$i.map.gz"
  
  # 下载文件到目标子目录
  curl -o "$target_dir/emd_$i.map.gz" $URL
  
  # 解压下载的.gz文件到目标子目录
  gunzip "$target_dir/emd_$i.map.gz"
  
  # 检查解压是否成功，如果是，则删除.gz文件
  if [ -f "$target_dir/emd_$i.map" ]; then
    echo "解压成功，删除压缩包：$target_dir/emd_$i.map.gz"
  fi
done
