"""
Runs ChimeraX in no-GUI mode to resample map.
"""
import sys
import subprocess
import os


def execute(input_path, chimera_path):
    """
    Creates resampling script and executes them in ChimeraX.
    """
    # 确保输入路径是一个有效目录
    if not os.path.isdir(input_path):
        print(f"### Error: {input_path} is not a directory or does not exist. ###")
        return
    
    # 列出输入路径中的所有子目录
    map_names = [fn for fn in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, fn))]

    # 检查子目录是否存在
    if not map_names:
        print(f"### Please check the directory!! No input files present in {input_path} ###")
        return

    for map_name in map_names:
        # 生成每个子目录的完整路径
        path = os.path.join(input_path, map_name)
        
        # 列出子目录中的所有 .map 文件
        emd_map = [e for e in os.listdir(path) if e.endswith(".map")]
        
        if not emd_map:  # 如果没有找到任何 .map 文件
            print(f"No .map files found in {path}. Skipping...")
            continue
        
        for density_map in emd_map:
            # 创建 resample.cxc 脚本文件
            resample_script_path = 'resample.cxc'
            with open(resample_script_path, 'w') as chimera_scripts:
                chimera_scripts.write(
                    f'open {os.path.join(path, density_map)}\n'
                    'vol resample #1 spacing 1.0\n'
                    f'save {os.path.join(path, density_map.split("_")[0] + "_resampled_map.mrc")} model #2\n'
                    'exit\n'
                )

            # 执行 ChimeraX 脚本
            try:
                subprocess.run([chimera_path, '--nogui', resample_script_path], check=True)
                print(f'### Resampled {density_map} and saved on new grid with voxel size of 1 ###')
            except subprocess.CalledProcessError as e:
                print(f"Error while running ChimeraX: {e}")
            finally:
                # 删除生成的脚本文件
                os.remove(resample_script_path)


if __name__ == "__main__":
    input_path = sys.argv[1]

    if len(sys.argv) > 2:
        chimera_path = sys.argv[2]
    else:
        chimera_path = '/gpfs/share/home/2201111701/szs/szs1/chimerax/usr/bin/chimerax'

    execute(input_path, chimera_path)
    print("Resampling Complete!")
