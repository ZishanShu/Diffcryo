import os
import sys
import requests
from time import sleep
import shutil

def download_pdb_file(pdb_id, dir_path, retries=3):
    """下载 PDB 文件的函数，包含重试机制。"""
    pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    
    for attempt in range(retries):
        try:
            pdb_response = requests.get(pdb_url)
            if pdb_response.status_code == 200:
                pdb_filename = os.path.join(dir_path, f"{pdb_id}.pdb")
                with open(pdb_filename, "wb") as file:
                    file.write(pdb_response.content)
                print(f"Successfully downloaded PDB file: {pdb_filename}")
                return True
            elif pdb_response.status_code == 404:
                print(f"PDB file for PDB ID {pdb_id} not found (404). Skipping download.")
                return False
            else:
                print(f"Attempt {attempt + 1} - Error downloading the PDB file for PDB ID: {pdb_id}, Status code: {pdb_response.status_code}")
                sleep(2)  # 等待一会儿再重试
        except Exception as e:
            print(f"Attempt {attempt + 1} - Exception occurred while downloading {pdb_id}: {e}")
            sleep(2)  # 等待一会儿再重试
    print(f"Failed to download PDB file for PDB ID: {pdb_id} after {retries} attempts.")
    return False

def get_pdb(em_path):
    # 获取子目录
    dir_names = [os.path.join(em_path, m) for m in os.listdir(em_path) if os.path.isdir(os.path.join(em_path, m))]
    print("Length of maps:", len(dir_names))
    
    for dir_path in dir_names:
        # 检查目录中是否已存在 .pdb 文件，如果存在则跳过
        if any(file.endswith('.pdb') for file in os.listdir(dir_path)):
            print(f"PDB file already exists in directory {dir_path}. Skipping download.")
            continue
        
        emdb_id = os.path.basename(dir_path)  # 取得目录名作为EMDB ID
        pdb_downloaded = False  # 标志变量，表示是否下载了PDB文件
        
        # Retrieve the PDB structures associated with the EMDB entry
        api_url = f"https://www.ebi.ac.uk/emdb/api/entry/{emdb_id}"
        response = requests.get(api_url)

        print(f"Processing EMDB ID: {emdb_id}")  # 调试输出：当前处理的EMDB ID
        print(f"API Response Status: {response.status_code}")  # 调试输出：API响应状态码

        if response.status_code == 200:
            data = response.json()
            print(f"API Response Data: {data}")  # 调试输出：API返回的完整数据

            if "crossreferences" in data and "pdb_list" in data["crossreferences"]:
                pdb_entries = data["crossreferences"]["pdb_list"]

                if "pdb_reference" in pdb_entries and isinstance(pdb_entries["pdb_reference"], list) and len(pdb_entries["pdb_reference"]) > 0:
                    pdb_id = pdb_entries["pdb_reference"][0]['pdb_id']
                    pdb_downloaded = download_pdb_file(pdb_id, dir_path)  # 调用下载函数
        else:
            print(f"Error retrieving data from the EMDB API for ID: {emdb_id}. Status code: {response.status_code}")
        
        # 如果没有下载任何PDB文件，则删除整个文件夹
        if not pdb_downloaded:
            print(f"No PDB files were downloaded for EMDB ID {emdb_id}. Deleting the directory: {dir_path}")
            # 检查目录是否存在再删除
            if os.path.exists(dir_path):
                try:
                    shutil.rmtree(dir_path)
                except FileNotFoundError:
                    print(f"Directory not found: {dir_path}. It may have already been deleted.")
                except Exception as e:
                    print(f"Error deleting directory {dir_path}: {e}")
            else:
                print(f"Directory does not exist: {dir_path}. Skipping deletion.")

if __name__ == "__main__":
    emd_path = sys.argv[1]
    get_pdb(emd_path)
