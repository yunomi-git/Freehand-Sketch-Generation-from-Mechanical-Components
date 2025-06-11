import os
from pathlib import Path

HOME_PATH = os.path.dirname(os.path.abspath(__file__)) + "/"
RAW_DATASETS_PATH = HOME_PATH + "../Datasets/"
DATA_PATH = HOME_PATH + "../CachedDatasets/"

def mkdir(path):
    if "." in path:
        Path(path[:path.rfind("/") + 1]).mkdir(parents=True, exist_ok=True)
    else:
        Path(path).mkdir(parents=True, exist_ok=True)

def get_files_multifolder(base_folder, files_per_folder):
    files = []
    folders = os.listdir(base_folder)
    for folder in folders:
        contents = os.listdir(base_folder + "/" + folder)
        contents.sort()
        contents = [folder + "/" + file for file in contents]
        max_files = files_per_folder
        if max_files > len(contents):
            max_files = len(contents)
        files.extend(contents[:max_files])
    return files


def get_files_in_folders(base_folder, folders=None, files_per_folder=10000, per_folder_subfolder=""):
    # Grabs files within a given folder
    # Assumes the folder structure is base_folder > [folders] > per_folder_subfolder > file
    files = []
    if folders is None:
        folders = os.listdir(base_folder)
    for folder in folders:
        contents = os.listdir(base_folder + "/" + folder + "/" + per_folder_subfolder)
        contents.sort()
        contents = [folder + "/" + per_folder_subfolder + "/" + file for file in contents]
        max_files = files_per_folder
        if max_files > len(contents):
            max_files = len(contents)
        files.extend(contents[:max_files])
    return files


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)