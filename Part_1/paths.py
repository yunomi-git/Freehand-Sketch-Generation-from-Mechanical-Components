import os
import tkinter.filedialog
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
# import torch
from pathlib import Path

HOME_PATH = os.path.dirname(os.path.abspath(__file__)) + "/"
MODELS_PATH = HOME_PATH + "heuristic_prediction/checkpoints/"
RAW_DATASETS_PATH = HOME_PATH + "../Datasets/"

ONSHAPE_STL_PATH = HOME_PATH + "../Datasets/Onshape_STL_Dataset/"
# THINGIVERSE_STL_PATH = HOME_PATH + "../Thingiverse_STL_Dataset/"
THINGIVERSE_STL_PATH = HOME_PATH + "../Datasets/Dataset_Thingiverse_10k/"

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


def get_files_in_directory(base_folder, max_files_per_subfolder, absolute=True):
    pass

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

def select_file(init_dir=MODELS_PATH,
                choose_type="file" # file, folder
                ):
    # choose file vs folder
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    if choose_type == "file":
        filename = askopenfilename(initialdir=init_dir,
                                   defaultextension="txt")  # show an "Open" dialog box and return the path to the selected file
        return filename
    else:
        foldername = tkinter.filedialog.askdirectory(initialdir=init_dir)
        return foldername + "/"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)