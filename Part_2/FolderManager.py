# Contents of a folder
import os
from pathlib import Path
import paths
from paths import DirectoryPathManager
from paths import FilePath



if __name__=="__main__":
    base_path = paths.HOME_PATH + "../Datasets/MCB_A/"
    directory_manager = DirectoryPathManager(base_path=base_path, max_files_per_subfolder=100,
                                             base_unit_is_file=True)
    absolutes = directory_manager.get_files_absolute()
    relative = directory_manager.get_files_relative(extension=False)

    folder_directory_manager = DirectoryPathManager(base_path=base_path, max_files_per_subfolder=100,
                                                    base_unit_is_file=False)
    folder_absolutes = folder_directory_manager.get_files_absolute()
    folder_relative = folder_directory_manager.get_files_relative(extension=False)

    print("a")