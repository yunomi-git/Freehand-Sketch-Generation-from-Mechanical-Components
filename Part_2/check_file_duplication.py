from paths import DirectoryPathManager
import pickle

class DictList:
    def __init__(self):
        self.dictionary = {}

    def add_to_key(self, key, value):
        if key not in self.dictionary:
            self.dictionary[key] = []
        self.dictionary[key].append(value)

    def get_key(self, key):
        if key not in self.dictionary:
            return None

        return self.dictionary[key]

    def key_exists(self, key):
        return key in self.dictionary

if __name__=="__main__":
    root_folder = "/home/ubuntu/nomi-fs/Projects/CachedDatasets/Freehand/logs/my_data01/"
    # log_folder = "logs/my_data01/"

    base_folder = root_folder

    # find all pickle files
    folder_manager = DirectoryPathManager(base_folder, base_unit_is_file=True, max_depth=0)
    files = folder_manager.get_files_absolute()
    unique_paths = DictList()
    for file in files:
        with open(file, 'rb') as f:
            output = pickle.load(f)
        for key in output.keys():
            path = output[key]["img_path"]
            unique_paths.add_to_key(path, 1)
    
    print(len(unique_paths.dictionary.keys()))