import pickle
from pathlib import Path
import cairosvg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import io
from paths import DirectoryPathManager
import paths

dataset_root = "../../CachedDatasets/Freehand/"
pickle_paths = dataset_root + "logs/my_data01/"
image_root = dataset_root + "vsm/"
svg_root = pickle_paths + "svgs/"

pickle_manager = DirectoryPathManager(pickle_paths, base_unit_is_file=True, max_depth=0)
pickle_files = pickle_manager.get_files_absolute()

output_path = pickle_paths + "_merged.pkl"
paths.mkdir(output_path)
merged_sketches = {}
num_sketches_found = 0

for file_name in pickle_files:
    path_name = file_name[file_name.rfind("/")+1:file_name.rfind(".")]
    with open(file_name, 'rb') as f:
        sketches = pickle.load(f)

    num_sketches_found += len(sketches.keys())
    for key in sketches.keys():
        merged_sketches[key] = sketches[key]

num_sketches_saved = len(merged_sketches.keys())

print("found:", num_sketches_found, "| saved:", num_sketches_saved)
with open(output_path, 'wb') as f:
    pickle.dump(merged_sketches, f)

    # plt.show()
    # print(orig_path)
    # result = str(Path(*Path(orig_path).parts[-3:]))
    # print(result)