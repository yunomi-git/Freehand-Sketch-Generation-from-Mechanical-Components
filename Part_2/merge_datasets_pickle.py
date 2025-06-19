import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import io
from paths import DirectoryPathManager
import paths
from tqdm import tqdm

dataset_root = "../../CachedDatasets/Freehand/logs/"
postfix_list = ["00", "01"]
get_dataset_file = lambda postfix: f"{dataset_root}my_data{postfix}/_merged{postfix}.pkl"

output_path = dataset_root + "_merged.pkl"

merged_sketches = {}
num_sketches_found = 0

index = 0
for postfix in tqdm(postfix_list):
    dataset_file = get_dataset_file(postfix)
    with open(dataset_file, 'rb') as f:
        sketches = pickle.load(f)

    num_sketches_found += len(sketches.keys())
    for key in sketches.keys():
        # Fix the image path
        item = sketches[key]
        item["img_path"] = str(Path(*Path(item["img_path"]).parts[-3:]))
        # Fix the index
        orig_idx, seed = key.split("_")
        merged_sketches[f"{index}_{seed}"] = sketches[key]
        index += 1

num_sketches_saved = len(merged_sketches.keys())

print("found:", num_sketches_found, "| saved:", num_sketches_saved)
with open(output_path, 'wb') as f:
    pickle.dump(merged_sketches, f)

    # plt.show()
    # print(orig_path)
    # result = str(Path(*Path(orig_path).parts[-3:]))
    # print(result)