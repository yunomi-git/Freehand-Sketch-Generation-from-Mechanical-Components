import pickle
from pathlib import Path

path = "../../../CachedDatasets/Freehand/logs/my_data0/path_0.pkl"
with open(path, 'rb') as f:
    x = pickle.load(f)

# print(x["0_0"].keys())
orig_path = x["0_0"]["img_path"]
print(orig_path)
result = str(Path(*Path(orig_path).parts[-3:]))
print(result)