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
pickles = pickle_manager.get_files_absolute()

save_folder = "clipasso_out/"
paths.mkdir(save_folder)

for path in pickles[:5]:
    # path = pickle_paths + "path_0.pkl"
    path_name = path[path.rfind("/")+1:path.rfind(".")]
    with open(path, 'rb') as f:
        x = pickle.load(f)

    for key in x.keys():
        # print(path, path_name)
        # print(key)
        save_name = save_folder + path_name + "_" + key + ".png"

        # print(x)
        fig, axs = plt.subplots(2)
        # print(x["0_0"].keys())
        image_name = x[key]["img_path"]
        image_path = image_root + image_name
        ax = axs[0]
        image = mpimg.imread(image_path)
        ax.imshow(image)
        ax = axs[1]
        svg_path = image_name[image_name.find("/")+1:image_name.find(".")] + "/2000.svg"
        svg_path = svg_root + svg_path
        png_bytes = cairosvg.svg2png(url=svg_path)
        img = Image.open(io.BytesIO(png_bytes))
        ax.imshow(img)
        plt.savefig(save_name)

    # plt.show()
    # print(orig_path)
    # result = str(Path(*Path(orig_path).parts[-3:]))
    # print(result)