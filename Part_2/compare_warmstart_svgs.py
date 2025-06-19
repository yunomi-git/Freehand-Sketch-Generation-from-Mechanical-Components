from matplotlib import pyplot as plt
import cairosvg
import paths
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import io
import paths

root_1 = paths.DATA_PATH + "Freehand/logs/warmstart00/svgs/"
root_2 = paths.DATA_PATH + "Freehand/logs/my_data00/svgs/"
iters = [0, 50, 100, 200, 400, 700, 1000, 1500, 2000]
out_folder = "warmstart_out/"
paths.mkdir(out_folder)
folder_manager = paths.DirectoryPathManager(root_1, base_unit_is_file=False)

for folder in folder_manager.get_files_relative(extension=False):
    fig, axs = plt.subplots(2, len(iters))
    fig.set_figheight(3)
    fig.set_figwidth(3 * len(iters) // 2)

    for i in range(len(iters)):
        svg_file = str(iters[i]) + ".svg"
        path_1 = root_1 + folder +  "/" + svg_file
        path_2 = root_2 + folder +  "/" + svg_file
        ax = axs[0, i]
        png_bytes = cairosvg.svg2png(url=path_1)
        img = Image.open(io.BytesIO(png_bytes))
        ax.imshow(img)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax = axs[1, i]
        png_bytes = cairosvg.svg2png(url=path_2)
        img = Image.open(io.BytesIO(png_bytes))
        ax.imshow(img)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_xlabel(str(iters[i]))


    paths.mkdir(out_folder + folder + ".png")
    plt.savefig(out_folder + folder + ".png",  bbox_inches='tight')