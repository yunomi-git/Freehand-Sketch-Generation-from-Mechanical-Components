import os
import time
import pickle
from pathlib import Path
import cairosvg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import io
from paths import DirectoryPathManager
import paths
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import paths
from datasets import get_dataset
from models.LBS import SketchModel
from models.clip_loss import Loss as CLIPLoss
from loss import LBS_loss_fn

import argparser
from utils.sketch_utils import *
from utils.shared import args, logger, update_args, update_config
from utils.shared import stroke_config as config

import warnings
from utils.my_utils import *


warnings.filterwarnings("ignore")

clip_loss_fn = None


def unpack_dataloader(datas):
    img, pos, path = datas

    return {
        "img": img.to(device, non_blocking=True),
        "pos": pos.to(device, non_blocking=True),                                           # [bs, 9, nL, 8]
        "path": path
    }


# S--------------------------------------------------
import svgwrite

def tensor2image(tensor, out="my_output_image.jpg"):
    tensor_to_pil = transforms.ToPILImage()
    image = tensor_to_pil(tensor)
    image.save(out)

def tensor_to_svg(tensor, filename='output.svg', margin=0.1):
    # Convert tensor to list of coordinates and find bounds
    all_coords = []
    
    for sketch in tensor:
        # Convert tensor values to coordinates
        coords = [sketch[i].cpu().item() for i in range(len(sketch))]
        all_coords.extend(coords[::2])  # x coordinates (even indices)
        all_coords.extend(coords[1::2])  # y coordinates (odd indices)
    
    # Find actual bounds of the data
    min_coord = min(all_coords)
    max_coord = max(all_coords)
    
    # Calculate dimensions with margin
    data_range = max_coord - min_coord
    margin_size = data_range * margin
    
    viewbox_min = min_coord - margin_size
    viewbox_max = max_coord + margin_size
    viewbox_size = viewbox_max - viewbox_min
    
    # Create SVG with proper scaling
    dwg = svgwrite.Drawing(filename, profile='full')
    
    # Set viewBox to match data bounds
    dwg.viewbox(viewbox_min, viewbox_min, viewbox_size, viewbox_size)
    
    # Optional: Set a reasonable display size (can be overridden by CSS)
    dwg.attribs['width'] = "400"
    dwg.attribs['height'] = "400"
    
    # Create group without hardcoded transforms
    g = dwg.g(transform="rotate(-90 5 5) scale(-1 1) translate(-10 0)")
    
    # Calculate appropriate stroke width relative to data size
    stroke_width = viewbox_size * 0.002  # 0.2% of viewbox size
    
    for sketch in tensor:
        # Convert tensor to coordinates (remove the +3 offset)
        coords = [sketch[i].cpu().item() for i in range(len(sketch))]
        
        # Create cubic Bezier path
        path_data = "M {} {} C {} {} {} {} {} {}".format(
            coords[0], coords[1],  # Move to start point
            coords[2], coords[3],  # First control point
            coords[4], coords[5],  # Second control point
            coords[6], coords[7]   # End point
        )
        
        g.add(dwg.path(
            d=path_data, 
            fill="none", 
            stroke="black",
            stroke_linecap="round", 
            stroke_linejoin="round", 
            stroke_opacity=1.0, 
            stroke_width=stroke_width
        ))
    
    dwg.add(g)
    dwg.save(pretty=True)

def plot_output(svg_output, contour_path, name, save_root):
    fig, axs = plt.subplots(2)
    ax = axs[0]
    image = mpimg.imread(contour_path)
    ax.imshow(image)

    ax = axs[1]
    svg_save_path = save_root + "svg/" + name + '.svg'
    paths.mkdir(svg_save_path)
    tensor_to_svg(svg_output, svg_save_path)
    png_bytes = cairosvg.svg2png(url=svg_save_path, output_width=512, output_height=512)
    img = Image.open(io.BytesIO(png_bytes))
    ax.imshow(img)
    # print(save_root)
    plt.savefig(save_root + name + '.png')
    
def save_outputs(inputs, img_outputs, svg_outputs, root, contour_path):
    if not os.path.exists(root):
        os.makedirs(root)
    # print("len(inputs):", len(inputs['path']))
    for i in range(len(inputs['path'])):
        p = inputs['path'][i]
        base_name = os.path.splitext(os.path.basename(p))[0]
        plot_output(save_root=os.path.join(root, base_name), svg_output=svg_outputs['stroke']['position'][i], contour_path=contour_path)
        # img_save_path = os.path.join(root, base_name, 'img_out.jpg')
        # svg_save_path = os.path.join(root, base_name, 'svg_out.svg')
        # contour_save_path = os.path.join(root, base_name, 'contour_in.png')
        # paths.mkdir(img_save_path)

        # tensor_to_svg(svg_outputs['stroke']['position'][i], svg_save_path)
        # tensor2image(img_outputs['sketch_black'][i], img_save_path)
        # copy(contour_inputs, contour_save_path)


import matplotlib.pyplot as plt
import numpy as np

def plot_loss_report(arrays, names, title):
    plt.figure(figsize=[16,12])

    for array in arrays:
        x = [point[0] for point in array]
        y = [point[1].cpu().item() for point in array]  # Assuming this is needed for PyTorch tensors
        plt.plot(x, y, marker='o')  # Plot the line with 'o' as the marker for points
        for i, value in enumerate(y):
            plt.text(x[i], y[i], f'{value:.2f}', color = 'black', ha = 'center', va = 'bottom')

    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.title(title, fontsize=22)
    plt.legend([names[i] for i in range(len(names))], fontsize=14)
    plt.grid(True)
    plt.savefig(os.path.join('logs', args.dataset, f'{title}.png'))
    plt.close()

# E--------------------------------------------------

def plot_results_gt(inputs, sketches, output_filename='logs/my_data/log.jpg'):
    my_stacked_results = torch.stack([
        inputs["img"], sketches["sketch_black"]
    ], dim=1).flatten(0, 1)

    my_save_plots(my_stacked_results, output_filename)


def test(test_loader, weight_path, phase, max_sketches=100):
    checkpoint = torch.load(weight_path)
    tmp_model = SketchModel()
    tmp_model = tmp_model.to(device)
    tmp_model.load_state_dict(checkpoint)
    tmp_model.eval()
    test_result_folder = os.path.join('logs', args.dataset, phase+"_outputs") + "/"
    if not os.path.exists(test_result_folder):
        os.makedirs(test_result_folder)
    
    num_items = 0
    for idx, data in enumerate(test_loader):
        img = data[0].to(device, non_blocking=True)
        contour_paths = data[1]
        with torch.no_grad():
            output = tmp_model(img)
        base_name = os.path.splitext(os.path.basename(data[1][0]))[0]

        for i_out in range(len(contour_paths)):
            save_path = test_result_folder + base_name + "_" + str(i_out)
            plot_output(save_root=test_result_folder, name=base_name + "_" + str(i_out), svg_output=output['stroke']['position'][i_out], contour_path=contour_paths[i_out])
            
            num_items += 1
            if num_items > max_sketches:
                return

        # img_save_path = os.path.join(test_result_folder, base_name, 'img_.jpg')
        # svg_save_path = os.path.join(test_result_folder, base_name, 'svg_out.svg')
        # contour_save_path = os.path.join(test_result_folder, base_name, 'contour_in.png')
        # paths.mkdir(img_save_path)

        # tensor2image(output['sketch_black'].squeeze(), img_save_path)
        # tensor_to_svg(output['stroke']['position'][0], svg_save_path)
        # copy(contour_inputs, contour_save_path)

    # for idx, datas in enumerate(test_loader):
    #     if idx >= 1:  # test for 20 steps
    #         break
    #     imgs_ = datas[0].to(device, non_blocking=True)
    #     # print(imgs_.shape)
    #     with torch.no_grad():
    #         lbs_output = tmp_model(imgs_)
    #     # print("output", lbs_output['sketch_black'].shape)
    #     imgs = {"img": imgs_, "path": datas[1]}

    # plot_results_gt(imgs, lbs_output, f'logs/{args.dataset}/test_log.jpg')

import shutil

def main():
    args_ = argparser.parse_arguments()

    train_set, test_set, image_shape = get_dataset(args_, test_only=True)

    args_.image_size = image_shape[1]
    args_.image_num_channel = image_shape[0]
    stroke_config = argparser.get_stroke_config(args_)

    update_args(args_)
    update_config(stroke_config)

    global device
    device = args.device

    # test_loader = DataLoader(test_set, shuffle=True, num_workers=8, pin_memory=True, batch_size=64)

    model = SketchModel()
    model = model.to(device)

    if args.load_path is not None:
        checkpoint = torch.load(os.path.join(args.load_path, "model.pt"))
        model.load_state_dict(checkpoint)
        args.start_epoch = torch.load(os.path.join(args.load_path, "optim.pt"))["epoch"]


# S ------------------------------------------------

    weight_root_path = args.test_weight_path
    test_loader = DataLoader(test_set, shuffle=True, num_workers=8, pin_memory=True, batch_size=16)
    train_loader = DataLoader(train_set, shuffle=True, num_workers=8, pin_memory=True, batch_size=16)
    best_weight_path = os.path.join(weight_root_path, 'model_best.pt')
    weight_path = best_weight_path if os.path.exists(best_weight_path) else os.path.join(weight_root_path, 'model.pt')
    test(test_loader, weight_path, phase="test")
    test(train_loader, weight_path, max_sketches=50, phase="train")
# E ------------------------------------------------


if __name__ == "__main__":
    main()

