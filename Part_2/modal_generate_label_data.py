import modal
import sys
sys.path.append("/app")


secret=modal.Secret.from_name("docker-registry")
image = modal.Image.from_registry(
    "yunomi134/onlypy", secret=secret, force_build=False
).apt_install(
    [
        "git",
        "libglfw3-dev",
        "libgles2-mesa-dev",
        "libglib2.0-dev",
        "libgl1-mesa-glx",  # Add this line to install libGL.so.1
        "xorg",
        "libxkbcommon0",
    ]
).run_commands([
        # Set CUDA environment variables
        "export CUDA_HOME=/usr/local/cuda",
        "export PATH=$CUDA_HOME/bin:$PATH",
        "export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH",
        # Clone and install diffvg
        "git clone https://github.com/BachiLi/diffvg.git /tmp/diffvg",
        "cd /tmp/diffvg && git submodule update --init --recursive",
        # Force CUDA compilation by setting environment variables
        "cd /tmp/diffvg && CUDA_HOME=/usr/local/cuda FORCE_CUDA=1 DIFFVG_CUDA=1 TORCH_CUDA_ARCH_LIST='7.0;7.5;8.0;8.6' python setup.py install",
    ]
).add_local_file("./third_party/clipasso/models/sam_vit_h_4b8939.pth", 
                 remote_path="/root/third_party/clipasso/models/sam_vit_h_4b8939.pth"
).add_local_file("./third_party/U2Net_/saved_models/u2net.pth", 
                 remote_path="/root/third_party/U2Net_/saved_models/u2net.pth"
)

app = modal.App("gen-guidance", image=image)
import paths
import itertools
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from pathlib import Path

from third_party.clipasso.models.painter_params import Painter, PainterOptimizer
from third_party.clipasso.models.loss import Loss
from third_party.clipasso import sketch_utils as utils
from argparser import parse_arguments
import argparse
from utils.config import Config

import numpy as np
from PIL import Image

import os
import pickle
import time
import random
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import subprocess as sp
import svgwrite
from xml.dom import minidom



abs_path = os.path.abspath(os.getcwd())
if not os.path.isfile(f"{abs_path}/third_party/U2Net_/saved_models/u2net.pth"):
    print("not found", f"{abs_path}/third_party/U2Net_/saved_models/u2net.pth")
    sp.run(["gdown", "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ",
            "-O", "third_party/U2Net_/saved_models/"])

# Builds a dataset with the specified image file-paths.
class ImageDataset(Dataset):
    def __init__(self, path_formats, transform=None):
        self.path_formats = path_formats
        self.transform = transform

        self.image_paths = []
        self.images = []
        for path_format in self.path_formats:
            image_path = glob(path_format, recursive='**' in path_format)
            self.image_paths += image_path
        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # dummy label
        return [image, image_path], -1


# Prefixes the index and drops the label for each sample.
class IndexedDataset(Dataset):
    def __init__(self, dataset, start_index=0):
        self.dataset = dataset
        self.start_index = start_index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Return the global index
        return self.start_index + index, self.dataset[index][0]


# The main class that generates sketch data with the specified image dataset.
class DataGenerator(nn.Module):
    def __init__(self, args):
        super(DataGenerator, self).__init__()
        
        self.args = args
        
        # Initialize the renderers.
        renderers = []
        renderers.append(Painter(
            args,
            args.num_strokes, args.num_segments,
            imsize=args.image_scale,
            device=args.device,
        ))
        for _ in range(args.batch_size - 1):
            renderers.append(Painter(
                args,
                args.num_strokes, args.num_segments,
                imsize=args.image_scale,
                device=args.device,
                clip_model=renderers[0].clip_model,
                clip_preprocess=renderers[0].clip_preprocess,
                dino_model=renderers[0].dino_model
            ))

        self.renderers = nn.ModuleList(renderers)
        self.criterion = Loss(args)
        self.u2net = utils.get_u2net(args)

    def save_sample_visualization(self, sample_name, sample_image, sample_paths):
        fig, axs = plt.subplots(1, len(sample_paths)+1, figsize=(3+3*len(sample_paths), 3))

        axs[0].set_title('image')
        axs[0].imshow(sample_image)

        t = np.linspace(0, 1, 10)
        for i, (step, step_paths) in enumerate(sample_paths.items()):
            curves = cubic_bezier(step_paths['pos'], t)
            axs[i+1].set_title(f'step {step}')
            for curve, color, width in zip(curves, step_paths['color'], step_paths['radius']):
                axs[i+1].plot(*curve.T[::-1], c=color)
            axs[i+1].set_ylim(1,-1)
            axs[i+1].tick_params(axis='both', which='major', labelsize=5)

        vis_filename = os.path.join(self.args.output_dir, f'vis/{sample_name}.jpg')
        fig.savefig(vis_filename)
        plt.close()

    def save_svgs(self, sample_paths, image_path):
        image_path_list = os.path.normpath(image_path).split(os.path.sep)[1:]
        specific_path = os.path.splitext(os.path.sep.join(image_path_list))[0]
        image_path = os.path.join(self.args.output_dir, 'svgs', specific_path)
        os.makedirs(image_path, exist_ok=True)

        for i, (step, step_paths) in enumerate(sample_paths.items()):
            strokes_ = []
            for stroke in step_paths['pos']:
                x1, y1, x2, y2, x3, y3, x4, y4 = stroke
                # stroke_ = [(x1 + 2, y1 + 2), (x2 + 2, y2 + 2), (x3 + 2, y3 + 2), (x4 + 2, y4 + 2)]
                # stroke_ = [(y1 + 2, x1 + 2), (y2 + 2, x2 + 2), (y3 + 2, x3 + 2), (y4 + 2, x4 + 2)]
                stroke_ = [y1 + 2, x1 + 2, y2 + 2, x2 + 2, y3 + 2, x3 + 2, y4 + 2, x4 + 2]
                strokes_.append(stroke_)

            # -------------------------------------------------

            svg_out_file_path = os.path.join(image_path, f'{step}.svg')
            dwg = svgwrite.Drawing(svg_out_file_path, profile='full', size=(200, 200), viewBox="0 0 4 4", version="1.2")
            g = dwg.add(dwg.g())
            for stroke in strokes_:
                d = f'M {stroke[0]} {stroke[1]} C {stroke[2]} {stroke[3]} {stroke[4]} {stroke[5]} {stroke[6]} {stroke[7]}'
                g.add(dwg.path(d=d, stroke_width="0.05", fill="none", stroke="rgb(0, 0, 0)", stroke_opacity="1.0", stroke_linecap="round", stroke_linejoin="round"))
            dom = minidom.parseString(dwg.tostring())
            with open(svg_out_file_path, 'w', encoding='utf-8') as f:
                f.write(dom.toprettyxml(indent="  "))

            # -------------------------------------------------

            # svg_header = '''<?xml version="1.0" ?>
            # <svg xmlns="http://www.w3.org/2000/svg" version="1.2" baseProfile="full" viewBox="0 0 4 4" width="200" height="200">
            # <defs />
            # <g>
            # '''

            # svg_footer = '''  </g>
            # </svg>'''

    def _generate(self, args, image, image_path, mask, num_iter, num_strokes, width, attn_colors, path_dicts=None, gradual_colors=True, use_tqdm=False):
        curr_batch_size = image.size(0)
        if path_dicts is None:
            path_dicts = [None] * curr_batch_size
        renderers = self.renderers[:curr_batch_size]

        for renderer, curr_image, curr_image_path, curr_mask, path_dict in zip(renderers, image, image_path, mask, path_dicts):
            renderer.set_random_noise(0)
            try:
                renderer.init_image(
                    target_im=curr_image.unsqueeze(0),
                    mask=curr_mask.unsqueeze(0) if curr_mask else None, # unused
                    stage=0,
                    randomize_colors=False,
                    attn_colors=attn_colors,
                    attn_colors_stroke_sigma=5.0,
                    path_dict=path_dict,
                    new_num_strokes=num_strokes,
                    new_width=width,
                    init_settings=(args.init_mode, args.sam_init_num),
                    log_path=args.output_dir
                )
            except Exception as err:
                print(err, curr_image_path)
                assert(False)

        if num_iter == 0:
            for renderer in renderers:
                for key_step in self.args.key_steps:
                    renderer.log_shapes(str(key_step))
            path_dicts = [renderer.path_dict_np(radius=width) for renderer in renderers]
            if gradual_colors:
                for sample_paths in path_dicts:
                    ts = np.linspace(0, 1, len(sample_paths))
                    for t, step_paths in zip(ts, sample_paths.values()):
                        step_paths['color'] *= t
            return path_dicts

        optimizer = PainterOptimizer(self.args, renderers)
        optimizer.init_optimizers()

        steps = range(num_iter)
        if use_tqdm:
            steps = tqdm(steps)

        for step in steps:
            for renderer in renderers:
                renderer.set_random_noise(step)

            optimizer.zero_grad_()
            sketches = torch.cat([renderer.get_image().to(self.args.device) for renderer in renderers], dim=0)
            loss = sum(self.criterion(sketches, image.detach(), step, points_optim=optimizer).values()).mean()
            loss.backward()
            optimizer.step_(optimize_points=True, optimize_colors=False)

            if (step+1) in self.args.key_steps:
                for renderer in renderers:
                    renderer.log_shapes()
                    renderer.log_shapes(str(step+1))

        return [renderer.path_dict_np(radius=width) for renderer in renderers]

    def generate_for_batch(self, args, index, image_info, use_tqdm=False):
        image, image_path = image_info
        sample_names = [f'{idx}_{self.args.seed}' for idx in index.tolist()]

        # NOTE: mainly generate mask stuffs using u2net here, and our 'foreground' is set to be 'image' by default
        if args.option_mask_image:
            foreground, background, mask, _ = utils.get_mask_u2net_batch(self.args, image, net=self.u2net, return_background=True)
            with torch.no_grad():
                mask_areas = mask.view(mask.size(0), -1).mean(dim=1).tolist()
                mask_areas = dict(zip(sample_names, mask_areas))
        else:
            foreground = image # not mask image by default
            mask = len(foreground) * [None]
            mask_areas = {}

        num_strokes_fg = self.args.num_strokes - self.args.num_background
        num_strokes_bg = self.args.num_background
        stroke_width_fg = self.args.width
        stroke_width_bg = self.args.width_bg

        # NOTE: generate label data in vector graphics format here, mask is unused
        path_dicts = self._generate(args, foreground, image_path, mask, self.args.num_iter, num_strokes_fg, stroke_width_fg, False, use_tqdm=use_tqdm)
        new_path_dicts = []
        for item1, item2 in zip(path_dicts, image_path):
            item2 = str(Path(*Path(item2).parts[-3:]))
            img_root = str(Path(*Path(item2).parts[:-3]))
            new_item = {'iterations': item1, 'img_path': item2, 'img_root': img_root}
            new_path_dicts.append(new_item)
        path_dicts = new_path_dicts

        # NOTE: we return here, mask_areas is unused
        if not self.args.enable_color:
            path_dicts = dict(zip(sample_names, path_dicts))
            return path_dicts, mask_areas

        color_dicts = self._generate(args, foreground, image_path, mask, 0, None, stroke_width_fg, True, path_dicts=path_dicts, use_tqdm=use_tqdm)
        for paths, colors in zip(path_dicts, color_dicts):
            for step in self.args.key_steps:
                step = str(step)
                paths[step]['color'] = colors[step]['color']

        if num_strokes_bg <= 0:
            path_dicts = dict(zip(sample_names, path_dicts))
            return path_dicts, mask_areas
        
        path_dicts_bg = self._generate(args, background, image_path, 1 - mask, 0, num_strokes_bg, stroke_width_bg, True, use_tqdm=use_tqdm)
        for paths, paths_bg in zip(path_dicts, path_dicts_bg):
            for step in self.args.key_steps:
                step = str(step)
                paths[step]['pos'] = np.concatenate([paths[step]['pos'], paths_bg[step]['pos']], axis=0)
                paths[step]['color'] = np.concatenate([paths[step]['color'], paths_bg[step]['color']], axis=0)
                if 'radius' in paths[step]:
                    paths[step]['radius'] = np.concatenate([paths[step]['radius'], paths_bg[step]['radius']], axis=0)

        path_dicts = dict(zip(sample_names, path_dicts))
        return path_dicts, mask_areas
    
    
    def generate_for_dataset(self, args, dataloader, use_tqdm=False, track_time=False):
        path_dicts = {}
        mask_areas = {}
        svgs = []

        if track_time:
            start_time = time.time()

        min_index = next(iter(dataloader.dataset.indices))
        max_index = next(iter(reversed(dataloader.dataset.indices)))

        generated_samples = 0
        total_samples = len(dataloader.dataset)
        
        for index, image_info in dataloader:
            image, image_path = image_info
            if track_time:
                print(f'generating samples for {index.min().item()}..{index.max().item()} of {min_index}..{max_index}:')
            image = image.to(self.args.device)
            
            batch_path_dicts, batch_mask_areas = self.generate_for_batch(args, index, (image, image_path), use_tqdm=use_tqdm)
            path_dicts.update(batch_path_dicts)
            # mask_areas.update(batch_mask_areas)

            # if self.args.visualize:
            #     for i, (sample_name, sample_paths) in enumerate(batch_path_dicts.items()):
            #         sample_image = image[i].detach().cpu().permute(1, 2, 0).numpy()
            #         sample_paths_to_show = sample_paths['iterations']
            #         vis_file_name = sample_name + '==' + os.path.splitext(os.path.basename(sample_paths['img_path']))[0]
            #         self.save_sample_visualization(vis_file_name, sample_image, sample_paths_to_show)

            # if self.args.save_svgs:
            for i, (sample_name, sample_paths) in enumerate(batch_path_dicts.items()):
                svgs.append({
                    "sample_paths": sample_paths['iterations'],
                    "image_path": sample_paths['img_path']
                })
                # self.save_svgs(sample_paths['iterations'], sample_paths['img_path'])

            if track_time:
                generated_samples += image.size(0)
                completion = generated_samples / total_samples
                time_passed = time.time() - start_time
                time_left = time_passed / completion - time_passed
                tp_days, tp_hours, tp_minutes, _ = to_dhms(time_passed)
                tl_days, tl_hours, tl_minutes, _ = to_dhms(time_left)
                print(
                    f'{completion*100:.02f}% ({generated_samples}/{total_samples}) complete.'\
                    f' {tp_days}d {tp_hours}h {tp_minutes}m passed.'\
                    f' expected {tl_days}d {tl_hours}h {tl_minutes}m left.'
                )

        if track_time:
            print(f'took {time_passed:.02f}s to generate {total_samples} samples.')

        return path_dicts, svgs

        # return path_dicts, mask_areas, svgs




def cubic_bezier(p, t):
    p = p.reshape(-1, 4, 1, 2)
    t = t.reshape(1, -1, 1)
    return ((1-t)**3)*p[:,0] + 3*((1-t)**2)*t*p[:,1] + 3*(1-t)*(t**2)*p[:,2] + (t**3)*p[:,3]

def to_dhms(seconds):
    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return days, hours, minutes, seconds


# def split_indexed_dataset(dataset, split_size):
    

def get_dataloader(args):
    assert (args.img_paths is not None) ^ (args.dataset is not None and args.data_root is not None),\
        "either \'img_paths\' or \'dataset\' and \'data_root\' must be specified!"

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    dataset = ImageDataset(args.img_paths, transform=transform)

    num_generation = len(dataset) if args.num_generation == -1 else args.num_generation
    print("num in dataset: ", num_generation)
    dataset = IndexedDataset(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    return dataloader
    # dataloaders = []
    # total_data = 0
    # # datasets = []
    # # for i in range(0, len(dataset.dataset), split_size):
    # #     subset_data = dataset.dataset[i:i+split_size]
    # #     datasets.append(IndexedDataset(subset_data, start_index=i))
    # # return datasets
    # chunk_size = int(np.ceil(num_generation / num_splits))
    # for i in range(0, len(dataset), chunk_size):

    # #     subset_data = dataset.dataset[i:i+chunk_size]
    # #     datasets.append(IndexedDataset(subset_data, start_index=i))

    # #     dataset = Subset(dataset, range(chunk_start, chunk_end))
    # #     total_data += len(dataset)
    # #     # print("dataset length", len(dataset))
    # #     dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    # #     dataloaders.append(dataloader)
    # chunk_size = int(np.ceil(num_generation / num_splits))
    # for i in range(0, len(dataset), chunk_size):
    #     subset_data = dataset[i:i+chunk_size]
    #     subset_data = IndexedDataset(subset_data, start_index=i)
    #     total_data += len(subset_data)
    #     dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    #     dataloaders.append(dataloader)
    # print("incoming data:", num_generation, "chunked:", total_data)
    # return dataloaders

def save_svgs(args, sample_paths, image_path):
        image_path_list = os.path.normpath(image_path).split(os.path.sep)[1:]
        specific_path = os.path.splitext(os.path.sep.join(image_path_list))[0]
        image_path = os.path.join(args.output_dir, 'svgs', specific_path)
        os.makedirs(image_path, exist_ok=True)

        for i, (step, step_paths) in enumerate(sample_paths.items()):
            strokes_ = []
            for stroke in step_paths['pos']:
                x1, y1, x2, y2, x3, y3, x4, y4 = stroke
                # stroke_ = [(x1 + 2, y1 + 2), (x2 + 2, y2 + 2), (x3 + 2, y3 + 2), (x4 + 2, y4 + 2)]
                # stroke_ = [(y1 + 2, x1 + 2), (y2 + 2, x2 + 2), (y3 + 2, x3 + 2), (y4 + 2, x4 + 2)]
                stroke_ = [y1 + 2, x1 + 2, y2 + 2, x2 + 2, y3 + 2, x3 + 2, y4 + 2, x4 + 2]
                strokes_.append(stroke_)

            # -------------------------------------------------

            svg_out_file_path = os.path.join(image_path, f'{step}.svg')
            dwg = svgwrite.Drawing(svg_out_file_path, profile='full', size=(200, 200), viewBox="0 0 4 4", version="1.2")
            g = dwg.add(dwg.g())
            for stroke in strokes_:
                d = f'M {stroke[0]} {stroke[1]} C {stroke[2]} {stroke[3]} {stroke[4]} {stroke[5]} {stroke[6]} {stroke[7]}'
                g.add(dwg.path(d=d, stroke_width="0.05", fill="none", stroke="rgb(0, 0, 0)", stroke_opacity="1.0", stroke_linecap="round", stroke_linejoin="round"))
            dom = minidom.parseString(dwg.tostring())
            with open(svg_out_file_path, 'w', encoding='utf-8') as f:
                f.write(dom.toprettyxml(indent="  "))

@app.function(image=image, gpu="A100",
              memory=8192,
              cpu=4.0,
              max_containers=1,
              timeout=30*60
              )
def generate_guidance_sketches(map_input):
    # import torch
    # import pydiffvg
    # print("torch avail", torch.cuda.is_available())
    # print("diffvg render function", hasattr(pydiffvg, 'RenderFunction'))

    batch = map_input[0]
    args = Config.from_dict(map_input[1])
    data_generator = DataGenerator(args).to(args.device)

    svgs = []
    index, info = batch
    image, image_path = info
    print(f'generating samples for {index.min().item()}..{index.max().item()}:')
    image = image.to(args.device)

    print(index)

    batch_path_dicts, batch_mask_areas = data_generator.generate_for_batch(args, index, (image, image_path), use_tqdm=True)
    # print("path dict", batch_path_dicts)

    for i, (sample_name, sample_paths) in enumerate(batch_path_dicts.items()):
        svgs.append({
            "sample_paths": sample_paths['iterations'],
            "image_path": sample_paths['img_path']
        })  

    return batch_path_dicts, svgs, int(index[0])

@app.local_entrypoint()
def main(*arglist):

    parser = argparse.ArgumentParser()

    # general arguments
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--dataset', type=str, help='clevr_train, clevr_val, stl10_train+unlabeled, stl10_test, etc..')
    parser.add_argument('--data_root', type=str, help='the path to the root directory containing datasets to process.')
    parser.add_argument('--img_paths', type=str, nargs='+', help='image file-paths (with wildcards) to process.')
    parser.add_argument('--key_steps', type=int, nargs='+',
                        default=[0, 50, 100, 200, 400, 700, 1000, 1500, 2000])
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--num_generation', type=int, default=-1, help='number of sketches to generate, generate all sketches in the dataset if -1')
    parser.add_argument('--chunk', type=int, nargs=2, help='--chunk (num_chunks) (chunk_index)')
    parser.add_argument('--init_mode', type=str, default='lbs', help='lbs: LBS first; sam: SAM first')
    parser.add_argument('--sam_init_num', type=int, default=3, help='the number of initial points for each contour if using SAM to initialize stroke points')
    parser.add_argument('--option_mask_image', type=int, default=0, help='get mask of image or not')

    # optimization arguments
    parser.add_argument('--width', type=float, default=1.5, help='foreground-stroke width')
    parser.add_argument('--width_bg', type=float, default=8.0, help='background-stroke width')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1.0)

    # extra arguments
    parser.add_argument('--no_tqdm', action='store_true')
    parser.add_argument('--no_track_time', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--save_svgs', action='store_true')

    # CLIPasso arguments
    parser.add_argument("--target", type=str)
    parser.add_argument("--path_svg", type=str, default="none")
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--num_iter", type=int, default=500)
    parser.add_argument("--num_stages", type=int, default=1)
    parser.add_argument("--color_vars_threshold", type=float, default=0.0)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--control_points_per_seg", type=int, default=4)
    parser.add_argument("--attention_init", type=int, default=1)
    parser.add_argument("--saliency_model", type=str, default="clip")
    parser.add_argument("--saliency_clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--xdog_intersec", type=int, default=0)
    parser.add_argument("--mask_object_attention", type=int, default=0)
    parser.add_argument("--softmax_temp", type=float, default=0.3)
    parser.add_argument("--percep_loss", type=str, default="none")
    parser.add_argument("--train_with_clip", type=int, default=0)
    parser.add_argument("--clip_weight", type=float, default=0)
    parser.add_argument("--start_clip", type=int, default=0)
    parser.add_argument("--num_aug_clip", type=int, default=3) # LBS: 4
    parser.add_argument("--include_target_in_aug", type=int, default=0) # unused
    parser.add_argument("--augment_both", type=int, default=1) # unused
    parser.add_argument("--augemntations", type=str, default="affine")
    parser.add_argument("--noise_thresh", type=float, default=0.5)
    parser.add_argument("--force_sparse", type=float, default=1)
    parser.add_argument("--clip_conv_loss", type=float, default=1)
    parser.add_argument("--clip_conv_loss_type", type=str, default="L2")
    parser.add_argument("--clip_model_name", type=str, default="RN101")
    parser.add_argument("--clip_fc_loss_weight", type=float, default=0.1)
    parser.add_argument("--clip_text_guide", type=float, default=0)
    parser.add_argument("--text_target", type=str, default="none")

    args1 = vars(parser.parse_known_args(args=arglist)[0])

    # if args is None:
    args = parse_arguments(argslist=arglist)
    args.update(args1)
    args.num_iter = max(args.key_steps)
    args.use_gpu = not args.no_cuda
    args.image_scale = args.image_size
    args.color_lr = 0.01

    # args.config_path = "config/my_data.yaml"
    # args.output_dir = "/root/Projects/CachedDatasets/Freehand/logs/my_data/"
    # args.data_root = "/root/Projects/CachedDatasets/Freehand/"
    # args.img_paths = "/root/Projects/CachedDatasets/Freehand/vsm/*/*/*.png"
    # args.visualize = False
    # args.save_svgs = True 
    # args.device = "gpu" 
    # args.num_strokes = 30

    # NUM_SAMPLES = 1000

    # desired_num_splits = 10

    # print("Loading dataloaders")
    # dataloaders = get_dataloaders(args, num_splits=desired_num_splits)
    # num_splits = len(dataloaders)

    # for dataloader in dataloaders:
    #     idx, _ = next(iter(dataloader))
    #     print(idx)
    #     dataset = dataloader.dataset
    #     print(len(dataset))
    #     # ind, _ = dataset.dataset.dataset[0]
    #     # print(ind)
    dataloader = get_dataloader(args)
    # print("There are", num_splits, "chunks / splits")
    print('--- batch_size:', args.batch_size)
    # return
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'vis'), exist_ok=True)


    path_dicts = {}
    mask_areas = {}

    # map_inputs = [{
    #     "dataloader": dataloaders[i],
    #     "args_dict": Config.extraction_dictionary(args),
    #     "idx": i
    # } for i in range(num_splits)]

    # for chunk_i in range(num_splits):
    #     dataloader = dataloaders[chunk_i]
    args_dict = Config.extraction_dictionary(args)
    # print("Chunk ", chunk_i , "/", num_splits)
    print("There are", len(dataloader), "batches in this dataloader")
    print("Rendering batches")

    for path_dicts, batch_svgs, batch_i in generate_guidance_sketches.map(list(zip(list(dataloader), itertools.repeat(args_dict)))):
        chunk_info = str(batch_i)
        paths.mkdir(args.output_dir)
        with open(os.path.join(args.output_dir, f'path_{chunk_info}.pkl'), 'wb') as file:
            pickle.dump(path_dicts, file)

        # if args.option_mask_image and mask_areas:
        #     with open(os.path.join(args.output_dir, f'maskareas_seed{args.seed}{chunk_info}.pkl'), 'wb') as file:
        #         pickle.dump(mask_areas, file)

        # path_dicts.update(path_dicts)
        for svg in batch_svgs:
            save_svgs(args, svg["sample_paths"], svg["image_path"])
            # generated_samples += num_generated


    # chunk_info = ''
    # with open(os.path.join(args.output_dir, f'path{chunk_info}.pkl'), 'wb') as file:
    #     pickle.dump(path_dicts, file)

    # if args.option_mask_image and mask_areas:
    #     with open(os.path.join(args.output_dir, f'maskareas_seed{args.seed}{chunk_info}.pkl'), 'wb') as file:
    #         pickle.dump(mask_areas, file)

if __name__=="__main__":
    main()

# modal run -m modal_generate_label_data --config_path config/my_data.yaml --output_dir ~/Projects/CachedDatasets/Freehand/logs/my_data/ --data_root ~/Projects/CachedDatasets/Freehand/ --img_paths ~/Projects/CachedDatasets/Freehand/test_outputs_vsm/\*/\*/\*.png --visualize --save_svgs --device gpu --num_strokes 30