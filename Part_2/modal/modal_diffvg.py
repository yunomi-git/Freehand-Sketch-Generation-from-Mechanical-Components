import modal

# Create a Modal image with GPU support and diffvg dependencies
image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install([
        "git",
        "build-essential",
        "cmake",
        "ninja-build",
        "pkg-config",
        "libgl1-mesa-dev",
        "libegl1-mesa-dev",
        "libgles2-mesa-dev",
        "libx11-dev",
        "libxext-dev",
        "libxrandr-dev",
        "libxcursor-dev",
        "libxinerama-dev",
        "libxi-dev",
        "libxss-dev",
        "libglu1-mesa-dev",
        "freeglut3-dev",
    ])
    .pip_install([
        "torch",
        "torchvision", 
        "numpy",
        "Pillow",
        "scikit-image",
        "pybind11",
    ])
    .run_commands([
        # Set CUDA environment variables
        "export CUDA_HOME=/usr/local/cuda",
        "export PATH=$CUDA_HOME/bin:$PATH",
        "export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH",
        # Clone and install diffvg
        "git clone https://github.com/BachiLi/diffvg.git /tmp/diffvg",
        "cd /tmp/diffvg && git submodule update --init --recursive",
        # Force CUDA compilation by setting environment variables
        "cd /tmp/diffvg && CUDA_HOME=/usr/local/cuda FORCE_CUDA=1 DIFFVG_CUDA=1 TORCH_CUDA_ARCH_LIST='7.0;7.5;8.0;8.6' python setup.py install",
    ])
)

app = modal.App("diffvg-app", image=image)

# Test function to verify diffvg installation
@app.function(gpu="A10G")  # or modal.gpu.T4() for smaller workloads
def test_diffvg():
    import diffvg
    import torch
    
    print(f"diffvg version: {diffvg.__version__ if hasattr(diffvg, '__version__') else 'unknown'}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print('diffvg CUDA kernels:', hasattr(diffvg, 'RenderFunction'))
    # Simple test to ensure diffvg works
    canvas_width, canvas_height = 256, 256
    shapes = []
    shape_groups = []
    
    # Create a simple circle
    circle = diffvg.Circle(radius=torch.tensor(40.0),
                          center=torch.tensor([128.0, 128.0]))
    shapes.append(circle)
    
    path_group = diffvg.ShapeGroup(shape_ids=torch.tensor([0]),
                                  fill_color=torch.tensor([0.3, 0.6, 0.9, 1.0]))
    shape_groups.append(path_group)
    
    # Render
    render = diffvg.RenderFunction.apply
    scene_args = diffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups)
    img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
    
    print(f"Successfully rendered image with shape: {img.shape}")
    return "diffvg installation successful!"

# Example usage function
@app.function(gpu="A10G")
def render_svg_example():
    import diffvg
    import torch
    import numpy as np
    from PIL import Image
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    canvas_width, canvas_height = 512, 512
    shapes = []
    shape_groups = []
    
    # Create a more complex example with multiple shapes
    # Circle
    circle = diffvg.Circle(radius=torch.tensor(50.0),
                          center=torch.tensor([200.0, 200.0]))
    shapes.append(circle)
    
    # Rectangle  
    rect = diffvg.Rect(p_min=torch.tensor([300.0, 150.0]),
                      p_max=torch.tensor([450.0, 250.0]))
    shapes.append(rect)
    
    # Add colors
    circle_group = diffvg.ShapeGroup(shape_ids=torch.tensor([0]),
                                    fill_color=torch.tensor([1.0, 0.2, 0.2, 1.0]))
    rect_group = diffvg.ShapeGroup(shape_ids=torch.tensor([1]),
                                  fill_color=torch.tensor([0.2, 1.0, 0.2, 1.0]))
    
    shape_groups.extend([circle_group, rect_group])
    
    # Render the scene
    render = diffvg.RenderFunction.apply
    scene_args = diffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups)
    
    img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
    
    # Convert to PIL Image and save
    img_np = img.detach().cpu().numpy()
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    
    pil_img = Image.fromarray(img_np)
    
    print(f"Rendered image successfully on {device}")
    return img_np

@app.function(gpu="any")
def check_nvidia_smi():
    import subprocess
    output = subprocess.check_output(["nvidia-smi"], text=True)
    assert "Driver Version:" in output
    assert "CUDA Version:" in output
    print(output)
    return output

# Entrypoint function
@app.local_entrypoint()
def main():
    # Test the installation
    print("Testing diffvg installation...")
    result = test_diffvg.remote()
    print(result)
    
    # # Run example rendering
    # print("\nRunning example rendering...")
    # img_array = render_svg_example.remote()
    # print(f"Rendered image with shape: {img_array.shape}")
    
    # print("All tests completed successfully!")