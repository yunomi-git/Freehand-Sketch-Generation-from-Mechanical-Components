import modal
import sys

sys.path.append("/app")


secret=modal.Secret.from_name("docker-registry")
image = modal.Image.from_registry(
    "yunomi134/onlypy", secret=secret
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
).add_local_file("./third_party/clipasso/models/sam_vit_h_4b8939.pth", 
                 remote_path="/root/third_party/clipasso/models/sam_vit_h_4b8939.pth")
app = modal.App("example-get-started", image=image)

# import modal
# print(modal.__version__)
# print(dir(modal))

@app.function(image=image)
def list_with_tree_structure():
    """Show directory tree structure"""
    import os
    # import cv2
    # print("OpenCV version:", cv2.__version__)
    # print("OpenCV build info:", cv2.getBuildInformation())
    from third_party.clipasso.models.painter_params import Painter, PainterOptimizer
    # print("Painter loaded")

    # painter = Painter(None)

    
    def print_tree(directory, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return
            
        try:
            items = sorted(os.listdir(directory))
            
            for i, item in enumerate(items):
                if item.startswith('.'):  # Skip hidden files/folders
                    continue
                    
                item_path = os.path.join(directory, item)
                is_last = i == len(items) - 1
                
                current_prefix = "└── " if is_last else "├── "
                print(f"{prefix}{current_prefix}{item}")
                
                if os.path.isdir(item_path):
                    extension = "    " if is_last else "│   "
                    print_tree(item_path, prefix + extension, max_depth, current_depth + 1)
                    
        except PermissionError:
            print(f"{prefix}└── [Permission Denied]")
    
    print("Directory Tree Structure:")
    print("=" * 40)
    print(".")
    print_tree("/app")
if __name__ == "__main__":
    # Run the function
    with app.run():
        # result = list_local_folders.remote()
        print("\n" + "=" * 40)
        list_with_tree_structure.remote()