import modal
import argparse

# Use your existing Dockerfile
# image = modal.Image.from_dockerfile(
#     "Dockerfile",
# )
secret=modal.Secret.from_name("docker-registry")
image = modal.Image.from_registry(
    "yunomi134/onlypy", secret=secret, force_build=False
).run_commands(
    "git clone https://github.com/BachiLi/diffvg.git",
    "cd diffvg && git submodule update --init --recursive",
    "cd diffvg && DIFFVG_CUDA=1 python setup.py install",
)
app = modal.App("example-get-started", image=image)
@app.function(image=image)
def square(x):
    import numpy as np
    # print("This code is running on a remote worker!")
    # return x * x
    object = np.zeros((x, x))
    return object.shape
    # return x**2

@app.function(image=image, gpu="H100")
def checktorch():
    import torch
    import diffvg
    print('diffvg CUDA kernels:', hasattr(diffvg, 'RenderFunction'))
    # print("This code is running on a remote worker!")
    # return x * x
    print(torch.cuda.is_available())

    import diffvg
    print('diffvg CUDA kernels available:', hasattr(diffvg, 'RenderFunction'))
    # return x**2

@app.local_entrypoint()
def main(*arglist):
    parser = argparse.ArgumentParser()
    # general arguments
    parser.add_argument('--output_dir', type=str, default='./output')
    args = parser.parse_args(args=arglist)
    checktorch.remote()
    # print("the square is", square.remote(42))


if __name__=="__main__":
    main()
