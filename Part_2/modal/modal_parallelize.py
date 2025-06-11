import modal

# Use your existing Dockerfile
# image = modal.Image.from_dockerfile(
#     "Dockerfile",
# )
secret=modal.Secret.from_name("docker-registry")
image = modal.Image.from_registry(
    "yunomi134/onlypy", secret=secret
)
app = modal.App("example-get-started", image=image)

@app.function(image=image, gpu="H100")
def checktorch(x):
    import torch
    # print("This code is running on a remote worker!")
    # return x * x
    return torch.Tensor([x*x])
    # return x**2

@app.local_entrypoint()
def main():
    import numpy as np
    inputs = list(np.arange(1000))
    # results = list(checktorch.map(inputs))

    a = modal_parallelizer()
    results = list(a.checktorch.map(inputs))

    print(results)
    # print("the square is", square.remote(42))


class modal_parallelizer:
    def __init__(self):
        self.value = 18

    @app.function(image=image, gpu="H100")
    def checktorch(self, x):
        import torch
        # print("This code is running on a remote worker!")
        # return x * x
        return torch.Tensor([x + self.value])

if __name__=="__main__":
    main()
