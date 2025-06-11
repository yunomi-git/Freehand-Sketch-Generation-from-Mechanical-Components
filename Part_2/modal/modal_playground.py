import modal

# Use your existing Dockerfile
# image = modal.Image.from_dockerfile(
#     "Dockerfile",
# )
secret=modal.Secret.from_name("docker-registry")
image = modal.Image.from_registry(
    "yunomi134/test", secret=secret
)#.env({
    #     "PATH": "/opt/miniconda3/envs/part_2/bin:/usr/local/bin:/usr/bin:/bin",
    #     "PYTHONPATH": "/opt/miniconda3/envs/part_2/lib/python3.10/site-packages",
    #     "CONDA_DEFAULT_ENV": "part_2",
    #     "CONDA_PREFIX": "/opt/miniconda3/envs/part_2"
    # })
app = modal.App("example-get-started", image=image)

@app.function(image=image)
def debug_environment():
    import subprocess
    import os
    
    results = {}
    
    # Check which python is being used
    results['which_python'] = subprocess.run(['which', 'python'], capture_output=True, text=True).stdout
    results['python_version'] = subprocess.run(['python', '--version'], capture_output=True, text=True).stdout
    
    # Check environment variables
    results['path'] = os.environ.get('PATH', 'NOT SET')
    results['conda_default_env'] = os.environ.get('CONDA_DEFAULT_ENV', 'NOT SET')
    results['conda_prefix'] = os.environ.get('CONDA_PREFIX', 'NOT SET')
    
    # Check if conda command works
    results['conda_info'] = subprocess.run(['conda', 'info'], capture_output=True, text=True).stdout
    results['conda_envs'] = subprocess.run(['conda', 'env', 'list'], capture_output=True, text=True).stdout
    
    # Check what's in the conda environment
    results['env_packages'] = subprocess.run(['/opt/miniconda3/envs/part_2/bin/pip', 'list'], capture_output=True, text=True).stdout
    
    # Try to import torch with explicit python path
    torch_test = subprocess.run(['/opt/miniconda3/envs/part_2/bin/python', '-c', 'import torch; print(torch.__version__)'], capture_output=True, text=True)
    results['torch_with_env_python'] = {
        'stdout': torch_test.stdout,
        'stderr': torch_test.stderr,
        'returncode': torch_test.returncode
    }
    print(results)
    return results

@app.function(image=image)
def square(x):
    import subprocess
    subprocess.run(["conda", "activate", "part_2"])
    import numpy as np
    # print("This code is running on a remote worker!")
    # return x * x
    object = np.zeros((x, x))
    return object.shape()
    # return x**2

@app.function(image=image)
def debug_python_path():
    import sys
    import subprocess
    
    results = {}
    
    # Check which Python we're actually using
    results['sys_executable'] = sys.executable
    results['sys_path'] = sys.path
    
    # Check if numpy is in the environment
    which_python = subprocess.run(['which', 'python'], capture_output=True, text=True)
    results['which_python'] = which_python.stdout
    
    # Try to find numpy specifically
    try:
        import numpy
        results['numpy_success'] = f"Found numpy {numpy.__version__} at {numpy.__file__}"
    except ImportError as e:
        results['numpy_error'] = str(e)
        
        # Check what packages are available
        pip_list = subprocess.run([sys.executable, '-m', 'pip', 'list'], capture_output=True, text=True)
        results['available_packages'] = pip_list.stdout
    
    print(results)


@app.local_entrypoint()
def main():
    square.remote(4)
    # print("the square is", square.remote(42))


if __name__=="__main__":
    main()
