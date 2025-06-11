import diffvg
import pydiffvg

print('diffvg CUDA kernels:', hasattr(diffvg, 'RenderFunction'))
print('diffvg CUDA kernels:', hasattr(pydiffvg, 'RenderFunction'))


