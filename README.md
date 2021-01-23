# Transform quantization for CNN compression (arXiv:2009.01174)
Source code for "transform quantization for CNN compression". Works out of the box for image classification CNNs on ImageNet. 
Slight modifications are necessary to quantize other types of networks. Commands below are for `resnet18py`.

1. Construct Karhunen–Loève transforms (KLTs) for convolution/dense layers.\
`CUDA_VISIBLE_DEVICES=X python generate_KL_basis.py inter klt resnet18py 10000`\
Here, `10000` is the number of ILSVRC2012 validation images for non-KLT transforms. Just keep the `10000` in there.

2. Construct rate–distortion curves for all convolution/dense layers.\
`CUDA_VISIBLE_DEVICES=X python generate_RD_curves_base.py inter klt resnet18py 100`\
`CUDA_VISIBLE_DEVICES=X python generate_RD_curves_kern.py inter klt resnet18py 100`\
These are separate calls to construct RD curves for transform basis and kernels. Here, `100` is the number of ILSVRC2012 validation images to use.


3. Compute the rate–distortion frontier for `your_network_name`.\
`CUDA_VISIBLE_DEVICES=X python generate_RD_frontier.py inter klt resnet18py 100 1 1 0`\
Here, `100 1 1 0` means use `100` ILSVRC2012 validation images, quantize basis `1` and kernels `1`, but not activations `0`.

4. Modifying the code to work with `your_own_cnn` instead of `resnet18py`.\
Essentially, you'll need to have a notion of MSE for your network output. First, make sure dense layers in `your_own_cnn` are represented as 
`1x1-conv` layers. Second, modify `common.py:loadnetwork` to load `your_own_cnn` and  `common.py:predict` to appropriately return the output 
of your CNN.

