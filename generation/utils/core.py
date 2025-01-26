'''
https://github.com/thstkdgus35/bicubic_pytorch/blob/master/core.py

A standalone PyTorch implementation for fast and efficient bicubic resampling.
The resulting values are the same to MATLAB function imresize('bicubic').
## Author:      Sanghyun Son
## Email:       sonsang35@gmail.com (primary), thstkdgus35@snu.ac.kr (secondary)
## Version:     1.1.0
## Last update: July 9th, 2020 (KST)
Depencency: torch
Example::
>>> import torch
>>> import core
>>> x = torch.arange(16).float().view(1, 1, 4, 4)
>>> y = core.imresize(x, sides=(3, 3))
>>> print(y)
tensor([[[[ 0.7506,  2.1004,  3.4503],
          [ 6.1505,  7.5000,  8.8499],
          [11.5497, 12.8996, 14.2494]]]])
'''

import math
import typing

import torch
from torch.nn import functional as F

__all__ = ['imresize']

K = typing.TypeVar('K', str, torch.Tensor)

def cubic_contribution(x: torch.Tensor, a: float=-0.5) -> torch.Tensor:
    '''
    This function calculates the weights for bicubic interpolation.

    The weights are calculated as a piecewise cubic polynomial as follows:

    1. For the range [0, 1], the weights are calculated as
        (a + 2) * x^3 - (a + 3) * x^2 + 1
    2. For the range (1, 2], the weights are calculated as
        (a * x^3) - (5 * a * x^2) + (8 * a * x) - (4 * a)

    The weights are then normalized to sum up to 1 across the support of the
    kernel.

    The default value of a is -0.5, which is the same as in the MATLAB
    imresize() function.
    '''
    ax = x.abs()
    ax2 = ax * ax
    ax3 = ax * ax2

    range_01 = (ax <= 1)
    range_12 = (ax > 1) * (ax <= 2)

    # For the range [0, 1]
    cont_01 = (a + 2) * ax3 - (a + 3) * ax2 + 1
    cont_01 = cont_01 * range_01.to(dtype=x.dtype)

    # For the range (1, 2]
    cont_12 = (a * ax3) - (5 * a * ax2) + (8 * a * ax) - (4 * a)
    cont_12 = cont_12 * range_12.to(dtype=x.dtype)

    cont = cont_01 + cont_12
    cont = cont / cont.sum(dim=0, keepdim=True)
    return cont

def gaussian_contribution(x: torch.Tensor, sigma: float=2.0) -> torch.Tensor:
    # Calculate the range of points within 3 * sigma of the center of the
    # kernel. This is the range of points that will be used to calculate the
    # weights.
    range_3sigma = (x.abs() <= 3 * sigma + 1)

    # Calculate the Gaussian weights.
    # The weights are calculated as exp(-x^2 / (2 * sigma^2)).
    # The normalization will be done after.
    cont = torch.exp(-x.pow(2) / (2 * sigma**2))

    # Select the weights for the points within the range.
    cont = cont * range_3sigma.to(dtype=x.dtype)

    return cont

def discrete_kernel(
        kernel: str, scale: float, antialiasing: bool=True) -> torch.Tensor:

    '''
    For downsampling with integer scale only.
    '''
    downsampling_factor = int(1 / scale)
    if kernel == 'cubic':
        kernel_size_orig = 4
    else:
        raise ValueError('Pass!')

    if antialiasing:
        kernel_size = kernel_size_orig * downsampling_factor
    else:
        kernel_size = kernel_size_orig

    if downsampling_factor % 2 == 0:
        a = kernel_size_orig * (0.5 - 1 / (2 * kernel_size))
    else:
        kernel_size -= 1
        a = kernel_size_orig * (0.5 - 1 / (kernel_size + 1))

    with torch.no_grad():
        r = torch.linspace(-a, a, steps=kernel_size)
        k = cubic_contribution(r).view(-1, 1)
        k = torch.matmul(k, k.t())
        k /= k.sum()

    return k

def reflect_padding(
        x: torch.Tensor,
        dim: int,
        pad_pre: int,
        pad_post: int) -> torch.Tensor:

    '''
    Apply reflect padding to the given Tensor.
    Note that it is slightly different from the PyTorch functional.pad,
    where boundary elements are used only once.
    Instead, we follow the MATLAB implementation
    which uses boundary elements twice.
    For example,
    [a, b, c, d] would become [b, a, b, c, d, c] with the PyTorch implementation,
    while our implementation yields [a, a, b, c, d, d].
    '''

    # Get the shape of the input tensor
    b, c, h, w = x.size()

    # If the padding is done on the height dimension
    if dim == 2 or dim == -2:
        # Create a new tensor padded in the height dimension
        padding_buffer = x.new_zeros(b, c, h + pad_pre + pad_post, w)
        # Copy the original tensor into the padded one
        padding_buffer[..., pad_pre:(h + pad_pre), :].copy_(x)

        # Copy the boundary elements of the original tensor
        # to the padding area in a way that is consistent with the
        # MATLAB implementation
        for p in range(pad_pre):
            padding_buffer[..., pad_pre - p - 1, :].copy_(x[..., p, :])
        for p in range(pad_post):
            padding_buffer[..., h + pad_pre + p, :].copy_(x[..., -(p + 1), :])

    # If the padding is done on the width dimension
    else:
        # Create a new tensor padded in the width dimension
        padding_buffer = x.new_zeros(b, c, h, w + pad_pre + pad_post)
        # Copy the original tensor into the padded one
        padding_buffer[..., pad_pre:(w + pad_pre)].copy_(x)

        # Copy the boundary elements of the original tensor
        # to the padding area in a way that is consistent with the
        # MATLAB implementation
        for p in range(pad_pre):
            padding_buffer[..., pad_pre - p - 1].copy_(x[..., p])
        for p in range(pad_post):
            padding_buffer[..., w + pad_pre + p].copy_(x[..., -(p + 1)])

    return padding_buffer

def padding(
        x: torch.Tensor,
        dim: int,
        pad_pre: int,
        pad_post: int,
        padding_type: str='reflect') -> torch.Tensor:

    if padding_type == 'reflect':
        x_pad = reflect_padding(x, dim, pad_pre, pad_post)
    else:
        raise ValueError('{} padding is not supported!'.format(padding_type))

    return x_pad

def get_padding(
        base: torch.Tensor,
        kernel_size: int,
        x_size: int) -> typing.Tuple[int, int, torch.Tensor]:
    # Convert the base tensor to long data type
    base = base.long()

    # Calculate the minimum value in the base tensor
    r_min = base.min()

    # Calculate the maximum value in the base tensor and adjust for kernel size
    r_max = base.max() + kernel_size - 1

    # Determine the amount of padding needed before the tensor
    if r_min <= 0:
        # If the minimum value is less than or equal to zero, calculate padding
        # required to bring it to zero or positive, and adjust the base tensor
        pad_pre = -r_min
        pad_pre = pad_pre.item()  # Convert to a Python integer
        base += pad_pre
    else:
        # No pre-padding needed if minimum value is already positive
        pad_pre = 0

    # Determine the amount of padding needed after the tensor
    if r_max >= x_size:
        # If the maximum value exceeds the size of the dimension being padded,
        # calculate the necessary post-padding
        pad_post = r_max - x_size + 1
        pad_post = pad_post.item()  # Convert to a Python integer
    else:
        # No post-padding needed if maximum value fits within the dimension size
        pad_post = 0

    # Return the calculated pre-padding, post-padding, and adjusted base tensor
    return pad_pre, pad_post, base

def get_weight(
        dist: torch.Tensor,
        kernel_size: int,
        kernel: str='cubic',
        sigma: float=2.0,
        antialiasing_factor: float=1) -> torch.Tensor:

    # Create a tensor to store the positions of the samples
    buffer_pos = dist.new_zeros(kernel_size, len(dist))

    # Iterate over the distances and calculate the positions of the samples
    for idx, buffer_sub in enumerate(buffer_pos):
        # Calculate the position of the sample by subtracting the index from the
        # distance
        buffer_sub.copy_(dist - idx)

    # Expand (downsampling) or shrink (upsampling) the receptive field by the
    # antialiasing factor
    buffer_pos *= antialiasing_factor

    # Calculate the weights by evaluating the kernel function at the positions
    if kernel == 'cubic':
        # Use the cubic kernel
        weight = cubic_contribution(buffer_pos)
    elif kernel == 'gaussian':
        # Use the gaussian kernel
        weight = gaussian_contribution(buffer_pos, sigma=sigma)
    else:
        # Raise an error if the kernel type is not supported
        raise ValueError('{} kernel is not supported!'.format(kernel))

    # Normalize the weights by dividing by the sum of the weights
    weight /= weight.sum(dim=0, keepdim=True)

    # Return the weights
    return weight

def reshape_tensor(x: torch.Tensor, dim: int, kernel_size: int) -> torch.Tensor:
    # Resize height
    if dim == 2 or dim == -2:
        k = (kernel_size, 1)
        h_out = x.size(-2) - kernel_size + 1
        w_out = x.size(-1)
    # Resize width
    else:
        k = (1, kernel_size)
        h_out = x.size(-2)
        w_out = x.size(-1) - kernel_size + 1

    unfold = F.unfold(x, k)
    unfold = unfold.view(unfold.size(0), -1, h_out, w_out)
    return unfold

def resize_1d(
        x: torch.Tensor,
        dim: int,
        side: int=None,
        kernel: str='cubic',
        sigma: float=2.0,
        padding_type: str='reflect',
        antialiasing: bool=True) -> torch.Tensor:

    scale = side / x.size(dim)
    # Identity case
    if scale == 1:
        return x

    # Default bicubic kernel with antialiasing (only when downsampling)
    if kernel == 'cubic':
        kernel_size = 4
    else:
        kernel_size = math.floor(6 * sigma)

    if antialiasing and (scale < 1):
        antialiasing_factor = scale
        kernel_size = math.ceil(kernel_size / antialiasing_factor)
    else:
        antialiasing_factor = 1

    # We allow margin to both sides
    kernel_size += 2

    # Weights only depend on the shape of input and output,
    # so we do not calculate gradients here.
    with torch.no_grad():
        d = 1 / (2 * side)
        pos = torch.linspace(
            start=d,
            end=(1 - d),
            steps=side,
            dtype=x.dtype,
            device=x.device,
        )
        pos = x.size(dim) * pos - 0.5
        base = pos.floor() - (kernel_size // 2) + 1
        dist = pos - base
        weight = get_weight(
            dist,
            kernel_size,
            kernel=kernel,
            sigma=sigma,
            antialiasing_factor=antialiasing_factor,
        )
        pad_pre, pad_post, base = get_padding(base, kernel_size, x.size(dim))

    # To backpropagate through x
    x_pad = padding(x, dim, pad_pre, pad_post, padding_type=padding_type)
    unfold = reshape_tensor(x_pad, dim, kernel_size)
    # Subsampling first
    if dim == 2 or dim == -2:
        sample = unfold[..., base, :]
        weight = weight.view(1, kernel_size, sample.size(2), 1)
    else:
        sample = unfold[..., base]
        weight = weight.view(1, kernel_size, 1, sample.size(3))

    # Apply the kernel
    down = sample * weight
    down = down.sum(dim=1, keepdim=True)
    return down

def downsampling_2d(
        x: torch.Tensor,
        k: torch.Tensor,
        scale: int,
        padding_type: str='reflect') -> torch.Tensor:

    c = x.size(1)
    k_h = k.size(-2)
    k_w = k.size(-1)

    k = k.to(dtype=x.dtype, device=x.device)
    k = k.view(1, 1, k_h, k_w)
    k = k.repeat(c, c, 1, 1)
    e = torch.eye(c, dtype=k.dtype, device=k.device, requires_grad=False)
    e = e.view(c, c, 1, 1)
    k = k * e

    pad_h = (k_h - scale) // 2
    pad_w = (k_w - scale) // 2
    x = padding(x, -2, pad_h, pad_h, padding_type=padding_type)
    x = padding(x, -1, pad_w, pad_w, padding_type=padding_type)
    y = F.conv2d(x, k, padding=0, stride=scale)
    return y

def imresize(
        x: torch.Tensor,
        scale: float=None,
        sides: typing.Tuple[int, int]=None,
        kernel: K='cubic',
        sigma: float=2,
        rotation_degree: float=0,
        padding_type: str='reflect',
        antialiasing: bool=True) -> torch.Tensor:
    '''
    Args:
        x (torch.Tensor): Input tensor to be resized.
        scale (float): Scale factor for resizing.
        sides (tuple(int, int)): Target size for the output (height, width).
        kernel (str, default='cubic'): Type of kernel to use for resizing.
        sigma (float, default=2): Standard deviation for the Gaussian kernel.
        rotation_degree (float, default=0): Degree of rotation (not implemented).
        padding_type (str, default='reflect'): Type of padding to use.
        antialiasing (bool, default=True): Whether to apply antialiasing.

    Return:
        torch.Tensor: Resized tensor.
    '''

    # Validate input parameters: either scale or sides must be specified
    if scale is None and sides is None:
        raise ValueError('One of scale or sides must be specified!')
    if scale is not None and sides is not None:
        raise ValueError('Please specify scale or sides to avoid conflict!')

    # Determine the dimensionality of the input tensor and extract size info
    if x.dim() == 4:
        b, c, h, w = x.size()
    elif x.dim() == 3:
        c, h, w = x.size()
        b = None
    elif x.dim() == 2:
        h, w = x.size()
        b = c = None
    else:
        raise ValueError('{}-dim Tensor is not supported!'.format(x.dim()))

    # Reshape tensor to 4D for uniform processing
    x = x.view(-1, 1, h, w)

    # If sides are not specified, calculate them based on the scale
    if sides is None:
        sides = (math.ceil(h * scale), math.ceil(w * scale))
        scale_inv = 1 / scale
        # Use a discrete kernel if the scale is an integer
        if isinstance(kernel, str) and scale_inv.is_integer():
            kernel = discrete_kernel(kernel, scale, antialiasing=antialiasing)
        # Raise an error if a predefined kernel is used without integer scale
        elif isinstance(kernel, torch.Tensor) and not scale_inv.is_integer():
            raise ValueError(
                'An integer downsampling factor '
                'should be used with a predefined kernel!'
            )

    # If the input tensor is not float, convert it to float for processing
    if x.dtype != torch.float32 or x.dtype != torch.float64:
        dtype = x.dtype
        x = x.float()
    else:
        dtype = None

    # Resize using specified kernel type
    if isinstance(kernel, str):
        # Shared keyword arguments for resizing
        kwargs = {
            'kernel': kernel,
            'sigma': sigma,
            'padding_type': padding_type,
            'antialiasing': antialiasing,
        }
        # Perform resizing along height and width
        x = resize_1d(x, -2, side=sides[0], **kwargs)
        x = resize_1d(x, -1, side=sides[1], **kwargs)
    elif isinstance(kernel, torch.Tensor):
        # Downsample using a predefined kernel
        x = downsampling_2d(x, kernel, scale=int(1 / scale))

    # Retrieve the resized dimensions
    rh = x.size(-2)
    rw = x.size(-1)
    # Reshape the tensor back to its original dimensionality
    if b is not None:
        x = x.view(b, c, rh, rw)        # 4D tensor
    else:
        if c is not None:
            x = x.view(c, rh, rw)       # 3D tensor
        else:
            x = x.view(rh, rw)          # 2D tensor

    # Convert tensor back to its original dtype if necessary
    if dtype is not None:
        if not dtype.is_floating_point:
            x = x.round()
        if dtype is torch.uint8:
            x = x.clamp(0, 255)
        x = x.to(dtype=dtype)

    return x

if __name__ == '__main__':
    # Just for debugging
    torch.set_printoptions(precision=4, sci_mode=False, edgeitems=16, linewidth=200)
    a = torch.arange(64).float().view(1, 1, 8, 8)
    z = imresize(a, 0.5)
    print(z)
    #a = torch.arange(16).float().view(1, 1, 4, 4)
    '''
    a = torch.zeros(1, 1, 4, 4)
    a[..., 0, 0] = 100
    a[..., 1, 0] = 10
    a[..., 0, 1] = 1
    a[..., 0, -1] = 100
    a = torch.zeros(1, 1, 4, 4)
    a[..., -1, -1] = 100
    a[..., -2, -1] = 10
    a[..., -1, -2] = 1
    a[..., -1, 0] = 100
    '''
    #b = imresize(a, sides=(3, 8), antialiasing=False)
    #c = imresize(a, sides=(11, 13), antialiasing=True)
    #c = imresize(a, sides=(4, 4), antialiasing=False, kernel='gaussian', sigma=1)
    #print(a)
    #print(b)
    #print(c)

    #r = discrete_kernel('cubic', 1 / 3)
    #print(r)
    '''
    a = torch.arange(225).float().view(1, 1, 15, 15)
    imresize(a, sides=[5, 5])
    '''