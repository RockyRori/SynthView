import torch
import math
import torch.nn as nn
from utils.core import imresize
from copy import deepcopy
from torch.nn import functional as F

__all__ = ['g_multivanilla']

def initialize_model(model, scale=1.):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # Initialize the weights with a normal distribution
            # with a mean of 0 and a standard deviation of
            # 0.02. This is the same as in the original paper.
            m.weight.data.normal_(0.0, 0.02)
            # Multiply the weights by the scale factor
            m.weight.data *= scale
            # Set the biases to 0
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            # Initialize the weights with a normal distribution
            # with a mean of 1.0 and a standard deviation of
            # 0.02. This is the same as in the original paper.
            m.weight.data.normal_(1.0, 0.02)
            # Set the biases to 0
            if m.bias is not None:
                m.bias.data.zero_()
        else:
            # If the layer is not a convolutional or batch norm
            # layer, we don't need to do anything.
            continue

class BasicBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True):
        super(BasicBlock, self).__init__()

        # The convolutional layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,  # The number of input channels
            out_channels=out_channels,  # The number of output channels
            kernel_size=kernel_size,  # The size of the kernel
            padding=padding,  # The amount of padding
            bias=bias  # Whether or not to use bias
        )

        # The batch normalization layer
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)

        # The LeakyReLU activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.lrelu(self.batch_norm(self.conv(x)))
        return x

class Vanilla(nn.Module):
    def __init__(self, in_channels, max_features, min_features, num_blocks, kernel_size, padding):
        super(Vanilla, self).__init__()

        # Calculate the total padding needed based on the number of blocks
        self.padding = (kernel_size // 2) * num_blocks

        # Create the feature extraction module
        # The feature extraction module is a sequence of BasicBlock layers.
        # Each BasicBlock layer applies a convolutional layer followed by
        # batch normalization and a leaky ReLU activation function.
        blocks = []
        # The first BasicBlock layer takes the input to the features module
        # and applies a convolutional layer with the same number of output
        # channels as the maximum number of features specified in the
        # constructor.
        blocks.append(BasicBlock(in_channels=in_channels, out_channels=max_features, kernel_size=kernel_size, padding=padding))
        # The remaining BasicBlock layers take the output of the previous
        # layer and apply a convolutional layer with the number of output
        # channels being the maximum of the minimum number of features
        # specified in the constructor and the number of features of the
        # previous layer divided by 2.
        for i in range(0, num_blocks - 2):
            f = max_features // pow(2, (i+1))
            blocks.append(BasicBlock(in_channels=max(min_features, f * 2), out_channels=max(min_features, f), kernel_size=kernel_size, padding=padding))
        # The features module is a sequence of the BasicBlock layers.
        self.features = nn.Sequential(*blocks)

        # Create the module that takes the output of the feature extraction
        # module and converts it to an image.
        self.features_to_image = nn.Sequential(
            # Apply a convolutional layer with the same number of output
            # channels as the input to the features module.
            nn.Conv2d(in_channels=max(f, min_features), out_channels=in_channels, kernel_size=kernel_size, padding=padding),
            # Apply a hyperbolic tangent activation function to ensure the
            # output is between -1 and 1.
            nn.Tanh())

        # Initialize the weights of the network using the initialize_model
        # function.
        initialize_model(self)

    def forward(self, z, x):
        z = F.pad(z, [self.padding, self.padding, self.padding, self.padding])
        z = self.features(z)
        z = self.features_to_image(z)
        
        return x + z

class MultiVanilla(nn.Module):
    def __init__(self, in_channels, max_features, min_features, num_blocks, kernel_size, padding):
        super(MultiVanilla, self).__init__()
        # parameters
        self.in_channels = in_channels
        self.max_features = max_features
        self.min_features = min_features
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.padding = padding
        self.scale = 0
        self.key = 's0'
        self.scale_factor = 0

        # current
        self.curr = Vanilla(in_channels, max_features, min_features, num_blocks, kernel_size, padding)
        self.prev = nn.Module()

    def add_scale(self, device):
        # Increment the scale number
        self.scale += 1

        # previous
        # Add the current generator to the previous module dictionary
        # using the current key.
        self.prev.add_module(self.key, deepcopy(self.curr))
        # Reset the gradients of the previous module to false
        self._reset_grad(self.prev, False)
        # Update the key to the new scale.
        self.key = 's{}'.format(self.scale)

        # current
        # Calculate the new maximum and minimum number of features
        # based on the scale.
        max_features = min(self.max_features * pow(2, math.floor(self.scale / 4)), 128)
        min_features = min(self.min_features * pow(2, math.floor(self.scale / 4)), 128)
        # If the scale has changed, update the current generator
        # to the new maximum and minimum number of features.
        if math.floor(self.scale / 4) != math.floor((self.scale - 1) / 4):
            self.curr = Vanilla(self.in_channels, max_features, min_features, self.num_blocks, self.kernel_size, self.padding).to(device)

    def _compute_previous(self, reals, amps, noises=None):
        # parameters
        keys = list(reals.keys())
        y = torch.zeros_like(reals[keys[0]])
        
        # loop over scales
        for key, single_scale in self.prev.named_children():
            # get the next key
            next_key = keys[keys.index(key) + 1]
            # fixed z
            if noises:
                # add the noise to the output of the previous generator
                z = y + amps[key].view(-1, 1, 1, 1) * noises[key]
            # random noise
            else:
                # generate a random noise
                n = self._generate_noise(reals[key], repeat=(key == 's0'))
                # add the noise to the output of the previous generator
                z = y + amps[key].view(-1, 1, 1, 1) * n
            # apply the current generator
            y = single_scale(z, y)
            # resize the output to the size of the next scale
            y = imresize(y, 1. / self.scale_factor)
            # crop the output to the size of the next scale
            y = y[:, :, 0:reals[next_key].size(2), 0:reals[next_key].size(3)]
            
        return y

    def forward(self, reals, amps, noises=None):
        # Compute the output of the previous generator
        # This is the "previous layer" in the paper
        with torch.no_grad():
            # Compute the output of the previous generator
            # This is the "previous layer" in the paper
            # The output of the previous generator is computed by
            # recursively applying the previous generators in the
            # sequence of generators
            y = self._compute_previous(reals, amps, noises).detach()
            
        # If we are given a fixed noise, use it
        # Otherwise, generate a random noise
        if noises:
            # If we are given a fixed noise, use it
            z = y + amps[self.key].view(-1, 1, 1, 1) * noises[self.key]
        else:
            # Otherwise, generate a random noise
            n = self._generate_noise(reals[self.key], repeat=(not self.scale))
            # Add the noise to the output of the previous generator
            z = y + amps[self.key].view(-1, 1, 1, 1) * n

        # Compute the output of the current generator
        # This is the "current layer" in the paper
        o = self.curr(z.detach(), y.detach()) 
        # Return the output of the current generator
        return o

    def _generate_noise(self, tensor_like, repeat=False):
        if not repeat:
            # Generate a full-size noise tensor
            noise = torch.randn(tensor_like.size()).to(tensor_like.device)
        else:
            # Generate a noise tensor that is the same size as the given tensor_like,
            # but with only 1 channel (i.e. a grayscale noise tensor).
            # The noise tensor is generated by repeating the same random values
            # along the channel dimension.
            noise = torch.randn((tensor_like.size(0), 1, tensor_like.size(2), tensor_like.size(3)))
            noise = noise.repeat((1, 3, 1, 1)).to(tensor_like.device)

        return noise

    def _reset_grad(self, model, require_grad=False):
        for p in model.parameters():
            p.requires_grad_(require_grad)

    def train(self, mode=True):
        self.training = mode
        # train
        for module in self.curr.children():
            module.train(mode)
        # eval
        for module in self.prev.children():
            module.train(False)
        return self

    def eval(self):
        self.train(False)

def g_multivanilla(**config):
    config.setdefault('in_channels', 3)
    config.setdefault('min_features', 32)
    config.setdefault('max_features', 32)
    config.setdefault('num_blocks', 5)
    config.setdefault('kernel_size', 3)
    config.setdefault('padding', 0)
    
    return MultiVanilla(**config)