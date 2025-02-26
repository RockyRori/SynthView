import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm

__all__ = ['d_vanilla', 'd_snvanilla']


def initialize_model(model, scale=1.):
    """
    Initializes weights of a PyTorch model by sampling from a normal distribution
    with a standard deviation of 0.02 and multiplying the result by a scale factor.
    
    This function is called by the constructors of the generator and discriminator
    models. It is used to initialize the weights of the models before they are
    used for training.
    
    The function takes a PyTorch model and an optional scale factor as arguments.
    The scale factor defaults to 1, which means that the weights are sampled from
    a normal distribution with a standard deviation of 0.02.
    
    If the module is not an instance of nn.Conv2d or nn.BatchNorm2d, it is
    skipped.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # Sample weights from a normal distribution with a standard deviation
            # of 0.02, multiply the result by the scale factor, and set the biases
            # to 0.
            m.weight.data.normal_(0.0, 0.02)
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            # Sample weights from a normal distribution with a standard deviation
            # of 0.02, and set the biases to 0.
            m.weight.data.normal_(1.0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        else:
            # If the module is not an instance of nn.Conv2d or nn.BatchNorm2d,
            # skip it.
            continue


class BasicBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True, normalization=False):
        """
        Initializes a BasicBlock instance.
        
        Args:
            in_channels (int): The number of input channels. Defaults to 64.
            out_channels (int): The number of output channels. Defaults to 64.
            kernel_size (int): The size of the convolution kernel. Defaults to 3.
            padding (int): The amount of padding to apply to the input. Defaults to 1.
            bias (bool): If True, a bias is added to the output of the convolution
                operation. Defaults to True.
            normalization (bool): If True, the weights of the convolution operation
                are normalized according to the Spectral Normalization method.
                Defaults to False.
        """
        super(BasicBlock, self).__init__()

        # Create a convolutional layer with the specified number of input and
        # output channels, kernel size, and padding.
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              padding=padding, bias=bias)

        # Create a batch normalization layer with the same number of features
        # as the output of the convolutional layer.
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)

        # Create a leaky ReLU activation function with a negative slope of 0.2.
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # If normalization is True, wrap the convolutional layer in a
        # SpectralNorm layer.
        if normalization:
            # Spectral Normalization is a method for normalizing the weights of
            # a neural network. It normalizes the weights of the network by
            # dividing them by their spectral norm, which is the maximum singular
            # value of the weights.
            self.conv = SpectralNorm(self.conv)

    def forward(self, x):
        # Apply the convolutional layer to the input
        x = self.conv(x)

        # Apply batch normalization to the output of the convolutional layer
        # Batch normalization is a method for normalizing the input to a layer
        # by subtracting the mean and dividing by the standard deviation.
        # Batch normalization helps to reduce overfitting by making the model
        # more robust to changes in the input.
        x = self.batch_norm(x)

        # Apply the leaky ReLU activation function to the output of the batch
        # normalization layer.
        # The leaky ReLU activation function is a variant of the ReLU activation
        # function that multiplies the input by a small value (in this case,
        # 0.2) to prevent the output from being zero.
        x = self.lrelu(x)

        # Return the output of the forward pass
        return x


class Vanilla(nn.Module):
    def __init__(self, in_channels, max_features, min_features, num_blocks, kernel_size, padding, normalization):
        super(Vanilla, self).__init__()

        # features
        # The features module is a sequence of BasicBlock layers. Each BasicBlock
        # layer applies a convolutional layer followed by batch normalization and
        # a leaky ReLU activation function. The BasicBlock layers are applied in
        # sequence, with the output of each layer being the input to the next layer.
        blocks = []
        # The first BasicBlock layer takes the input to the features module and
        # applies a convolutional layer with the same number of output channels as
        # the maximum number of features specified in the constructor.
        blocks.append(
            BasicBlock(in_channels=in_channels, out_channels=max_features, kernel_size=kernel_size, padding=padding,
                       normalization=normalization))
        # The remaining BasicBlock layers take the output of the previous layer and
        # apply a convolutional layer with the number of output channels being the
        # maximum of the minimum number of features specified in the constructor
        # and the number of features of the previous layer divided by 2.
        for i in range(0, num_blocks - 2):
            f = max_features // pow(2, (i + 1))
            blocks.append(BasicBlock(in_channels=max(min_features, f * 2), out_channels=max(min_features, f),
                                     kernel_size=kernel_size, padding=padding, normalization=normalization))
        # The features module is a sequence of the BasicBlock layers.
        self.features = nn.Sequential(*blocks)

        # classifier
        # The classifier module is a convolutional layer that takes the output of
        # the features module and applies a convolutional layer with one output
        # channel. The output of the classifier is the output of the network.
        self.classifier = nn.Conv2d(in_channels=max(f, min_features), out_channels=1, kernel_size=kernel_size,
                                    padding=padding)

        # initialize weights
        # The weights of the network are initialized using the initialize_model
        # function.
        initialize_model(self)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def d_vanilla(**config):
    config.setdefault('in_channels', 3)
    config.setdefault('min_features', 32)
    config.setdefault('max_features', 32)
    config.setdefault('num_blocks', 5)
    config.setdefault('kernel_size', 3)
    config.setdefault('padding', 0)
    config.setdefault('normalization', False)

    return Vanilla(**config)


def d_snvanilla(**config):
    config.setdefault('in_channels', 3)
    config.setdefault('min_features', 32)
    config.setdefault('max_features', 32)
    config.setdefault('num_blocks', 5)
    config.setdefault('kernel_size', 3)
    config.setdefault('padding', 0)
    config.setdefault('normalization', True)

    return Vanilla(**config)
