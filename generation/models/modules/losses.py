import torch
from utils.core import imresize


class ConsistencyLoss(torch.nn.Module):
    def __init__(self, scale=0.5, criterion=torch.nn.MSELoss()):
        """
        Initialize the Consistency loss module.

        Parameters
        ----------
        scale : float
            The scale of the image to be resized before calculating the loss.
            The default is 0.5.
        criterion : torch.nn.Module
            The loss function to use. The default is torch.nn.MSELoss().
        """
        super(ConsistencyLoss, self).__init__()
        self.scale = scale
        self.criterion = criterion

    def forward(self, inputs, targets):
        """
        Calculate the loss between the `inputs` and `targets`.

        Parameters
        ----------
        inputs : torch.Tensor
            The input tensor to be compared with the targets.
        targets : torch.Tensor
            The target tensor used as the ground truth.

        Returns
        -------
        loss : torch.Tensor
            The calculated loss between the `inputs` and `targets`.
        """
        # resize the targets to the desired scale
        targets = imresize(targets, scale=self.scale)

        # resize the inputs to the desired scale
        inputs = imresize(inputs, scale=self.scale)

        # calculate the loss between the resized inputs and targets
        # note that the targets are detached from the computation graph
        # so that the gradients are not propagated through them
        loss = self.criterion(inputs, targets.detach())

        return loss
