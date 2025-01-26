import torch
import random
from math import floor
from torchvision import transforms
from PIL import Image

class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, root='', batch_size=1, crop_size=0):
        """
        Initialize dataset with options

        Parameters:
        root (str): root directory of the dataset
        batch_size (int): number of samples per batch to load
        crop_size (int): size of the crop to apply to the images
        """
        # root directory of the dataset
        self.root = root
        # number of samples per batch to load
        self.batch_size = batch_size
        # size of the crop to apply to the images
        self.crop_size = crop_size

        # initialize
        self._init()

    def _init(self):
        """
        Initialize image and prepare for further modification
        """

        # convert PIL image to tensor
        self.to_tensor = transforms.ToTensor()

        # open image
        image = Image.open(self.root).convert('RGB')
        self.image = self.to_tensor(image).unsqueeze(dim=0)
        # normalize to range [-1, 1]
        self.image = (self.image - 0.5) * 2

        # set from outside
        self.reals = None
        self.noises = None
        self.amps = None

    def _get_augment_params(self, size):
        """
        Get random position and flip parameters for random crop augmentation.

        Parameters:
        size (tuple): size of the image to be augmented (w, h)

        Returns:
        dict: a dictionary with random position and flip parameters
        """
        # get a new random seed
        random.seed(random.randint(0, 12345))

        # get image size
        w_size, h_size = size

        # get a random position within the image
        # make sure the crop size does not exceed the image size
        x = random.randint(0, max(0, w_size - self.crop_size))
        y = random.randint(0, max(0, h_size - self.crop_size))

        # randomly flip the image
        # 0.5 because it's a 50% chance
        flip = random.random() > 0.5

        # return the augmentation parameters
        return {'pos': (x, y), 'flip': flip}

    def _augment(self, image, aug_params, scale=1):
        # Extract the random position from the augmentation parameters
        x, y = aug_params['pos']

        # Calculate the crop region using the given scale
        # Crop the image based on the position and crop size, adjusted by the scale
        image = image[:, 
                      round(x * scale):(round(x * scale) + round(self.crop_size * scale)), 
                      round(y * scale):(round(y * scale) + round(self.crop_size * scale))]

        # Check if the flip parameter is set
        # If true, horizontally flip the image
        if aug_params['flip']:
            image = image.flip(-1)

        # Return the augmented image
        return image

    def __getitem__(self, index):
        """
        Returns a batch of images and corresponding noise/amp values.
        This is the core function of the dataset class, which is called by the DataLoader.

        Parameters:
        index (int): a batch index

        Returns:
        dict: a dictionary with the following keys:
            reals (dict): a dictionary with the real images at different scales
            noises (dict): a dictionary with the corresponding noise values at different scales
            amps (tensor): a tensor with the amplification values for each scale
        """
        amps = self.amps

        # cropping
        if self.crop_size:
            # Initialize empty dictionaries to store the cropped images and noises
            reals, noises = {}, {}

            # Get a set of random augmentation parameters
            # The parameters include a random position and a flip boolean
            aug_params = self._get_augment_params(self.image.size()[-2:])

            # Loop over the available scales
            for key in self.reals.keys():
                # Calculate the scale
                scale = self.reals[key].size(-1) / float(self.image.size(-1))

                # Crop the real image and the corresponding noise using the augmentation parameters
                # The _augment function takes the image/noise and the augmentation parameters
                # and returns the cropped result
                reals.update({key: self._augment(self.reals[key].clone(), aug_params, scale)})
                noises.update({key: self._augment(self.noises[key].clone(), aug_params, scale)})

        # full size
        else:
            # No cropping needed, just copy the original data
            reals = self.reals #TODO: clone when crop
            noises = self.noises #TODO: clone when crop

        # Return the batch
        return {'reals': reals, 'noises': noises, 'amps': amps}
       
    def __len__(self):
        return self.batch_size