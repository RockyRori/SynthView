from .datasets import  Dataset
from torch.utils.data import DataLoader

def get_loader(args):
    # Initialize the dataset with provided arguments
    # root: directory path to the dataset
    # batch_size: number of samples per batch to load
    # crop_size: size of the crop to apply to the images
    dataset = Dataset(root=args.root, batch_size=args.batch_size, crop_size=args.crop_size)

    # Create a data loader for the dataset
    # batch_size: number of samples per batch to load (same as dataset)
    # shuffle: whether to shuffle the dataset at every epoch (set to False here)
    # num_workers: number of subprocesses to use for data loading (0 means data will be loaded in the main process)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Return the data loader
    return loader
