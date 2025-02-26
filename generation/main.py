"""
This project is based on https://github.com/kligvasser/SinGAN. We modified the code to fit our needs.
We modified the parameters to fit our needs and added a few more parameters to the code.
We added the detailed explanation of the code.
"""
import argparse
import torch
import logging
import signal
import sys
import torch.backends.cudnn as cudnn
from trainer import Trainer
from datetime import datetime
from os import path
from utils import misc
from random import randint


def get_arguments():
    parser = argparse.ArgumentParser(description='SinGAN: Learning a Generative Model from a Single Natural Image')
    parser.add_argument('--device', default='cuda', help='device assignment ("cpu" or "cuda")')
    parser.add_argument('--device-ids', default=[0], type=int, nargs='+', help='device ids assignment (e.g 0 1 2 3)')

    parser.add_argument('--gen-model', default='g_multivanilla',
                        help='generator architecture (default: g_multivanilla)')
    parser.add_argument('--dis-model', default='d_vanilla', help='discriminator architecture (default: d_vanilla)')
    parser.add_argument('--min-features', default=32, type=int, help='minimum features (default: 32)')
    parser.add_argument('--max-features', default=32, type=int, help='maximum features (default: 32)')
    parser.add_argument('--num-blocks', default=5, type=int, help=' (default: 5)')
    parser.add_argument('--kernel-size', default=3, type=int, help=' (default: 3)')
    parser.add_argument('--padding', default=0, type=int, help=' (default: 0)')

    parser.add_argument('--root', default='', help='image source')
    parser.add_argument('--min-size', default=20, type=int, help='minimum scale size (default: 25)')
    parser.add_argument('--max-size', default=200, type=int, help='maximum scale size  (default: 250)')
    parser.add_argument('--scale-factor-init', default=0.75, type=float,
                        help='initilize scaling factor (default: 0.75)')
    parser.add_argument('--noise-weight', default=0.1, type=float, help='noise amplitude (default: 0.1)')

    parser.add_argument('--batch-size', default=1, type=int, help='batch-size (default: 1)')
    parser.add_argument('--crop-size', default=0, type=int, help='cropping-size of last scale (default: 0)')
    parser.add_argument('--num-steps', default=2000, type=int, help='number of steps per scale (default: 4000)')
    parser.add_argument('--lr', default=4e-4, type=float, help='learning rate (default: 5e-4)')
    parser.add_argument('--gen-betas', default=[0.5, 0.9], nargs=2, type=float, help='adam betas (default: 0.5 0.9)')
    parser.add_argument('--dis-betas', default=[0.5, 0.9], nargs=2, type=float, help='adam betas (default: 0.5 0.9)')
    parser.add_argument('--num-critic', default=1, type=int, help='critic iterations (default: 1)')
    parser.add_argument('--step-size', default=2000, type=int, help='scheduler step size (default: 2000)')
    parser.add_argument('--gamma', default=0.1, type=float, help='scheduler gamma (default: 0.1)')
    parser.add_argument('--penalty-weight', default=0.1, type=float, help='gradient penalty weight (default: 0.1)')
    parser.add_argument('--reconstruction-weight', default=10., type=float, help='reconstruction-weight (default: 10)')
    parser.add_argument('--adversarial-weight', default=1., type=float, help='adversarial-weight (default: 1)')

    parser.add_argument('--seed', default=-1, type=int, help='random seed (default: random)')
    parser.add_argument('--print-every', default=200, type=int, help='print-every (default: 200)')
    parser.add_argument('--eval-every', default=100, type=int, help='eval-every (default: 100)')
    parser.add_argument('--results-dir', metavar='RESULTS_DIR', default='./results', help='results dir')
    parser.add_argument('--save', metavar='SAVE', default='', help='saved folder')
    parser.add_argument('--evaluation', default=False, action='store_true', help='evaluate a model (default: false)')
    parser.add_argument('--model-to-load', default='', help='evaluating from file (default: None)')
    parser.add_argument('--amps-to-load', default='', help='evaluating from file (default: None)')
    parser.add_argument('--use-tb', default=False, action='store_true', help='use tensorboardx (default: false)')
    args = parser.parse_args()

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.save == '':
        args.save = time_stamp
    args.save_path = path.join(args.results_dir, args.save)
    if args.seed == -1:
        args.seed = randint(0, 12345)
    return args


def main():
    """
    This is the main entry point of the program. It parses arguments, sets up
    the environment, and starts the training or evaluation process.
    """
    # arguments
    args = get_arguments()

    # set the random seed. This is important for reproducibility of the results
    torch.manual_seed(args.seed)

    # If we are using a CUDA device, set the device to the first device in the
    # list of devices specified in the arguments. Also, set the random seed for
    # CUDA. This is important because the random number generator is not
    # deterministic when using CUDA.
    if 'cuda' in args.device and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(args.device_ids[0])
        # Enable cudnn benchmarking mode. This is a mode that allows cuDNN to
        # optimize the algorithms used for convolutional neural networks based
        # on the input data.
        cudnn.benchmark = True
    else:
        # If we are not using a CUDA device, set the list of device IDs to None.
        args.device_ids = None

    # Create a directory for saving the results, and set up the logging
    # configuration. The log file will be saved in the result directory.
    misc.mkdir(args.save_path)
    misc.setup_logging(path.join(args.save_path, 'log.txt'))

    # Print the arguments to the log file
    logging.info(args)

    # Create an instance of the Trainer class, which is responsible for
    # training or evaluating the model.
    trainer = Trainer(args)

    # If the evaluation flag is set, evaluate the model. Otherwise, train the
    # model.
    if args.evaluation:
        trainer.eval()
    else:
        trainer.train()


if __name__ == '__main__':
    # enables a ctrl-c without triggering errors
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))
    main()
