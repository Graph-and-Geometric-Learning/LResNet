# -----------------------------------------------------
# Change working directory to parent HyperbolicCV/code
import os
import sys

working_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../")
os.chdir(working_dir)

lib_path = os.path.join(working_dir)
sys.path.append(lib_path)
# -----------------------------------------------------

import torch
import torch.nn.functional as F
from torch.nn import DataParallel

import configargparse

import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.initialize import select_dataset, select_model, load_model_checkpoint
from train import evaluate

from utils.utils import AverageMeter, accuracy

def getArguments():
    """ Parses command-line options. """
    parser = configargparse.ArgumentParser(description='Image classification training', add_help=True)

    parser.add_argument('-c', '--config_file', required=False, default=None, is_config_file=True, type=str, 
                        help="Path to config file.")
    
    # Modes
    parser.add_argument('--mode', default="test_accuracy", type=str, 
                        choices=[
                            "test_accuracy",
                            "visualize_embeddings",
                            "fgsm",
                            "pgd"
                        ],
                        help = "Select the testing mode.")
    
    # Output settings
    parser.add_argument('--output_dir', default=None, type=str, 
                        help = "Path for output files (relative to working directory).")

    # General settings
    parser.add_argument('--device', default="cuda:0", type=lambda s: [str(item) for item in s.replace(' ','').split(',')],
                        help="List of devices split by comma (e.g. cuda:0,cuda:1), can also be a single device or 'cpu')")
    parser.add_argument('--dtype', default='float32', type=str, choices=["float32", "float64"], 
                        help="Set floating point precision.")
    parser.add_argument('--seed', default=1, type=int, 
                        help="Set seed for deterministic training.")
    parser.add_argument('--load_checkpoint', default=None, type=str, 
                        help = "Path to model checkpoint.")

    # Testing parameters
    parser.add_argument('--batch_size', default=128, type=int, 
                        help="Training batch size.")
    parser.add_argument('--batch_size_test', default=128, type=int, 
                        help="Testing batch size.")

    # Model selection
    parser.add_argument('--num_layers', default=18, type=int, choices=[18, 50], 
                        help = "Number of layers in ResNet.")
    parser.add_argument('--embedding_dim', default=512, type=int, 
                        help = "Dimensionality of classification embedding space (could be expanded by ResNet)")
    parser.add_argument('--encoder_manifold', default='lrn', type=str, 
                        help = "Select conv model encoder manifold.")
    parser.add_argument('--decoder_manifold', default='lrn', type=str, 
                        help = "Select conv model decoder manifold.")
    

    # Hyperbolic geometry settings
    parser.add_argument('--learn_k', action='store_true',
                        help="Set a learnable curvature of hyperbolic geometry.")
    parser.add_argument('--encoder_k', default=1.0, type=float, 
                        help = "Initial curvature of hyperbolic geometry in backbone (geoopt.K=-1/K).")
    parser.add_argument('--decoder_k', default=1.0, type=float, 
                        help = "Initial curvature of hyperbolic geometry in decoder (geoopt.K=-1/K).")
    parser.add_argument('--clip_features', default=1.0, type=float, 
                        help = "Clipping parameter for hybrid HNNs proposed by Guo et al. (2022)")
    
    # Dataset settings
    parser.add_argument('--dataset', default='CIFAR-100', type=str, choices=["MNIST", "CIFAR-10", "CIFAR-100", "Tiny-ImageNet"], 
                        help = "Select a dataset.")


    args, _ = parser.parse_known_args()

    return args


def main(args):
    device = args.device[0]
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    print("Arguments:")
    print(args)

    print("Loading dataset...")
    train_loader, _, test_loader, img_dim, num_classes = select_dataset(args, validation_split=False)

    print("Creating model...")
    model = select_model(img_dim, num_classes, args)
    model = model.to(device)
    print('-> Number of model params: {} (trainable: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    if args.load_checkpoint is not None:
        print("Loading model checkpoint from {}".format(args.load_checkpoint))
        model = load_model_checkpoint(model, args.load_checkpoint)
    else:
        print("No model checkpoint given. Using random weights.")

    model = DataParallel(model, device_ids=args.device)
    model.eval()

    if args.mode=="test_accuracy":
        print("Testing accuracy of model...")
        criterion = torch.nn.CrossEntropyLoss()
        loss_test, acc1_test, acc5_test = evaluate(model, test_loader, criterion, device)
        print("Results: Loss={:.4f}, Acc@1={:.4f}, Acc@5={:.4f}".format(
            loss_test, acc1_test, acc5_test))
        

    else:
        print(f"Mode {args.mode} not implemented yet.")

    print("Finished!")


if __name__ == '__main__':
    args = getArguments()

    if args.dtype == "float64":
        torch.set_default_dtype(torch.float64)
    elif args.dtype == "float32":
        torch.set_default_dtype(torch.float32)
    else:
        raise "Wrong dtype in configuration -> " + args.dtype
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    main(args)