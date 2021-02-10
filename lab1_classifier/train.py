"""
Created by Edward Li at 2/9/21
"""
import torch
import tqdm
import os
import argparse
import datetime
from torch.utils.data import DataLoader
import torch.optim as optim
from model import MLPClassifier
from data import MnistDataset


def train(args):
    # seed everything for reproduceability
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    # prefer use Gpu for everything
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    now = datetime.datetime.now()
    out_dir = args.output_dir + "_" + now.strftime("%Y-%m-%d_%H_%M") + "_mnist_classifier"
    # make output directory
    os.makedirs(out_dir,exist_ok=False)

    # create our model
    model = MLPClassifier().to(device)

    # we will use SGD + momentum optimizer for this task
    optimizer = optim.SGD(model.parameters(),lr=args.learning_rate, momentum=args.momentum)

    # dataset
    dataset = MnistDataset("../mnist_data")

    training_dataset = DataLoader(dataset.training_loader,batch_size=args.batch_size,shffle=True)
    # don't shuffle validation dataset for consistent loss
    validation_dataset = DataLoader(dataset.test_loader,batch_size=args.batch_size,shuffle=False)




if __name__ == "__main__" :
    parser = argparse.ArgumentParser("mnist classifier arg parser")
    parser.add_argument("--epochs", help="number of epochs to train", type = int, default=10)
    parser.add_argument("--batch_size", help="number of images in a batch", type=int, default=64)
    parser.add_argument("--learning_rate", help="learning rate", type = float, default= 0.001)
    parser.add_argument("--momentum", help="momentum", type = int, default=0.01)
    parser.add_argument("--seed", help="random seed for reproducibility", type = int, default=24)
    parser.add_argument("--log_interval", help="interval for logging cool info", type = int, default=10)
    parser.add_argument("--output_dir", help="directory for storing outputs", type = int, default="./output/")
    args = parser.parse_args()

    train(args)

