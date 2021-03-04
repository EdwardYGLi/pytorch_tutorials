"""
Created by Edward Li at 2/9/21
"""
"""
Created by Edward Li at 2/9/21
"""
import argparse
import datetime
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from data import MnistDataset
from model import FullyConectedAutoEncoder


def calculate_psnr(pred, target):
    # implement PSNR (peak to peak signal to noise ratio) or SSIM (structured similarity index),
    # rename your function if implementing SSIM
    # between prediction and target here
    mse = torch.mean((pred - target) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def my_loss_fn(pred, target):
    # define your loss function here. or define it whithin the loop.
    return F.mse_loss(pred, target)


def eval(model, out_dir, data, labels, epoch, device, fig, axs):
    data, target = data.to(device), labels.to(device)
    with torch.no_grad():
        output, latent = model(data)
        # create some plots here to visualize the outputs, you can also use cv2 instead of matplot lib.
        for i in range(0, len(axs), 2):
            ax1 = axs[i]
            ax2 = axs[i+1]
            ind = i//2
            im = output[ind][0].detach().cpu().numpy()
            gt = labels[ind][0].detach().cpu().numpy()
            ax1.imshow(im, cmap="gray", interpolation="none")
            ax2.imshow(gt, cmap="gray", interpolation="none")
            ax1.set_title("pred",fontsize=20)
            ax2.set_title("gt",fontsize=20)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_aspect('equal')
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_aspect('equal')

    plt.savefig(os.path.join(out_dir, "eval_epoch_{}.png".format(epoch)))


def train(args):
    # seed everything for reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    # prefer use Gpu for everything
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create our model
    model = FullyConectedAutoEncoder().to(device)
    print("model")
    print(model)

    now = datetime.datetime.now()
    out_dir = os.path.join(args.output_dir, now.strftime("%Y-%m-%d_%H_%M") + "_{}".format(model.__class__.__name__))
    # make output directory
    os.makedirs(out_dir, exist_ok=True)

    # we will use SGD + momentum optimizer for this task
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    # dataset
    dataset = MnistDataset("../mnist_data")

    training_dataloader = DataLoader(dataset.training_dataset, batch_size=args.batch_size, shuffle=True)
    # don't shuffle validation dataset for consistent loss
    validation_dataloader = DataLoader(dataset.test_dataset, batch_size=args.batch_size, shuffle=False)

    example_data, example_labels = next(iter(validation_dataloader))
    # reuse same figure
    fig, axs = plt.subplots(4, 4, figsize=(20, 20), facecolor='w', edgecolor='k', num=1)
    fig.subplots_adjust(hspace=.1, wspace=.1)
    fig.tight_layout()

    axs = axs.ravel()

    dataloaders = {"val": validation_dataloader,
                   "train": training_dataloader}

    losses_charts = {"val":
                         {"loss": [],
                          "psnr": [],
                          "step": []},
                     "train":
                         {"loss": [],
                          "step": []}
                     }

    for epoch in range(args.epochs):
        for phase in ["val", "train"]:
            dataloader = dataloaders[phase]
            if phase == "train":
                model.train()
            else:
                model.eval()
                # generate and store some eval images
                eval(model, out_dir, example_data, example_data, epoch, device, fig, axs)
                # save a checkpoint
                torch.save(model.state_dict(), os.path.join(out_dir, "epoch_{}_ckpt.pt".format(epoch)))

            back_prop = phase == "train"
            epoch_loss = 0
            psnr = 0
            print(phase)
            with torch.set_grad_enabled(back_prop):
                # [hint] you can modify what your target is for the auto encoder training.
                # instead of using classification labels as target
                for b, (data, target) in enumerate(dataloader):
                    data, target = data.to(device), data.to(device)

                    if back_prop:
                        # clear gradients
                        optimizer.zero_grad()
                    output, latent = model(data)

                    # Select a suitable loss function here.
                    loss = my_loss_fn(output, target)
                    if back_prop:
                        # get gradients
                        loss.backward()
                        # update weights
                        optimizer.step()
                    else:
                        # instead of accuracy implement PSNR (peak to peak signal to noise ratio)
                        # or SSIM (structured similarity index)
                        # between prediction and target
                        psnr += calculate_psnr(output, target)
                    epoch_loss += loss.item()

                    if b % args.log_interval == 0 and phase == "train":
                        print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(phase,
                                                                                    epoch, b * len(data),
                                                                                    len(dataloader.dataset),
                                                                                    100. * b * len(data) / len(
                                                                                        dataloader.dataset),
                                                                                    loss.item()))
                        losses_charts[phase]["loss"].append(loss.item())
                        losses_charts[phase]["step"].append(b * args.batch_size + (epoch) * len(dataloader.dataset))

                # we will only aggregate loss for validation on an epoch basis
                if phase == "val":
                    losses_charts[phase]["loss"].append(epoch_loss / len(dataloader.dataset))
                    losses_charts[phase]["step"].append(epoch * len(dataloaders["train"].dataset))
                    losses_charts[phase]["psnr"].append(psnr / len(dataloader.dataset))

    plt.figure(2)
    plt.plot(losses_charts["train"]["step"], losses_charts["train"]["loss"], color="blue")
    plt.plot(losses_charts["val"]["step"], losses_charts["val"]["loss"], color="Red")
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.savefig(os.path.join(out_dir, "loss_charts.png"))

    plt.figure(3)
    plt.plot(losses_charts["val"]["step"], losses_charts["val"]["psnr"], color="Red")
    plt.legend(['PSNR (db)'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('Test PSNR')
    plt.savefig(os.path.join(out_dir, "psnr.png"))
    plt.show()

    # at the end of training report the hyper parameters used and the best metrics achieved.


if __name__ == "__main__":
    parser = argparse.ArgumentParser("mnist classifier arg parser")
    parser.add_argument("--epochs", help="number of epochs to train", type=int, default=10)
    parser.add_argument("--batch_size", help="number of images in a batch", type=int, default=64)
    parser.add_argument("--learning_rate", help="learning rate", type=float, default=0.1)
    parser.add_argument("--momentum", help="momentum", type=float, default=0.5)
    parser.add_argument("--seed", help="random seed for reproducibility", type=int, default=24)
    parser.add_argument("--log_interval", help="interval for logging cool info", type=int, default=10)
    parser.add_argument("--output_dir", help="directory for storing outputs", type=str, default="./output/")
    args = parser.parse_args()

    train(args)
