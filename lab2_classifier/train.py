"""
Created by Edward Li at 2/9/21
"""
import argparse
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from torch.utils.data import DataLoader

from data import MnistDataset
from model import CNNClassifier


def eval(model, out_dir, data, labels, epoch, device, fig, axs):
    data, target = data.to(device), labels.to(device)
    with torch.no_grad():
        output = model(data)
        # take the first 16 images
        for ind, ax in enumerate(axs):
            ax.imshow(data[ind][0].detach().cpu().numpy(), cmap="gray", interpolation="none")
            ax.set_title("Label: {} Pred: {}".format(target[ind].detach().cpu().numpy(),
                                                     output.data.max(1, keepdim=True)[1][ind].item()), fontsize=20)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')

        plt.savefig(os.path.join(out_dir, "eval_epoch_{}.png".format(epoch)))


def plot_results(losses_charts, out_dir):
    # plot our loss and accuracy charts
    plt.figure(2)
    plt.plot(losses_charts["train"]["step"], losses_charts["train"]["loss"], color="blue")
    plt.plot(losses_charts["val"]["step"], losses_charts["val"]["loss"], color="Red")
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.savefig(os.path.join(out_dir, "loss_charts.png"))

    plt.figure(3)
    # plt accuracy/precision/recall/f1 score 
    losses_charts["val"]["accuracy"] = []
    losses_charts["val"]["precision"] = []
    losses_charts["val"]["recall"] = []
    losses_charts["val"]["f1"] = []
    for stat in losses_charts["val"]["stats"]:
        losses_charts["val"]["accuracy"].append(stat["accuracy"])
        losses_charts["val"]["precision"].append(stat['weighted avg']["precision"])
        losses_charts["val"]["recall"].append(stat['weighted avg']["recall"])
        losses_charts["val"]["f1"].append(stat['weighted avg']["f1-score"])

    plt.plot(losses_charts["val"]["step"], losses_charts["val"]["accuracy"], color="Red")
    plt.plot(losses_charts["val"]["step"], losses_charts["val"]["precision"], color="Green")
    plt.plot(losses_charts["val"]["step"], losses_charts["val"]["recall"], color="Blue")
    plt.plot(losses_charts["val"]["step"], losses_charts["val"]["f1"], color="Cyan")

    best_ind = np.argmax(losses_charts["val"]["accuracy"])
    plt.plot(losses_charts["val"]["step"][best_ind], losses_charts["val"]["accuracy"][best_ind], 'b*')
    plt.text(losses_charts["val"]["step"][best_ind], losses_charts["val"]["accuracy"][best_ind],
             'Best Accuracy: {}'.format(losses_charts["val"]["accuracy"][best_ind]), horizontalalignment='right')
    plt.legend(['Test Accuracy', 'Test Precision', 'Test recall', 'Test F1 score'], loc='lower right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('Test accuracy')
    plt.savefig(os.path.join(out_dir, "accuracy.png"))

    # for each class plot a figure for precision recall f1 score
    for cls in range(10):
        cls_str = str(cls)
        precision = []
        recall = []
        f1_score = []
        for ind, stat in enumerate(losses_charts["val"]["stats"]):
            precision.append(stat[cls_str]["precision"])
            recall.append(stat[cls_str]["recall"])
            f1_score.append(stat[cls_str]["f1-score"])
        plt.figure(cls + 4)
        plt.plot(losses_charts["val"]["step"], precision, color="Green")
        plt.plot(losses_charts["val"]["step"], recall, color="Blue")
        plt.plot(losses_charts["val"]["step"], f1_score, color="Cyan")
        best_ind = np.argmax(f1_score)
        plt.plot(losses_charts["val"]["step"][best_ind], f1_score[best_ind], 'b*')
        plt.text(losses_charts["val"]["step"][best_ind], f1_score[best_ind],
                 'Best f1-score: {}'.format(f1_score[best_ind]), horizontalalignment='right')
        plt.legend(['Test Precision', 'Test recall', 'Test F1 score'], loc='lower right')
        plt.xlabel('number of training examples seen')
        plt.ylabel('Precision')
        plt.title("{} class statistics".format(cls_str))
        plt.savefig(os.path.join(out_dir, "digit_{}_class_stats.png".format(cls_str)))


def train(args):
    # seed everything for reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    # prefer use Gpu for everything
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create our model
    model = CNNClassifier().to(device)
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
                          "step": [],
                          "stats": []},
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
                eval(model, out_dir, example_data, example_labels, epoch, device, fig, axs)
                # save a checkpoint
                torch.save(model.state_dict(), os.path.join(out_dir, "epoch_{}_ckpt.pt".format(epoch)))

            back_prop = phase == "train"
            epoch_loss = 0
            # in addition to accuracy, implement and calculate per class f1 score/precision/recall/accuarcy.
            print(phase)

            pred_epoch = []
            targ_epoch = []
            with torch.set_grad_enabled(back_prop):
                for b, (data, target) in enumerate(dataloader):
                    data, target = data.to(device), target.to(device)

                    if back_prop:
                        # clear gradients
                        optimizer.zero_grad()
                    output = model(data)
                    # use negative log likelinhood loss
                    loss = F.nll_loss(output, target)
                    if back_prop:
                        # get gradients
                        loss.backward()
                        # update weights
                        optimizer.step()
                    else:
                        # add logic here for the per class f1/precision/recall/accuracy, etc.
                        pred = output.data.max(1, keepdim=True)[1].view(-1)
                        targ = target.data.view_as(pred).detach().cpu().numpy()
                        pred = pred.detach().cpu().numpy()
                        pred_epoch.extend(pred)
                        targ_epoch.extend(targ)

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

                if phase == "val":
                    losses_charts[phase]["stats"].append(
                        metrics.classification_report(targ_epoch, pred_epoch, output_dict=True))

                    # we will only aggregate loss for validation on an epoch basis
                    losses_charts[phase]["loss"].append(epoch_loss / len(dataloader.dataset))
                    losses_charts[phase]["step"].append(epoch * len(dataloaders["train"].dataset))

    plot_results(losses_charts, out_dir)
    # plt.show()

    # add any figures as needed.


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
