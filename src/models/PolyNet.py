import torch
import argparse
import matplotlib.pyplot as plt

from torch import nn
from tqdm import trange, tqdm

from components.transformers import ViT, Decoder
from utils.plots import subplot
from utils import metrics


class Model(nn.Module):
    def __init__(
        self,
        embed_dim=16,
        encoder=ViT,
        decoder=Decoder,
    ):
        super(Model, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.mapper = nn.Linear(3, embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.Sigmoid(),
            nn.Linear(4 * embed_dim, 3),
            nn.Sigmoid()
        )

    def forward(self, x, p, mask=None, pad_mask=None):
        x = self.encoder(x)
        p = self.mapper(p)
        x = self.decoder(p, x[:, 0, :], mask, pad_mask)
        x = self.mlp(x)

        return x


def train(epochs, model, mask, optimizer, criterion, train_loader, test_loader, device="cpu"):
    losses = []
    for epoch in trange(epochs, desc="Training"):
        model.train()

        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            image, polygon, pad_mask = batch

            image = image.to(device)
            polygon = polygon.to(device)
            pad_mask = pad_mask.to(device)

            pred = model(image, polygon, mask, pad_mask)

            # loss = criterion(pred[0, :-1, :][pad_mask[0]], polygon[0, 1:, :][pad_mask[0]])
            # for i in range(1, len(polygon)):
            #     loss += criterion(pred[i, :-1, :][pad_mask[i]], polygon[i, 1:, :][pad_mask[i]])
            # r = (pred[:, :, 0] > 0.9).nonzero()
            loss = criterion(pred[:, :-1, :], polygon[:, 1:, :])

            # loss = criterion(pred[:, :-1, :][pad_mask], polygon[:, 1:, :][pad_mask])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item() / len(train_loader)

        print(f"Epoch {epoch + 1}/{epochs} loss: {train_loss:.4f}")

        with torch.no_grad():
            total = 0
            test_loss = 0.0
            miou = 0.0
            model.eval()

            for batch in tqdm(test_loader, desc="Testing"):
                image, polygon, pad_mask = batch
                image, polygon, pad_mask = image.to(
                    device), polygon.to(device), pad_mask.to(device)

                pred = model(image, polygon, mask, pad_mask)

                # loss = criterion(pred[0, :-1, :][pad_mask[0]], polygon[0, 1:, :][pad_mask[0]])
                miou += metrics.iou(pred[0, :-1, :][pad_mask[0]],
                                    polygon[0, 1:, :][pad_mask[0]])
                for i in range(1, len(polygon)):
                    # loss += criterion(pred[i, :-1, :][pad_mask[i]], polygon[i, 1:, :][pad_mask[i]])
                    miou += metrics.iou(pred[i, :-1, :], polygon[i, 1:, :])

                loss = criterion(pred[:, :-1, :], polygon[:, 1:, :])

                test_loss += loss.detach().cpu().item() / len(test_loader)

                total += len(image)
            print(f"Test loss: {test_loss:.4f}")
            print(f"miou: {(miou / len(test_loader.dataset)):.4f}")

        losses.append(
            f"Epoch {epoch + 1}/{epochs}\n\ttrain loss: {train_loss:.4f}\n\ttest loss:{test_loss:.4f}\n")
    return losses


def predict(model, name, test_loader, mask, idx_list = [0, 5, 10], max_pred = 11, device="cpu"):
    length = len(idx_list)

    fig = plt.figure(figsize=(20, 4))

    it = iter(test_loader)
    miou = 0
    for j, idx in zip(range(length), idx_list):
        with torch.no_grad():
            model.eval()
            image, polygon, pad_mask = test_loader.dataset.__getitem__(idx)
            inputs = torch.zeros(1, 1, 3).to(device)
            image_device = image.unsqueeze(0).to(device)

            for i in range(max_pred):
                pred = model(image_device, inputs, mask[:, :i + 1, :i + 1])

                if (pred[0, -1, 0] > 0.95):
                    break

                inputs = torch.cat(
                    (inputs, pred[:, -1, :].view(1, 1, -1)), dim=1)

        pos_gt = j + 1
        pos_pred = pos_gt + length
        miou += metrics.iou(polygon[1:][pad_mask], inputs[0, 1:])

        subplot(fig, 2, length, pos_gt,
                "Ground truth", image, polygon, cmap='gray')
        subplot(fig, 2, length, pos_pred,
                "Prediction", image, inputs[0], cmap='gray')

    print(f"miou: {(miou / length):.4f}")
    fig.savefig(f"./images/{name}.png")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=4)
    parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-d", "--dim_embedding", type=int, default=16)
    parser.add_argument("-b", "--blocks", type=int, default=2)
    parser.add_argument("-bs", "--batch_size", type=int, default=30)
    parser.add_argument("-hd", "--heads", type=int, default=1)
    parser.add_argument("-sq", "--seq_len", type=int, default=20)
    parser.add_argument("-cp", "--checkpoint", type=str, default=None)
    parser.add_argument(
        "-s", "--strict", action=argparse.BooleanOptionalAction, default=False)

    a = parser.parse_args()
    name = f"{a.epochs}eps_{a.learning_rate}lr_{a.batch_size}bs_{a.dim_embedding}d_{a.seq_len}sq_{a.blocks}b_{a.heads}hd"

    return name, a
