import cv2
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange, tqdm
from torch.nn import MSELoss
from torch.optim import Adam
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import nn

from src.datasets import PolyMNIST
from src.transformers import ViT, Decoder

torch.manual_seed(0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(
    device) if torch.cuda.is_available() else "cpu"
print("Using device: ", device, f"({device_name})")


class Net(nn.Module):
    def __init__(self, embed_dim=16, seq_len=12, n_blocks=2, n_heads=2):
        super(Net, self).__init__()

        self.encoder = ViT(embed_dim=embed_dim,
                           n_blocks=n_blocks, n_heads=n_heads, img_shape=(3, 320, 320), patch_size=16)

        self.decoder = Decoder(embed_dim=embed_dim, seq_len=seq_len,
                               n_blocks=n_blocks, encoder_dim=embed_dim, n_heads=n_heads)

        self.mapper = nn.Linear(3, embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 3),
            nn.Sigmoid()
        )

    def forward(self, x, p, mask=None):
        x = self.encoder(x)
        p = self.mapper(p)
        x = self.decoder(p, x[:, 0, :], mask)
        x = self.mlp(x)

        return x


def load_datasets(batch_size):
    train = PolyMNIST(csv_file="poly-bean/train/polygon-bean-leaf.csv",
                      transform=ToTensor())

    test = PolyMNIST(csv_file="poly-bean/test/polygon-bean-leaf.csv",
                     transform=ToTensor())

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train(epochs, model, mask, optimizer, criterion, train_loader, test_loader):
    losses = []
    for epoch in trange(epochs, desc="Training"):
        model.train()

        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            image, polygon = batch

            image = image.to(device)
            polygon = polygon.to(device)

            pred = model(image, polygon, mask)
            loss = criterion(pred[:, :-1, :], polygon[:, 1:, :])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item() / len(train_loader)

        print(f"Epoch {epoch + 1}/{epochs} loss: {train_loss:.4f}")

        with torch.no_grad():
            total = 0
            test_loss = 0.0
            model.eval()
            for batch in tqdm(test_loader, desc="Testing"):
                x, p = batch
                x, p = x.to(device), p.to(device)

                pred = model(image, polygon, mask)
                loss = criterion(pred[:, :-1, :], polygon[:, 1:, :])

                test_loss += loss.detach().cpu().item() / len(test_loader)

                total += len(x)
            print(f"Test loss: {test_loss:.4f}")

        losses.append(
            f"Epoch {epoch + 1}/{epochs}\n\ttrain loss: {train_loss:.4f}\n\ttest loss:{test_loss:.4f}\n")
    return losses


def subplot(fig, rows, cols, pos, title, img_tensor, poly_tensor):
    image = img_tensor.permute(1, 2, 0).cpu().numpy() * 255

    # out = np.zeros((320, 320, 3))
    # out[:, :, 0] = image[:, :, 0]
    # out[:, :, 1] = image[:, :, 1]
    # out[:, :, 2] = image[:, :, 2]
    out = np.copy(image)

    poly_tensor = poly_tensor[1:]
    filter1 = poly_tensor[:, 0] < 1.0
    poly_tensor = poly_tensor[filter1]

    poly = poly_tensor.cpu().numpy()
    poly = np.delete(poly, 0, 1).reshape(-1, 1, 2) * 320
    poly = poly.astype(np.int32)
    x = poly[:, 0, 0]
    y = poly[:, 0, 1]

    # out = cv2.polylines(out, [poly], True, (100, 100, 255), 1)
    # out[y, x] = [255.0, 0.0, 0.0]

    fig.add_subplot(rows, cols, pos)
    plt.imshow(out.astype(np.uint8))
    # plt.scatter(x, y, c='red', markersize=1)
    plt.plot(x, y)
    plt.plot(x, y, marker=".", markersize=2, c='red', linestyle='None')
    plt.axis('off')
    plt.title(title)


def predict(test_loader, name):
    images, poly = next(iter(test_loader))

    fig = plt.figure(figsize=(20, 4))
    idx_range = 10

    for idx in range(idx_range):
        with torch.no_grad():
            model.eval()
            inputs = torch.zeros(1, 1, 3).to(device)
            image = images[idx].reshape(1, 3, 320, 320).to(device)

            for i in range(199):
                pred = model(image, inputs, mask[:, :i + 1, :i + 1])

                if (pred[0, -1, 0] > 0.95):
                    break

                inputs = torch.cat(
                    (inputs, pred[:, -1, :].view(1, 1, -1)), dim=1)

        pos_gt = idx + 1
        pos_pred = idx + 1 + idx_range
        subplot(fig, 2, idx_range, pos_gt, "Ground truth",
                images[idx], poly[idx])
        subplot(fig, 2, idx_range, pos_pred, "Prediction",
                images[idx], inputs[0])
    fig.savefig(f"images/{name}.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=12)
    parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-d", "--dim_embedding", type=int, default=256)
    parser.add_argument("-b", "--blocks", type=int, default=2)
    parser.add_argument("-bs", "--batch_size", type=int, default=20)
    parser.add_argument("-hd", "--heads", type=int, default=1)
    parser.add_argument("-sq", "--seq_len", type=int, default=12)
    parser.add_argument("-cp", "--checkpoint", type=str, default=None)
    parser.add_argument(
        "-s", "--strict", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    batch, embed_dim, seq_len, n_blocks, n_heads = args.batch_size, args.dim_embedding, args.seq_len, args.blocks, args.heads

    model = Net(embed_dim, seq_len, n_blocks, n_heads).to(device)
    lr = args.learning_rate
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = MSELoss()

    epochs = args.epochs

    name = f"{epochs}eps_{lr}lr_{batch}bs_{embed_dim}dim_{seq_len}sq_{n_blocks}blk_{n_heads}hds"
    print(name)

    train_loader, test_loader = load_datasets(batch)
    mask = torch.tril(torch.ones(200, 200)).view(1, 200, 200).to(device)

    losses = train(epochs, model, mask, optimizer,
                   criterion, train_loader, test_loader)

    log_file = open(f"logs/{name}.txt", "w")
    loss_str = "\n".join(losses)
    log_file.write(loss_str)

    path = f"checkpoints/{name}.pth"
    torch.save(model.state_dict(), path)

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint), strict=args.strict)

    predict(test_loader, name)
