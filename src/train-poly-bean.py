import torch

from os import makedirs
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from models import PolyNet
from components.datasets import PolyBean, AugmentBean
from components.transformers import ViT, Decoder


def load_datasets(batch_size):
    train = PolyBean(csv_file="datasets/poly-bean/train/polygon-bean-leaf2.csv",
                        transform=ToTensor())

    test = PolyBean(csv_file="datasets/poly-bean/test/polygon-bean-leaf2.csv",
                       transform=ToTensor())

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


if __name__ == "__main__":
    torch.manual_seed(0)

    name, args = PolyNet.parse_args()

    device = None

    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(device)
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    print("Using device: ", device, f"({device_name})")

    encoder = ViT(
        embed_dim=args.dim_embedding,
        n_blocks=args.blocks,
        n_heads=args.heads,
        img_shape=(3, 320, 320),
        patch_size=16
    )

    decoder = Decoder(
        embed_dim=args.dim_embedding,
        seq_len=200,
        n_blocks=args.blocks,
        n_heads=args.heads
    )

    model = PolyNet.Model(args.dim_embedding, encoder, decoder)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = MSELoss()

    print("Loading dataset to memory...")
    train_loader, test_loader = load_datasets(args.batch_size)
    print("Dataset loaded")

    mask = torch.tril(torch.ones(200, 200)).view(1, 200, 200).to(device)

    model = model.to(device)
    folder_name = "tetest"

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint), strict=args.strict)

    losses = PolyNet.train(args.epochs, model, mask, optimizer,
                           criterion, train_loader, test_loader, device)

    makedirs(f"./logs/{folder_name}/", exist_ok=True)
    log_file = open(f"./logs/{folder_name}/{name}.txt", "w")
    loss_str = "\n".join(losses)
    log_file.write(loss_str)

    makedirs(f"./checkpoints/{folder_name}/", exist_ok=True)
    path = f"./checkpoints/{folder_name}/{name}.pth"
    torch.save(model.state_dict(), path)

    idx_list = [0, 10, 23, 34, 48, 58, 70, 79, 89, 99]
    makedirs(f"./images/{folder_name}/", exist_ok=True)
    PolyNet.predict(model, f"./images/{folder_name}/{name}.png", test_loader, mask, idx_list, 199, device)
