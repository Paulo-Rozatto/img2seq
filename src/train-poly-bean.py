import torch

from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from models import PolyNet
from components.datasets import PolyBean
from components.transformers import ViT, Decoder


def load_datasets(batch_size):
    train = PolyBean(csv_file="../datasets/poly-bean/train/polygon-bean-leaf.csv",
                     transform=ToTensor())

    test = PolyBean(csv_file="../datasets/poly-bean/test/polygon-bean-leaf.csv",
                    transform=ToTensor())

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


if __name__ == "__main__":
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
        seq_len=args.seq_len,
        n_blocks=args.blocks,
        n_heads=args.heads
    )

    model = PolyNet.Model(args.dim_embedding, encoder, decoder)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = MSELoss()

    train_loader, test_loader = load_datasets(args.batch_size)
    mask = torch.tril(torch.ones(200, 200)).view(1, 200, 200).to(device)

    model = model.to(device)
    name = "bean_" + name

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint), strict=args.strict)

    losses = PolyNet.train(args.epochs, model, mask, optimizer,
                           criterion, train_loader, test_loader, device)

    log_file = open(f"../logs/{name}.txt", "w")
    loss_str = "\n".join(losses)
    log_file.write(loss_str)

    path = f"../checkpoints/{name}.pth"
    torch.save(model.state_dict(), path)

    PolyNet.predict(model, name, test_loader, mask, 10, 199, device)