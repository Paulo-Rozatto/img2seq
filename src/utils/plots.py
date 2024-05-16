import numpy as np
import matplotlib.pyplot as plt


def subplot(fig, rows, cols, pos, title, img_tensor, poly_tensor, cmap=None):
    image = img_tensor.permute(1, 2, 0).cpu().numpy() * 255

    out = np.copy(image)

    poly_tensor = poly_tensor[1:]
    filter_1_out = poly_tensor[:, 0] != 0.0
    poly_tensor = poly_tensor[filter_1_out]

    poly = poly_tensor.cpu().numpy()
    poly = np.delete(poly, 0, 1).reshape(-1, 1, 2) * image.shape[1]
    poly = poly.astype(np.int32)
    x = poly[:, 0, 0]
    y = poly[:, 0, 1]
    x = np.append(x, x[0])
    y = np.append(y, y[0])

    fig.add_subplot(rows, cols, pos)
    plt.imshow(out.astype(np.uint8), cmap=cmap)
    plt.plot(x, y)
    plt.plot(x, y, marker=".", markersize=2, c='red', linestyle='None')
    plt.axis('off')
    plt.title(title)
