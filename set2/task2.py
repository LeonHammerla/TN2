import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn . decomposition import PCA
from sklearn . manifold import TSNE


def visualize(data: np.ndarray,
              labels: np.ndarray,
              n: int):
    data = data[:n]
    labels = labels[:n]
    # 2D
    # datacreation
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data)
    tsne = TSNE(n_components=2)
    tsne_data = tsne.fit_transform(data)
    # PLOTS:
    fig = plt.scatter(*np.transpose(pca_data), c=labels)
    plt.legend(*fig.legend_elements(),
                    loc="lower right", title="Classes")
    plt.show()

    fig = plt.scatter(*np.transpose(tsne_data), c=labels)
    plt.legend(*fig.legend_elements(),
               loc="lower right", title="Classes")
    plt.show()


if __name__ == "__main__":
    # Get images (70k, 748)
    mnist = fetch_openml(name="mnist_784", as_frame = False, parser ="auto")
    data = mnist.data
    # Get labels (70k,)
    labels = np.array([int(k) for k in mnist.target])

    visualize(data, labels, 2000)

