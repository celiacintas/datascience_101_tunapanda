import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from PIL import Image
import PIL.ImageOps
import warnings
warnings.filterwarnings("ignore")

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

def proj(X, ax, ax2d):
        """ From a 3D point in axes ax1, 
            calculate position in 2D in ax2 """
        x,y,z = X
        x2, y2, _ = proj3d.proj_transform(x,y,z, ax.get_proj())
        tr = ax.transData.transform((x2, y2))
        return ax2d.transData.inverted().transform(tr)


def plot_tsne_3D(X_tsne, images, labels, azim=120, distance=70000):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection=Axes3D.name)
    ax2d = fig.add_subplot(111,frame_on=False) 
    ax2d.axis("off")
    ax.view_init(elev=30., azim=azim)
    for i in range(X_tsne.shape[0]):
            ax.scatter(X_tsne[i, 0], X_tsne[i, 1], X_tsne[i, 2],  c=plt.cm.Set1(labels[i] / 10.), s=100)
    if hasattr(offsetbox, 'AnnotationBbox'):
        shown_images = np.array([[1., 1., 1.]])
        for i in range(images.shape[0]):           
            dist = np.sum((X_tsne[i] - shown_images) ** 2, 1)

            if np.min(dist) < distance:
                # don't show points that are too close
                continue

            shown_images = np.r_[shown_images, [X_tsne[i]]]
            image =  Image.fromarray(images[i].reshape(28, 28), 'L')
            #inverted_image = PIL.ImageOps.invert(image)
            
            image.thumbnail((40, 40), Image.ANTIALIAS)
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(image),
                proj(X_tsne[i], ax, ax2d))
            ax2d.add_artist(imagebox)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title('t-SNE over the Fashion-MNIST')
    plt.savefig("/tmp/fashion/movie%d.png" % azim)
    #plt.show()

