import torch
import torchvision
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
from ipdb import set_trace as pdb
import numpy as np
import cv2


class MobileNet:
    def __init__(self):
        self.model = torchvision.models.mobilenet_v2(pretrained=True)
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.img_to_tensor = transforms.ToTensor()
        self.RESIZE = 512

    def get_feature(self, img):
        img = self.preprocess_img(img)
        feature = self.model(img)
        feature = self.tensor_to_numpy(feature)

        return feature

    def preprocess_img(self, img):
        h, w, c = img.shape
        if max(h, w) > self.RESIZE:
            resize_factor = self.RESIZE / h
            img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
        
        img = np.transpose(img, (2, 0, 1))
        img = self.img_to_tensor(img)[:3].unsqueeze(0)
        img = img.to(self.device)
        return img

    def tensor_to_numpy(self, x):
        x = x.transpose(0,1)
        x = x.detach().cpu()
        x = x.reshape(-1).numpy()
        return x


def make_tsne(fts):
    from sklearn.manifold import TSNE
    from matplotlib import pyplot as plt
    tsne = TSNE(n_components=2, init='pca')
    x = tsne.fit_transform(fts)
    
    plt.figure()
    plt.scatter(x[:,0], x[:,1])
    plt.xticks([])
    plt.yticks([])
    plt.savefig('test.png')

if __name__ == '__main__':
    model = MobileNet()
    img = cv2.imread('../../data/general/image/IMG_ (31).jpg')      # (h, w, chennel)
    fts = model.get_feature(img)
    # make_tsne([fts, fts*2])
    
