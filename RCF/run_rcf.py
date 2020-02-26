import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
import scipy.io as sio
from models import RCF
from torch.utils.data import Dataset, DataLoader
from os.path import join, split, isdir, isfile, splitext, split
from ipdb import set_trace as pdb

img_dir = '../input/image/'
save_dir = '../input/rcf_edge_image/'
test_list_dir = 'test_all.lst'
vggmodel_dir='pretrained/vgg16convs.mat'
checkpoint_dir = 'pretrained/RCFcheckpoint_epoch12.pth'


class TestLoader(Dataset):
    def __init__(self, list_dir, root=img_dir):
        self.root = root
        with open(list_dir, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, index):
        img_file = self.filelist[index].rstrip()
        img = cv2.imread(join(self.root, img_file)).astype(np.float32)      # (1205, 900, 3)
        return img


def load_vgg16pretrain(model, vggmodel=vggmodel_dir):
    vgg16 = sio.loadmat(vggmodel)
    torch_params =  model.state_dict()
    for k in vgg16.keys():
        name_par = k.split('-')
        size = len(name_par)
        if size  == 2:
            name_space = name_par[0] + '.' + name_par[1]
            data = np.squeeze(vgg16[k])
            torch_params[name_space] = torch.from_numpy(data)
    model.load_state_dict(torch_params)


def main():
    test_dataset = TestLoader(test_list_dir)
    test_loader = DataLoader(test_dataset, shuffle=False)
    with open(test_list_dir, 'r') as f:
        test_list = f.readlines()
    test_list = [split(i.rstrip())[1] for i in test_list]
    assert len(test_list) == len(test_loader), "%d vs %d" % (len(test_list), len(test_loader))

    # model
    model = RCF()
    model.cuda()
    load_vgg16pretrain(model)
    if isfile(checkpoint_dir): 
        print("=> loading checkpoint '{}'".format(checkpoint_dir))
        checkpoint = torch.load(checkpoint_dir)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(checkpoint_dir))

    model.eval()
    for idx, image in enumerate(test_loader):
        image = image[0].numpy()
        filename = splitext(test_list[idx])[0]
        resize_factor = 1.0
        while True:
            try:
                resi_image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)
                resi_image = np.transpose(resi_image, (2, 0, 1))                                  # (3, 736, 550)
                resi_image = torch.FloatTensor(resi_image).unsqueeze(0)
                resi_image = resi_image.cuda()
                results = model(resi_image)
                break
            except Exception as e:
                resize_factor *= 0.95
                print(filename)
        result = torch.squeeze(results[-1].detach()).cpu().numpy()
        result = cv2.resize(result, (0, 0), fx=1/resize_factor, fy=1/resize_factor)
        result = Image.fromarray((result * 255).astype(np.uint8))
        result.save(join(save_dir, "%s_rcf.png" % filename))
        print("Running test [%d/%d]" % (idx + 1, len(test_loader)))


if __name__ == '__main__':
    main()
