import torch
import numpy
import PIL
import PIL.Image
import cv2
from os import listdir
from .model import Network
from tqdm import tqdm
from ipdb import set_trace as pdb

pretrained_dir = 'HED/pretrained/network-bsds500.pytorch'
torch.set_grad_enabled(False)
torch.backends.cudnn.enabled = True


def make_single_hed(img_path, img_dir, output_dir):
    moduleNetwork = Network(pretrained_dir).cuda().eval()

    img_name = '.'.join(img_path.split('.')[:-1])   # remove .jpg
    arguments_strIn = img_dir + img_path
    arguments_strOut = output_dir + img_name + '_hed.png' 

    img = cv2.imread(arguments_strIn)
    resize_factor = 1.0
    while True:
        try:
            resi_img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
            resi_img = resi_img.transpose(2, 0, 1)
            resi_img = resi_img.astype(numpy.float32) * (1.0 / 255.0)
            tensorInput = torch.FloatTensor(resi_img)
            tensorOutput = moduleNetwork(tensorInput.cuda().view(1, 3, tensorInput.size(1), tensorInput.size(2)))[0, :, :, :].cpu()
            break
        except:
            resize_factor *= 0.95
    tensorOutput = (tensorOutput.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)
    tensorOutput = cv2.resize(tensorOutput, (0, 0), fx=1/resize_factor, fy=1/resize_factor)
    tensorOutput = PIL.Image.fromarray(tensorOutput).save(arguments_strOut)

