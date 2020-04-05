import torch
import torchvision
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchsummary import summary
from ipdb import set_trace as pdb


# model = torchvision.models.vgg19_bn(pretrained=False, progress=True)
# ================================================================
# Total params: 143,678,248
# Trainable params: 143,678,248
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 352.00
# Params size (MB): 548.09
# Estimated Total Size (MB): 900.66
# ----------------------------------------------------------------
# Top1 error: 27.62
# Top5 error: 9.12
# ================================================================


# model = torchvision.models.resnet152(pretrained=False, progress=True)
# ================================================================
# Total params: 60,192,808
# Trainable params: 60,192,808
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 606.59
# Params size (MB): 229.62
# Estimated Total Size (MB): 836.78
# ----------------------------------------------------------------
# Top1 error: 21.69
# Top5 error: 5.94
# ================================================================


# model = torchvision.models.inception_v3(pretrained=False, progress=True)
# === input image shape should shange to (3, 299, 299) ===
# ================================================================
# Total params: 60,192,808
# Trainable params: 60,192,808
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 606.59
# Params size (MB): 229.62
# Estimated Total Size (MB): 836.78
# ----------------------------------------------------------------
# Top1 error: 21.69
# Top5 error: 5.94
# ================================================================

model = torchvision.models.mobilenet_v2(pretrained=False, progress=True)
# ================================================================
# Total params: 3,504,872
# Trainable params: 3,504,872
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 152.87
# Params size (MB): 13.37
# Estimated Total Size (MB): 166.81
# ----------------------------------------------------------------
# Top1 error: 28.12
# Top5 error: 9.71
# ================================================================

# model = torchvision.models.resnext101_32x8d(pretrained=False, progress=True)
# ================================================================
# Total params: 88,791,336
# Trainable params: 88,791,336
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 772.54
# Params size (MB): 338.71
# Estimated Total Size (MB): 1111.83
# ----------------------------------------------------------------
# Top1 error: 20.69
# Top5 error: 5.47
# ================================================================

# model = torchvision.models.wide_resnet101_2(pretrained=False, progress=True)
# ================================================================
# Total params: 126,886,696
# Trainable params: 126,886,696
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 544.00
# Params size (MB): 484.03
# Estimated Total Size (MB): 1028.61
# ----------------------------------------------------------------
# Top1 error: 21.16
# Top5 error: 5.72
# ================================================================


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
summary(model, (3, 224, 224))

pdb()