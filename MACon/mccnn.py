import torch
import torch.nn as nn
import torchvision.models as models
resnet50 = models.resnet50(pretrained=False)

num_ftrs = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_ftrs, 2)

#pic 1 pic+0 为正样本  pic+[-d,-1],[1,d]为负样本
#pic 2 测试