import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
import os

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET
from model import U2NETP

import random
import numpy as np
from skimage import transform

class RandomHorizontalFlipDict(object):
    """Horizontally flip the given image and label randomly with a given probability."""
    def __init__(self, p=0.25):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample['image'] = np.fliplr(sample['image']).copy()
            sample['label'] = np.fliplr(sample['label']).copy()
            
        # Return the entire sample dict so 'imidx' is preserved
        return sample


class RandomRotationDict(object):
    """Rotate the image and label by a random angle using skimage."""
    def __init__(self, degrees=(0, 10)):
        self.degrees = degrees

    def __call__(self, sample):
        angle = random.uniform(self.degrees[0], self.degrees[1])
        
        # order=1 means Bilinear interpolation
        sample['image'] = transform.rotate(
            sample['image'], angle, order=1, mode='constant', cval=0, preserve_range=True
        )
        
        # order=0 means Nearest-Neighbor (CRITICAL so class IDs stay exact integers)
        sample['label'] = transform.rotate(
            sample['label'], angle, order=0, mode='constant', cval=0, preserve_range=True
        )
        
        # Return the entire sample dict so 'imidx' is preserved
        return sample

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)

# ------- 1. define loss function --------

def dice_loss(preds, targets, num_classes, smooth=1e-5):
    """
    preds: tensor of shape [B, C, H, W] containing raw logits
    targets: tensor of shape [B, 1, H, W] or [B, H, W] containing class indices
    """
    # 1. ADD THIS LINE: Convert raw logits to probabilities
    preds = F.softmax(preds, dim=1)
    
    # Remove channel dimension if it exists
    if targets.dim() == 4:
        targets = targets.squeeze(1)
        
    # Convert to one-hot encoding [B, C, H, W]
    targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    # Calculate intersection and union (now mathematically valid)
    intersection = (preds * targets_one_hot).sum(dim=(2, 3)) 
    union = preds.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
    
    # Dice coefficient
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return 1.0 - dice.mean()

def multi_dice_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v, num_classes):
    loss0 = dice_loss(d0, labels_v, num_classes)
    loss1 = dice_loss(d1, labels_v, num_classes)
    loss2 = dice_loss(d2, labels_v, num_classes)
    loss3 = dice_loss(d3, labels_v, num_classes)
    loss4 = dice_loss(d4, labels_v, num_classes)
    loss5 = dice_loss(d5, labels_v, num_classes)
    loss6 = dice_loss(d6, labels_v, num_classes)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(
        loss0.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.item()))

    return loss0, loss

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

	loss0 = bce_loss(d0,labels_v)
	loss1 = bce_loss(d1,labels_v)
	loss2 = bce_loss(d2,labels_v)
	loss3 = bce_loss(d3,labels_v)
	loss4 = bce_loss(d4,labels_v)
	loss5 = bce_loss(d5,labels_v)
	loss6 = bce_loss(d6,labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

	return loss0, loss


# ------- 2. set the directory of training dataset --------
NUM_CLASSES = 5  # <-- CHANGE THIS TO YOUR ACTUAL NUMBER OF CLASSES (including background)

model_name = 'u2net'
data_dir = '/home/wot-amd/Projects/Gotilo-container/container-inspection/damage_detection/u2net_multiclass_dataset_20260702_single/'
tra_image_dir = 'train/images/'
tra_label_dir = 'train/masks/'

image_ext = '.jpg'
label_ext = '.png'

model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep )
os.makedirs(model_dir, exist_ok=True)

epoch_num = 500
batch_size_train = 12
batch_size_val = 1
train_num = 0
val_num = 0

tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

tra_lbl_name_list = []
for img_path in tra_img_name_list:
	img_name = img_path.split(os.sep)[-1]

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

train_num = len(tra_img_name_list)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
            RescaleT(320),
            RandomRotationDict(degrees=(0, 10)),   
            RandomCrop(288),
            RandomHorizontalFlipDict(p=0.25),      
            ToTensorLab(flag=0)
        ])
    )
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)

# ------- 3. define model --------
# define the net
if(model_name=='u2net'):
    net = U2NET(in_ch=3, out_ch=NUM_CLASSES)
elif(model_name=='u2netp'):
    net = U2NETP(in_ch=3, out_ch=NUM_CLASSES)

# ---> ADDED: Load pre-trained weights for transfer learning <---
pretrained_weight_path = os.path.join(os.getcwd(), 'saved_models', model_name + "_pretrained", model_name + '.pth')

if os.path.exists(pretrained_weight_path):
    print(f"--- Loading pretrained weights from {pretrained_weight_path} ---")
    
    # 1. Load the pre-trained state dictionary
    pretrained_dict = torch.load(pretrained_weight_path)
    
    # 2. Get the state dictionary of your current (multi-class) model
    model_dict = net.state_dict()
    
    # 3. Filter out weights where the shapes don't match (i.e., the final side and outconv layers)
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    
    # 4. Update your model's state dictionary with the matching weights
    model_dict.update(filtered_dict)
    
    # 5. Load the updated dictionary back into the network
    net.load_state_dict(model_dict)
    
    print(f"Successfully loaded {len(filtered_dict)} out of {len(pretrained_dict)} layers. Skipped final output layers.")
else:
    print(f"--- WARNING: Pretrained weights not found at {pretrained_weight_path}. Training from scratch! ---")

if torch.cuda.is_available():
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.AdamW(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
save_frq = 500

for epoch in range(0, epoch_num):
    net.train()

    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels = data['image'], data['label']

        inputs = inputs.type(torch.FloatTensor)
        # CRITICAL FIX: Labels must be integers (class indices) for multi-class
        labels = labels.type(torch.LongTensor) 

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v = inputs.cuda()
            labels_v = labels.cuda()
        else:
            inputs_v = inputs
            labels_v = labels
        
        # ---> ADD THIS TEMPORARY DEBUG LINE <---
        print("Unique pixel values in mask:", torch.unique(labels_v))
        # import sys; sys.exit() # Stop execution here temporarily

        optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = multi_dice_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v, num_classes=NUM_CLASSES)

        loss.backward()
        
        # ADD THIS LINE:
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # NOTE: If you want to visualize or save the predictions during training, 
        # this is where you use argmax to convert probabilities back to class maps:
        # predicted_mask = torch.argmax(d0, dim=1) # Shape: [B, H, W]

        running_loss += loss.item()
        running_tar_loss += loss2.item()

        del d0, d1, d2, d3, d4, d5, d6, loss2, loss

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
        epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

        if ite_num % save_frq == 0:
            torch.save(net.state_dict(), model_dir + model_name+"_dice_itr_%d_train_%3f_tar_%3f.pth" % (
                ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()
            ite_num4val = 0

