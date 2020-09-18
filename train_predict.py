import torch
from torch.utils.data import DataLoader
import unet3d
import torch.nn as nn
from dataset import MyDataset

img_path = 'D:/github_code/ModelsGenesis/data/train/imgs'
# img_path = '/home/user/panghaowen/github_code/ModelsGenesis/data/train/imgs'
mask_path = 'D:/github_code/ModelsGenesis/data/train/masks'
# mask_path = '/home/user/panghaowen/github_code/ModelsGenesis/data/train/masks'

# prepare your own data
train_Dataset = MyDataset(img_path, mask_path)
train_loader = DataLoader(train_Dataset, batch_size=1, shuffle=True)

# prepare the 3D model
model = unet3d.UNet3D()

# Load pre-trained weights
# weight_dir = 'pretrained_weights/Genesis_Chest_CT.pt'
# checkpoint = torch.load(weight_dir)
# state_dict = checkpoint['state_dict']
# unParalled_state_dict = {}
# for key in state_dict.keys():
#     unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
# model.load_state_dict(unParalled_state_dict)

model.to('cuda')
# model = nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
criterion = torch.nn.CrossEntropyLoss()  # 损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0, nesterov=False)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.9 * 0.8), gamma=0.5)

# train the model

for epoch in range(100):
    # scheduler.step(epoch)
    model.train()
    for x, y in train_loader:
        x, y = x.float().to('cuda'), y.float().to('cuda')
        print(x.size())
        print(y.size())
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
