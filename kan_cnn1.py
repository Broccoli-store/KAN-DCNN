import torch
import torch.nn as nn
import torch.nn.functional as F

from efficient_kan import KAN
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


setup_seed(123456)

incoming_data = np.load('./incoming_data.npy')
incoming_data = incoming_data.astype('float32')
flowfield_data = np.load('./unet_data_in0.npy')
flowfield_data = flowfield_data.astype('float32')


class MyDataset(Dataset):
    def __init__(self, incoming_data, flowfield_data):
        self.len = len(incoming_data)
        self.x_data = incoming_data
        self.y_data = flowfield_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


mydataset = MyDataset(incoming_data, flowfield_data)

np.random.seed(42)
test_ratio = 0.2  # 测试集占比
shuffled_indices = list(range(0, mydataset.len))  # 生成和原数据等长的无序索引
train_set_size = int(mydataset.len - mydataset.len * test_ratio)
train_indices = shuffled_indices[:train_set_size]
test_indices = shuffled_indices[train_set_size:]


class TrainDataset(Dataset):
    def __init__(self, incoming_data, flowfield_data):
        self.len = int(len(incoming_data))
        self.x_data = incoming_data[shuffled_indices]
        self.y_data = flowfield_data[shuffled_indices]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


train_set = TrainDataset(incoming_data, flowfield_data)


class TestDataset(Dataset):
    def __init__(self, incoming_data, flowfield_data):
        self.len = int(len(incoming_data) * test_ratio)
        self.x_data = incoming_data[test_indices]
        self.y_data = flowfield_data[test_indices]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


test_set = TestDataset(incoming_data, flowfield_data)

train_data = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
test_data = DataLoader(dataset=test_set, batch_size=1, shuffle=False)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = KAN([2, 32 , 256])
        self.l2 = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU())
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU())
        self.l4 = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU())
        self.l5 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU())
        self.l6 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU())
        self.l7 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU())

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.l1(x)
        x = torch.reshape(x, (-1, 16, 4, 4))
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        return x


# 训练神经网络
Epoch = 5000
learning_rate = 0.001
model = Model()
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10000, 0.5)
criterion = torch.nn.MSELoss()
train_losses = []
train_losses_all = []
train_loss_all = 0
test_losses = []
test_losses_all = []
test_loss_all = 0
time_start = time.time()

test_loss_min = 1

for epoch in range(1, Epoch + 1):
    "train"
    train_loss = 0

    for im, label in train_data:
        im = Variable(im)
        im = im.cuda()
        label = Variable(label)
        label = label.cuda()
        out = model(im)
        myloss = torch.sqrt(criterion(out, label))
        optimizer.zero_grad()
        myloss.backward()
        optimizer.step()
        train_loss += myloss.data
    scheduler.step()
    train_losses.append(train_loss.cpu() / len(train_data))
    train_loss_all = train_loss_all + train_loss.cpu() / len(train_data)
    if epoch % 20 == 0:
        train_losses_all.append(train_loss_all / 20)
        train_loss_all = 0

    "test"
    test_loss = 0
    for im, label in test_data:
        im = Variable(im)
        im = im.cuda()
        label = Variable(label)
        label = label.cuda()
        out = model(im)
        RMSEloss = torch.sqrt(criterion(out, label))
        test_loss += RMSEloss.data
    scheduler.step()
    test_losses.append(test_loss.cpu() / len(test_data))
    test_loss_all = test_loss_all + test_loss.cpu() / len(test_data)
    if epoch % 20 == 0:
        test_losses_all.append(test_loss_all / 20)
        test_loss_all = 0
    if (test_loss / len(test_data)) < test_loss_min:
        test_loss_min = (test_loss / len(test_data))
        torch.save(Model, './minloss_net.pth')
    if epoch % 100 == 0:
        print(
            'epoch:%d, Train_loss:%.6f, Test_loss:%.6f' % (
            epoch, train_loss / len(train_data), test_loss / len(test_data)))

time_end = time.time()
print('train_time cost', time_end - time_start, 's')
print(np.mean(test_losses))
print(test_loss_min)

# 保存损失函数和网络模型参数
np.savetxt('./train_loss.csv', train_losses)
np.savetxt('./test_loss.csv', test_losses)
np.savetxt('./train_loss_all.csv', train_losses_all)
np.savetxt('./test_loss_all.csv', test_losses_all)
torch.save(Model, './newnet.pth')

# 流场预测值与真实值对比
out = out.cpu()
label = label.cpu()