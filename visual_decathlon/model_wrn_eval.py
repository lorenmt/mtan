import torch
import torch.nn as nn

import torch.nn.init as init
import torch.nn.functional as F
import torchvision
import numpy as np
import torch.optim as optim
import pickle
import argparse

from torchvision.transforms import transforms

parser = argparse.ArgumentParser(description='Visual Decathlon Challenge: Evaluation')
parser.add_argument('--dataset', default='imagenet', type=str, help='choose dataset: imagenet, notimagenet')
opt = parser.parse_args()

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        n = int((depth - 4) / 6)
        k = widen_factor
        filter = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, filter[0], stride=1)
        self.layer1 = self._wide_layer(wide_basic, filter[1], n, stride=2)
        self.layer2 = self._wide_layer(wide_basic, filter[2], n, stride=2)
        self.layer3 = self._wide_layer(wide_basic, filter[3], n, stride=2)
        self.bn1 = nn.BatchNorm2d(filter[3], momentum=0.9)

        self.linear = nn.ModuleList([nn.Sequential(
            # nn.Linear(filter[3], filter[3]),
            # nn.ReLU(inplace=True),
            nn.Linear(filter[3], num_classes[0]),
            nn.Softmax(dim=1))])

        # attention modules
        self.encoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])])
        self.encoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[1]])])

        for j in range(10):
            if j < 9:
                self.encoder_att.append(nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])]))
                self.linear.append(nn.Sequential(nn.Linear(filter[3], num_classes[j + 1]),
                                                 nn.Softmax(dim=1)))
            for i in range(3):
                self.encoder_att[j].append(self.att_layer([2 * filter[i + 1], filter[i + 1], filter[i + 1]]))

        for i in range(3):
            if i < 2:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 2]]))
            else:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))

    def conv_layer(self, channel):
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=channel[1]),
            nn.ReLU(inplace=True),
        )
        return conv_block

    def att_layer(self, channel):
        att_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[2]),
            nn.Sigmoid(),
        )
        return att_block

    def _wide_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, k):
        g_encoder = [0] * 4

        atten_encoder = [0] * 10
        for i in range(10):
            atten_encoder[i] = [0] * 4
        for i in range(10):
            for j in range(4):
                atten_encoder[i][j] = [0] * 3

        g_encoder[0] = self.conv1(x)
        g_encoder[1] = self.layer1(g_encoder[0])
        g_encoder[2] = self.layer2(g_encoder[1])
        g_encoder[3] = F.relu(self.bn1(self.layer3(g_encoder[2])))

        # apply attention modules
        for j in range(4):
            if j == 0:
                atten_encoder[k][j][0] = self.encoder_att[k][j](g_encoder[0])
                atten_encoder[k][j][1] = (atten_encoder[k][j][0]) * g_encoder[0]
                atten_encoder[k][j][2] = self.encoder_block_att[j](atten_encoder[k][j][1])
                atten_encoder[k][j][2] = F.max_pool2d(atten_encoder[k][j][2], kernel_size=2, stride=2)
            else:
                atten_encoder[k][j][0] = self.encoder_att[k][j](
                    torch.cat((g_encoder[j], atten_encoder[k][j - 1][2]), dim=1))
                atten_encoder[k][j][1] = (atten_encoder[k][j][0]) * g_encoder[j]
                atten_encoder[k][j][2] = self.encoder_block_att[j](atten_encoder[k][j][1])
                if j < 3:
                    atten_encoder[k][j][2] = F.max_pool2d(atten_encoder[k][j][2], kernel_size=2, stride=2)

        pred = F.adaptive_avg_pool2d(atten_encoder[k][-1][-1], 1)
        pred = pred.view(pred.size(0), -1)

        out = self.linear[k](pred)
        return out

    def model_fit(self, x_pred, x_output, num_output):
        # convert a single label into a one-hot vector
        x_output_onehot = torch.zeros((len(x_output), num_output)).to(device)
        x_output_onehot.scatter_(1, x_output.unsqueeze(1), 1)

        # apply cross-entropy loss
        loss = x_output_onehot * torch.log(x_pred + 1e-20)
        return torch.sum(-loss, dim=1)


def data_transform(data_path, name, train=True):
    with open(data_path + 'decathlon_mean_std.pickle', 'rb') as handle:
        dict_mean_std = pickle._Unpickler(handle)
        dict_mean_std.encoding = 'latin1'
        dict_mean_std = dict_mean_std.load()

    means = dict_mean_std[name + 'mean']
    stds = dict_mean_std[name + 'std']

    if name in ['gtsrb', 'omniglot', 'svhn']:  # no horz flip
        transform_train = transforms.Compose([
            transforms.Resize(72),
            transforms.CenterCrop(72),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(72),
            transforms.RandomCrop(72),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    if name in ['gtsrb', 'omniglot', 'svhn']:  # no horz flip
        transform_test = transforms.Compose([
            transforms.Resize(72),
            transforms.CenterCrop(72),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    else:
        transform_test = transforms.Compose([
            transforms.Resize(72),
            transforms.CenterCrop(72),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    if train:
        return transform_train
    else:
        return transform_test


im_train_set = [0] * 10
im_test_set = [0] * 10
im_val_set = [0] * 10
data_path = 'decathlon-1.0-data/'
data_name = ['imagenet12', 'aircraft', 'cifar100', 'daimlerpedcls', 'dtd',
             'gtsrb', 'omniglot', 'svhn', 'ucf101', 'vgg-flowers']
data_class = [1000, 100, 100, 2, 47, 43, 1623, 10, 101, 102]
for i in range(10):
    im_train_set[i] = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(data_path + data_name[i] + '/train',
                                                  transform=data_transform(data_path, data_name[i])),
                                                  batch_size=128,
                                                  shuffle=True,
                                                  num_workers=4, pin_memory=True)
    im_val_set[i] = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(data_path + data_name[i] + '/val',
                                                transform=data_transform(data_path,data_name[i])),
                                                batch_size=128,
                                                shuffle=True,
                                                num_workers=4, pin_memory=True)
    im_test_set[i] = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(data_path + data_name[i] + '/test',
                                                 transform=data_transform(data_path, data_name[i], train=False)),
                                                 batch_size=100,
                                                 shuffle=False)

# define WRN model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
WideResNet_MTAN = WideResNet(depth=28, widen_factor=4, num_classes=data_class).to(device)
if opt.dataset == 'imagenet':
    optimizer = optim.SGD(WideResNet_MTAN.parameters(), lr=0.1 * (0.5 ** 6), weight_decay=5e-5, nesterov=True, momentum=0.9)
    WideResNet_MTAN.load_state_dict(torch.load('model_weights/imagenet'))
    start_index = 0
if opt.dataset == 'notimagenet':
    optimizer = optim.SGD(WideResNet_MTAN.parameters(), lr=0.01 * (0.5 ** 2), weight_decay=5e-5, nesterov=True, momentum=0.9)
    WideResNet_MTAN.load_state_dict(torch.load('model_weights/wrn_final'))
    start_index = 1

avg_cost = np.zeros([10, 4], dtype=np.float32)
ans = {}
for k in range(start_index, 10):
    WideResNet_MTAN = WideResNet_MTAN.train()
    cost = np.zeros(2, dtype=np.float32)
    train_dataset = iter(im_train_set[k])
    train_batch = len(train_dataset)
    for i in range(train_batch):
        train_data, train_label = train_dataset.next()
        train_label = train_label.type(torch.LongTensor)
        train_data, train_label = train_data.to(device), train_label.to(device)
        train_pred1 = WideResNet_MTAN(train_data, k)

        # reset optimizer with zero gradient
        optimizer.zero_grad()
        train_loss1 = WideResNet_MTAN.model_fit(train_pred1, train_label, num_output=data_class[k])
        train_loss = torch.mean(train_loss1)
        train_loss.backward()
        optimizer.step()

        # calculate training loss and accuracy
        train_predict_label1 = train_pred1.data.max(1)[1]
        train_acc1 = train_predict_label1.eq(train_label).sum().item() / train_data.shape[0]

        cost[0] = torch.mean(train_loss1).item()
        cost[1] = train_acc1
        avg_cost[k][0:2] += cost / train_batch

    train_dataset = iter(im_val_set[k])
    train_batch = len(train_dataset)
    for i in range(train_batch):
        train_data, train_label = train_dataset.next()
        train_label = train_label.type(torch.LongTensor)
        train_data, train_label = train_data.to(device), train_label.to(device)
        train_pred1 = WideResNet_MTAN(train_data, k)

        # reset optimizer with zero gradient
        optimizer.zero_grad()
        train_loss1 = WideResNet_MTAN.model_fit(train_pred1, train_label, num_output=data_class[k])
        train_loss = torch.mean(train_loss1)
        train_loss.backward()
        optimizer.step()

        # calculate training loss and accuracy
        train_predict_label1 = train_pred1.data.max(1)[1]
        train_acc1 = train_predict_label1.eq(train_label).sum().item() / train_data.shape[0]

        cost[0] = torch.mean(train_loss1).item()
        cost[1] = train_acc1
        avg_cost[k][2:4] += cost / train_batch

    # evaluating test data
    with torch.no_grad():
        WideResNet_MTAN = WideResNet_MTAN.eval()
        test_dataset = iter(im_test_set[k])
        test_batch = len(test_dataset)
        test_label = []
        for i in range(test_batch):
            test_data, _ = test_dataset.next()
            test_data = test_data.to(device)
            test_pred1 = VGG16(test_data, k)

            # calculate testing loss and accuracy
            test_predict = test_pred1.data.max(1)[1]
            test_pred = test_predict.cpu().numpy()
            test_label.extend(test_pred)
        ans[data_name[k]] = test_label

        print('DATASET: {:s} || TRAIN {:.4f} {:.4f} | TEST {:.4f} {:.4f}'
              .format(data_name[k], avg_cost[k][0], avg_cost[k][1], avg_cost[k][2], avg_cost[k][3]))
        print('Evaluating DATASET: {:s} ...'.format(data_name[k]))

pickle_out = open("ans.pickle", "wb")
pickle.dump(ans, pickle_out)
pickle_out.close()
