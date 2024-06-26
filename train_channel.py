import os
import platform
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from core import Net, MyDataset, GetDataset
import matplotlib.pyplot as plt
import numpy as np


def train(train_data, net, lossF, optimizer, device, runningLoss, count):
    net.train()
    for H_full, H_split1, H_split2 in train_data:
        H_full, H_split1, H_split2 = H_full.float().to(device), H_split1.float().to(device), H_split2.float().to(device)
        # grad zero for parameter updating
        optimizer.zero_grad()
        # start training
        # print('Size of H_full:')
        H_full = H_full.unsqueeze(0)
        # print(np.shape(H_full))

        outputs = net(H_full)
        # outputs1 = net(H_split1.unsqueeze(0))
        # outputs2 = net(H_split2.unsqueeze(0))
        # print('Size of outputs:' + str(np.shape(outputs)))
        # print('Size of H_split1:' + str(np.shape(H_split1)))
        # print('Size of H_split2:' + str(np.shape(H_split2)))
        loss = lossF(torch.sum(outputs), torch.sum(H_split1 + H_split2))
        # loss = lossF(outputs, outputs1 + outputs2)
        loss.backward()     # backward
        optimizer.step()    # parameters update
        runningLoss += loss.item()  # sum the loss for every batch
        count += 1
    return runningLoss, count


def eval(eval_data, net, lossF, device, runningLoss, count):
    pass


if __name__ == "__main__":
    # Define Path and Params
    RootPath = os.getcwd()
    train_DataPath = './Dataset/train/'
    eval_DataPath = './Dataset/eval/'
    model_savepath = './save/'
    batches = 1
    epochs = 5
    gpu_id = 0
    if platform.system() == 'Windows':
        data_threads = 0
    elif platform.system() == 'Linux':
        data_threads = 4
    elif platform.system() == 'Darwin':
        data_threads = 0
        print('MacOS System may have some problems')
    else:
        print(platform.system())
        print('Unknown System')
        assert False

    # Initial Network
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)
    print(" Net Structure:\n", net)
    # Create Dataset
    train_Dataset = MyDataset(train_DataPath, mode='train')
    eval_Dataset = MyDataset(eval_DataPath, mode='test')
    train_loader = DataLoader(dataset=train_Dataset, batch_size=batches, shuffle=True)
    eval_loader = DataLoader(dataset=eval_Dataset, batch_size=batches, shuffle=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # 初始化迭代器
    lossF = torch.nn.MSELoss().to(device)  # 初始化损失函数

    # Train
    history = []
    for ite in range(epochs):
        runningLoss = 0.0
        count = 0
        print(f"Epoch {ite + 1}\n-------------------------------")
        runningLoss, count = train(train_loader, net, lossF, optimizer, device, runningLoss, count)
        # dynamically update the running loss
        averageLoss = runningLoss / (count + 1)
        if len(history) != 0 and averageLoss < min(history):
            torch.save(net, model_savepath + 'best.pth')
            print(f'current best loss:{averageLoss}')
        history += [averageLoss]
        print('[INFO] Epoch %d loss: %.3f' % (ite + 1, averageLoss))
        runningLoss = 0.0
    print('[INFO] Finished Training \n')
    print(f'MSE of the last epoch{epochs}:{averageLoss}')
    torch.save(net, model_savepath + 'last_epoch.pth')
    # print model
    plt.plot(history, label='MSE')
    plt.legend(loc='best')
    f = plt.gcf()
    f.savefig('training_loss.png')
    f.clear()

    # Eval
