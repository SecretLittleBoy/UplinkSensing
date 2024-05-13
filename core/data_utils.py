# Load Data
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.io import loadmat
from torch.utils.data import DataLoader

def tensor_to_pil(tensor):
    image = tensor.cpu().clone()
    # image = image.squeeze(0)
    unloader = transforms.ToPILImage()
    image = unloader(image)
    return image

def img_show(tensor, title=None):
    image = tensor.cpu().clone()
    # image = image.squeeze(0)
    unloader = transforms.ToPILImage()
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    
# Get Training data
class GetDataset(Dataset):
    def __init__(self, img_path, mode='train', cropped_size=48, extern=2, up_scale=2):
        super(GetDataset, self).__init__()
        self.upscale = up_scale
        self.imgs = []
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomCrop(cropped_size),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor()
            ])
            print(f'Initializing the {mode} Dataset...')
            img_path += mode + f'_{cropped_size}/'
            for _, _, files in os.walk(img_path):
                self.imgs.extend([img_path + file for file in files] * extern)  # To duplicate the list for 2 times
        elif mode == 'test':
            # Maybe it is not good to use RandomCrop
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
            print(f'Initializing the {mode} Dataset...')
            img_path += mode + '/'
            print("Img Path:"+img_path)
            for _, _, files in os.walk(img_path):
                self.imgs.extend([img_path + file for file in files])  # To duplicate the list for 2 times            
        else:
            print('mode error')
        
        np.random.shuffle(self.imgs)
        print(f"Length of {mode} dataset: {len(self.imgs)}")
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        imgpath = self.imgs[item]
        _Img = Image.open(imgpath).convert('RGB')
        _Img = self.transform(_Img)

        return imgpath.split('/')[-1], _Img

class MyDataset(Dataset):
    def __init__(self, data_path, mode='train'):
        super(MyDataset, self).__init__()
        fullname_list, filename_list = [], []
        if mode == 'train' :
            print(f'Initializing the {mode} Dataset...')
            for root, dirs, files in os.walk(data_path):
                for filename in files:
                    fullname_list.append(os.path.join(root, filename)) # 文件名列表，包含完整路径
                    filename_list.append(filename)                     # 文件名列表，只包含文件名
            self.fullname_list = fullname_list
        elif mode == 'test':
            print(f'Initializing the {mode} Dataset...')
            for root, dirs, files in os.walk(data_path):
                for filename in files:
                    fullname_list.append(os.path.join(root, filename)) # 文件名列表，包含完整路径
                    filename_list.append(filename)                     # 文件名列表，只包含文件名
            self.fullname_list = fullname_list
        else:
            print('mode error')
    def __getitem__(self, index):
        Mat_dict = loadmat(self.fullname_list[index])
        Mat_file = Mat_dict['H_full_spplit12']
        # print(np.shape(Mat_file))
        H_full = Mat_file[0:16,:]
        H_split1 = Mat_file[16:32,:]
        H_split2 = Mat_file[32:48,:]
        return H_full,H_split1,H_split2
    def __len__(self):
        return len(self.fullname_list)




