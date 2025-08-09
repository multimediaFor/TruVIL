import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from metric import F1, IoU
from base_dataset import FramesDataset
from model import TruVIL
from loss import Focal_IoU_LOSS
import os

# 定义超参数
BATCH_SIZE = 16  # 批次大小
NUM_WORKERS = 8  # 数据加载的线程数

# 验证集
val_dataset1 = FramesDataset(root_dir='/test_VI', n_frames=5, istrain=True)
val_dataset2 = FramesDataset(root_dir='/test_OP', n_frames=5, istrain=True)
val_dataset3 = FramesDataset(root_dir='/test_CP', n_frames=5, istrain=True)

# 定义数据加载器
val_loader1 = DataLoader(val_dataset1, batch_size=BATCH_SIZE, shuffle=False)
val_loader2 = DataLoader(val_dataset2, batch_size=BATCH_SIZE, shuffle=False)
val_loader3 = DataLoader(val_dataset3, batch_size=BATCH_SIZE, shuffle=False)
val_loaders = [val_loader1, val_loader2, val_loader3]

device = torch.device("cuda")
model = TruVIL().to(device)
model.load_state_dict(torch.load('./checkpoints/TruVIL_train_VI_OP.pth'))


def test(model, val_loaders):
    model.eval()  
    val_f1s = [0.0] * len(val_loaders)  
    val_mious = [0.0] * len(val_loaders) 
    with torch.no_grad():
        for i, val_loader in enumerate(val_loaders):
            with tqdm(total=len(val_loader)) as pbar:
                for batch_idx, (inputs, targets) in enumerate(val_loader):
                    inputs, targets = inputs.to(device), targets.to(device)  
                    outputs = model(inputs)  
                    
                    f1 = F1(outputs, targets) 
                    val_f1s[i] += f1 * inputs.size(0)  
                    miou = IoU(outputs, targets)  
                    val_mious[i] += miou * inputs.size(0) 
                    pbar.update(1)  
                    
            val_f1s[i] /= len(val_loader.dataset)  
            val_mious[i] /= len(val_loader.dataset)  

            print('Val Set {}: F1: {:.6f}, IoU: {:.6f}'.format(
                i + 1,
                val_f1s[i],
                val_mious[i]))

def main():
    test(model, val_loaders)


if __name__ == "__main__":
    main()
