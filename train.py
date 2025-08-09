import warnings

warnings.filterwarnings("ignore")
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from metric import F1, IoU
from base_dataset import FramesDataset
from model import TruVIL
from loss import Focal_IoU_LOSS
import os

# 定义超参数
LR = 0.0005  # 学习率
EPOCHS = 200  # 训练轮数
BATCH_SIZE = 8  # 批次大小
NUM_WORKERS = 2  # 数据加载的线程数

# 定义训练集和验证集
train_dataset = FramesDataset(root_dir='/users/u202220081200013/jupyterlab/experiments/n_5/train_800_QP23/train_VI_CP',
                              n_frames=5,
                              istrain=True)
val_dataset1 = FramesDataset(root_dir='/users/u202220081200013/jupyterlab/experiments/n_5/test/test_VI', n_frames=5,
                             istrain=True)
val_dataset2 = FramesDataset(root_dir='/users/u202220081200013/jupyterlab/experiments/n_5/test/test_OP', n_frames=5,
                             istrain=True)
val_dataset3 = FramesDataset(root_dir='/users/u202220081200013/jupyterlab/experiments/n_5/test/test_CP', n_frames=5,
                             istrain=True)

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader1 = DataLoader(val_dataset1, batch_size=BATCH_SIZE, shuffle=False)
val_loader2 = DataLoader(val_dataset2, batch_size=BATCH_SIZE, shuffle=False)
val_loader3 = DataLoader(val_dataset3, batch_size=BATCH_SIZE, shuffle=False)

val_loaders = [val_loader1, val_loader2, val_loader3]

# 定义训练器
device = torch.device("cuda")
model = TruVIL().to(device)
# model.load_state_dict(torch.load(''))

# 定义损失函数
criterion = Focal_IoU_LOSS()

# 定义优化器
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# 定义学习率衰减策略
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)

# 权重保存位置
save_dir = './weights'


# 定义训练函数
def train(model, train_loader, val_loaders, criterion, optimizer, scheduler, save_dir):
    # 加载最新的权重文件，如果存在的话
    start_epoch = 0
    latest_weight_file = os.path.join(save_dir, 'latest.pth')
    if os.path.exists(latest_weight_file):
        checkpoint = torch.load(latest_weight_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print('Loaded the latest checkpoint from epoch %d.' % start_epoch)

    for epoch in range(start_epoch, EPOCHS):
        model.train()  # 将模型设置为训练模式
        train_loss = 0.0  # 记录每个epoch的平均loss
        train_f1 = 0.0  # 记录每个epoch的平均f1
        train_miou = 0.0  # 记录每个epoch的平均miou

        # 使用tqdm显示进度条
        with tqdm(total=len(train_loader)) as pbar:
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()  # 梯度清零
                outputs = model(inputs)  # 前向传播
                loss = criterion(outputs, targets)  # 计算loss
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数

                train_loss += loss.item() * inputs.size(0)  # 累加loss
                f1 = F1(outputs, targets)  # 计算F1值
                train_f1 += f1 * inputs.size(0)  # 累加F1值
                miou = IoU(outputs, targets)  # 计算IoU
                train_miou += miou * inputs.size(0)  # 累加IoU
                # 更新进度条
                pbar.update(1)
                pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch + 1,
                    batch_idx * len(inputs),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()))

        scheduler.step()  # 更新学习率

        train_loss = train_loss / len(train_loader.dataset)  # 计算平均loss
        train_f1 = train_f1 / len(train_loader.dataset)  # 计算平均f1
        train_miou = train_miou / len(train_loader.dataset)  # 计算平均iou

        print('Epoch: [{}/{}], Train Loss: {:.6f}, F1: {:.4f}, IoU: {:.4f}, nextLR: {:.8f}'.format(epoch + 1,
                                                                                                    EPOCHS,
                                                                                                    train_loss,
                                                                                                    train_f1,
                                                                                                    train_miou,
                                                                                                    scheduler.get_last_lr()[                                                                                           0]))
        print("\n")

        # 保存每一轮的权重文件
        epoch_weight_file = os.path.join(save_dir, 'epoch_{:0>3}.pth').format(epoch)
        torch.save(model.state_dict(), epoch_weight_file)

        # 保存最新的权重文件
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        latest_weight_file = os.path.join(save_dir, 'latest.pth')
        torch.save(checkpoint, latest_weight_file)

        # 验证
        model.eval()  # 将模型设置为验证模式
        val_losses = [0.0] * len(val_loaders)  # 记录每个epoch的平均loss
        val_f1s = [0.0] * len(val_loaders)  # 记录每个epoch的平均F1值
        val_mious = [0.0] * len(val_loaders)  # 记录每个epoch的平均IoU

        # 不需要计算梯度
        with torch.no_grad():
            for i, val_loader in enumerate(val_loaders):
                # 使用tqdm显示进度条
                with tqdm(total=len(val_loader)) as pbar:
                    for batch_idx, (inputs, targets) in enumerate(val_loader):
                        inputs, targets = inputs.to(device), targets.to(device)  # 将inputs和targets转移到GPU上
                        outputs = model(inputs)  # 前向传播
                        loss = criterion(outputs, targets)  # 计算loss
                        val_losses[i] += loss.item() * inputs.size(0)  # 累加loss
                        f1 = F1(outputs, targets)  # 计算F1值
                        val_f1s[i] += f1 * inputs.size(0)  # 累加F1值
                        miou = IoU(outputs, targets)  # 计算mIoU
                        val_mious[i] += miou * inputs.size(0)  # 累加mIoU
                        pbar.update(1)  # 更新进度条
                        pbar.set_description('Val Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                            epoch + 1,
                            batch_idx * len(inputs),
                            len(val_loader.dataset),
                            100. * batch_idx / len(val_loader),
                            loss.item()))
                val_losses[i] /= len(val_loader.dataset)  # 计算平均loss
                val_f1s[i] /= len(val_loader.dataset)  # 计算平均F1值
                val_mious[i] /= len(val_loader.dataset)  # 计算平均mIoU

                print('Epoch: [{}/{}], Val Set {}: Val Loss: {:.6f}, F1: {:.4f}, IoU: {:.4f}'.format(
                    epoch + 1,
                    EPOCHS,
                    i + 1,
                    val_losses[i],
                    val_f1s[i],
                    val_mious[i]))

                print("\n")

        print("=======================================================================================================")


# 训练模型
def main():
    train(model, train_loader, val_loaders, criterion, optimizer, scheduler, save_dir)


if __name__ == "__main__":
    main()
