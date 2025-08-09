from model import TruVIL
from PIL import Image
from torchvision.transforms import transforms
import os
import torch
import numpy as np
from tqdm import tqdm


# 获取文件夹中的所有文件夹名
def getFlist(file_dir):
    for root, subdirs, files in os.walk(file_dir):
        dir_list = subdirs
        break
    return dir_list


def vis(frame_path, output_dir, checkpoint_path):
    os.makedirs(output_dir, exist_ok=True)
    
    # transforms
    resize_frame = transforms.Resize((240, 432), interpolation=Image.BILINEAR)
    totensor = transforms.ToTensor()

    # 获取n帧组里面的图像
    frame_list = []
    for root, dirs, files in os.walk(frame_path):
        for file in files:
            frame_list.append(os.path.basename(file))

    # 读取n帧图像，存入列表中
    frame_images = []
    for i in range(5):
        # 构造每一帧图像的路径
        frame_image_path = os.path.join(frame_path, frame_list[i])

        # 使用PIL库读取图像
        frame_image = Image.open(frame_image_path)
        frame_images.append(frame_image)

    # resize
    frame_images = [resize_frame(frame) for frame in frame_images]

    # 将n帧图像和标签图像转换为张量，并返回
    frame_images = torch.stack([totensor(frame) for frame in frame_images], dim=1).unsqueeze(0).cuda()

    model = TruVIL().cuda()
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    with torch.no_grad():
        pred = model(frame_images)
        pred = pred.squeeze(0)

        # 二值化
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()

        # 转换为numpy数组
        np_img = pred.cpu().numpy().squeeze()

        # 将numpy数组转换为PIL图像
        img = Image.fromarray(np.uint8(np_img * 255.0))
        img.save(os.path.join(output_dir, os.path.basename(frame_path) + '.png'))


def batch_vis(file_dir, output_dir, checkpoint_path):
    dir_list = getFlist(file_dir)
    for frames_dir in tqdm(dir_list):
        frame_path = os.path.join(file_dir, frames_dir)
        vis(frame_path, output_dir, checkpoint_path)

if __name__ == "__main__":
    checkpoint_path = './checkpoints/TruVIL_train_VI_OP.pth'
    batch_vis('./demo', './output', checkpoint_path)

