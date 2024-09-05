import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

from Nets import resnet
from ENFIEC import config


# 定义Grad-CAM类
class GradCAM:
    def __init__(self, model, fc, target_layer_names):
        self.model = model
        self.fc = fc
        self.target_layer_names = target_layer_names
        self.gradients = []
        self.activations = []

        # 注册钩子
        self.hooks = []
        for name in self.target_layer_names:
            layer = dict([*self.model.named_modules()])[name]
            self.hooks.append(layer.register_forward_hook(self.save_activation))
            self.hooks.append(layer.register_backward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        self.activations.append(output)

    def save_gradient(self, module, grad_in, grad_out):
        self.gradients.append(grad_out[0].detach())

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        output = self.model(x)
        output = torch.flatten(output, 1)
        output = self.fc(output)
        return output, self.gradients, self.activations


# 定义图像预处理函数
def preprocess_image(img_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    original_img = Image.open(img_path).convert('RGB')
    img = preprocess(original_img)
    img = img.to(config.DEVICE)
    return original_img, img


# 可视化热力图
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    plt.imshow(cam)
    plt.show()


def generate_heatmap(gradients, activations):
    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)  # 计算权重
    heatmap = torch.sum(weights * activations, dim=1).squeeze()  # 加权和
    # heatmap = F.relu(heatmap)  # 通过 ReLU 函数使热力图非负
    heatmap = heatmap.detach().cpu().numpy()

    # 归一化热力图
    heatmap -= heatmap.min()
    heatmap /= heatmap.max()

    return heatmap


def apply_threshold(heatmap, threshold):
    # 应用阈值，低于阈值的区域设为0
    heatmap[heatmap < 0.3] *= 0
    return heatmap


number = 0


def overlay_heatmap(heatmap, image, image_name, save_path, original_img):
    global number
    # heatmap = apply_threshold(heatmap, 0.5)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    cam = np.float32(heatmap) * 0.7 + np.float32(image)  # 大值相加
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)

    # 原始图像
    original_path = f"{save_path}/{number}_{image_name}_original.png"
    # 归一化的图像
    guiyi_path = f"{save_path}/{number}_{image_name}_guiyi.png"
    # 保存热力图和叠加图像
    heatmap_path = f"{save_path}/{number}_{image_name}_heatmap.png"
    overlay_path = f"{save_path}/{number}_{image_name}_overlay.png"
    number += 1
    cv2.imwrite(heatmap_path, heatmap)
    cv2.imwrite(overlay_path, cam)
    cv2.imwrite(guiyi_path, image)
    original_img.save(original_path, 'PNG')


def enhance_contrast(heatmap):
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.equalizeHist(heatmap)
    heatmap = np.float32(heatmap) / 255
    return heatmap


# 主函数
def main(img_paths):
    # 加载预训练模型
    model1 = resnet.ShareEncoderModel(pretrained=True)
    model4 = resnet.ClassifierModel(model1.shared_encoder).to(config.DEVICE)
    model4.load_state_dict(
        torch.load(
            config.Model_lesion_CL_weight_path + 'xxxx.pth',
            map_location=config.DEVICE), strict=False)
    grad_cam = GradCAM(model4.model, model4.fc, target_layer_names=['layer4'])

    for img_path in img_paths:
        # 预处理图像
        image_name = img_path.split('/')[-1].split('_')[0]
        original_img, image = preprocess_image(img_path)
        image = image.unsqueeze(0)
        image.requires_grad = True

        # 前向传播
        output, gradients, activations = grad_cam(image)
        # output = F.softmax(output, dim=1)
        class_idx = torch.argmax(output, dim=1)
        # class_idx2 = np.argmax(output.cpu().data.numpy(), axis=1)

        # 反向传播
        loss = output[0, class_idx]
        model4.zero_grad()
        loss.backward()

        # 获取梯度
        heatmap = generate_heatmap(gradients[0], activations[0])
        # heatmap = enhance_contrast(heatmap)
        # 移除钩子
        # grad_cam.remove_hooks()

        # 叠加热力图到原始图像
        img = image[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = np.uint8(255 * (img - img.min()) / (img.max() - img.min()))  # 归一化并转换为整数
        overlay_heatmap(heatmap, img, image_name, config.heatmap_save_path_lesion, original_img)


if __name__ == "__main__":
    with open(config.data_path_lesion + "/PB(P)/test.txt", "r", encoding='utf8') as f:
        malignant_3 = f.readlines()
    with open(config.data_path_lesion + "/PB(N)/test.txt", "r", encoding='utf8') as f:
        benign_3 = f.readlines()
    # 获取所有图像路径
    images_path = [config.data_path_lesion + line.split(';')[0] for line in malignant_3] + [config.data_path_lesion + line.split(';')[0] for line in benign_3]
    main(images_path)
