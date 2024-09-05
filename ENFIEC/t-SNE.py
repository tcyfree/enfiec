import shutil
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from torchvision import transforms
from PIL import Image
from Nets import resnet
from ENFIEC import config


# 读取图片路径和标签
def read_data():
    with open(config.data_path_lesion + "/PB(P)/train.txt", "r", encoding='utf8') as f:
        malignant_3 = f.readlines()
    with open(config.data_path_lesion + "/LM/train.txt", "r", encoding='utf8') as f:
        malignant_4a = f.readlines()  # 300
    with open(config.data_path_lesion + "/PB(N)/train.txt", "r", encoding='utf8') as f:
        benign_3 = f.readlines()  # 600
    #
    # # 3e由300扩至600
    extended_data_3e = []
    for line in malignant_3:
        line = line.strip()
        extended_data_3e.append(line)
        extended_data_3e.append(line)
        extended_data_3e.append(line)
        extended_data_3e.append(line)
    # # 4a由600扩至1200
    for line in malignant_4a:
        line = line.strip()
        extended_data_3e.append(line)
        extended_data_3e.append(line)
    # # 3l由600扩至1200
    for line in benign_3:
        line = line.strip()
        extended_data_3e.append(line)
        extended_data_3e.append(line)

    image_paths = []
    labels = []
    for line in extended_data_3e:
        path, label = line.strip().split(';')
        label = label.replace("\n", "")
        path = config.data_path_lesion + path
        image_paths.append(path)
        labels.append(int(label))
    return image_paths, labels


# 加载图片并进行预处理
def load_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image


# 加载预训练模型并提取特征
def extract_features_1(model, image_paths, transform):
    model.eval()
    features = []
    with torch.no_grad():
        for image_path in image_paths:
            image = load_image(image_path, transform).unsqueeze(0)
            image = image.to(config.DEVICE)
            feature = model(image).cpu().numpy().flatten()
            features.append(feature)
    return np.array(features)


def extract_features_2(image_paths, transform):
    images = []
    with torch.no_grad():
        for image_path in image_paths:
            image = load_image(image_path, transform).unsqueeze(0)
            image = image.to(config.DEVICE)
            images.append(image)
    return torch.cat(images, dim=0)


# 可视化特征向量
def visualize_tsne_3d(features, labels, class_names, dir):
    # 初始化 t-SNE
    tsne = TSNE(n_components=3, random_state=42)
    features_tsne = tsne.fit_transform(features)
    # 归一化特征
    # scaler = StandardScaler()
    # features_tsne = scaler.fit_transform(features_tsne)
    # x_min, x_max = features_tsne.min(0), features_tsne.max(0)
    # X_norm = (features_tsne - x_min) / (x_max - x_min)

    # 设置颜色
    # custom_colors = ['#2464ab', '#f2a584']
    custom_colors = ['#2464ab', '#f2a584', '#b1182d']

    # 创建 3D 图形
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制散点图
    for i, label in enumerate(np.unique(labels)):
        idx = np.where(labels == label)
        ax.scatter(features_tsne[idx, 0], features_tsne[idx, 1], features_tsne[idx, 2], color=custom_colors[i],
                   label=class_names[i], alpha=1)
    # 设置初始视角
    ax.view_init(elev=-13, azim=94)  # elev 为仰角，azim 为方位角

    # 设置网格不可见
    ax.grid(None)
    # 设置X、Y、Z面的背景是白色
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # 设置坐标轴不可见
    ax.axis('off')

    # 设置标题和标签
    ax.set_title('3D t-SNE Visualization of Image Features', fontsize=16)
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.set_zlabel('t-SNE Component 3', fontsize=12)
    ax.legend(loc='best', fontsize=10)

    # 显示图形
    plt.savefig(config.Model_lesion_path + '/t-SNE/' + dir + '/3d-' + datetime.now().strftime("%m.%d.%H.%M") + ".pdf")
    plt.show()


def visualize_tsne_2d(features, labels, class_names, dir):
    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features)
    # 归一化
    x_min, x_max = features_tsne.min(0), features_tsne.max(0)
    features_tsne = (features_tsne - x_min) / (x_max - x_min)

    plt.figure(figsize=(12, 12))
    color_list = ['#5555DD', '#55DD55', '#DD5555']

    for i, label in enumerate(np.unique(labels)):
        idx = np.where(labels == label)[0]
        plt.scatter(features_tsne[idx, 0], features_tsne[idx, 1], color=color_list[i], label=class_names[i], alpha=1,
                    linewidth=2)
    # 调整图例大小

    plt.legend(loc="best", fontsize='xx-large', frameon=False)
    ax = plt.gca()  # gca:get current axis得到当前轴
    ax.spines['right'].set_linewidth(2)  # 设置边框线宽为2.0
    ax.spines['top'].set_linewidth(2)  # 设置边框线宽为2.0
    ax.spines['bottom'].set_linewidth(2)  # 设置边框线宽为2.0
    ax.spines['left'].set_linewidth(2)  # 设置边框线宽为2.0
    plt.xticks(fontsize=30)  # 定义坐标轴刻度
    plt.yticks(fontsize=30)
    plt.xlabel('t-SNE Dimension 1', fontsize=20)  # 定义坐标轴标题
    plt.ylabel('t-SNE Dimension 2', fontsize=20)
    plt.title('t-SNE Visualization', fontsize=24)  # 定义图题
    plt.savefig(config.Model_lesion_path + '/t-SNE/' + dir + '/2d-' + datetime.now().strftime("%m.%d.%H.%M") + ".pdf")
    # plt.show()


# 主程序
if __name__ == '__main__':
    image_paths, labels = read_data()
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model1 = resnet.ShareEncoderModel2TSNE(pretrained=False).to(config.DEVICE)
    model2 = resnet.ShareEncoderModelTSNE(pretrained=True).to(config.DEVICE)
    model3 = resnet.ClassifierBaseModelTSNE().to(config.DEVICE)
    # 初始化多编码器模型
    with torch.no_grad():
        images_3e = extract_features_2(image_paths[:600], transform).to(device=config.DEVICE)
        images_4a = extract_features_2(image_paths[600:1200], transform).to(device=config.DEVICE)
        images_3l = extract_features_2(image_paths[1200:], transform).to(device=config.DEVICE)

        # 随机初始化
        feature_malignant_3 = model1(images_3e)
        feature_malignant_4a = model1(images_4a)
        feature_benign_3 = model1(images_3l)
        features = [feature_malignant_3, feature_malignant_4a, feature_benign_3]
        features = torch.cat(features, dim=0)
        features = np.array(features.cpu().detach())
        features_original = features

        # 对比学习
        model2.load_state_dict(
            torch.load(
                config.Model_lesion_CL_weight_path + "stage1_xxxx.pth",
                map_location='cuda:0'), strict=False)
        feature_malignant_3 = model2(images_3e)
        feature_malignant_4a = model2(images_4a)
        feature_benign_3 = model2(images_3l)
        features = [feature_malignant_3, feature_malignant_4a, feature_benign_3]
        features = torch.cat(features, dim=0)
        features = np.array(features.cpu().detach())
        features_CL = features

        # 交叉熵
        model3.load_state_dict(
            torch.load(
                config.Model_lesion_CE_weight_path + 'xxxx.pth',
                map_location='cuda:0'), strict=False)
        feature_malignant_3 = model3(images_3e)
        feature_malignant_4a = model3(images_4a)
        feature_benign_3 = model3(images_3l)
        features = [feature_malignant_3, feature_malignant_4a, feature_benign_3]
        features = torch.cat(features, dim=0)
        features = np.array(features.cpu().detach())
        features_CE = features

    # 使用t-SNE降维并可视化
    class_names = ['PB negative', 'PB positive', 'LM positive']
    visualize_tsne_2d(features_original, labels, class_names, 'Random')
    visualize_tsne_2d(features_CE, labels, class_names, 'CE')
    visualize_tsne_2d(features_CL, labels, class_names, 'CL')
