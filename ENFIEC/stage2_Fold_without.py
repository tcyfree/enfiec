import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import Nets.resnet as resnet
import torch
from torch import nn, optim
import torch.nn.functional as F
from datetime import datetime
import ENFIEC.config as config
from torch.utils.data import DataLoader
from tqdm import tqdm
from ENFIEC.dataset import Cancer_Dataset_2


def train_and_validation(train_loader, val_loader, model, optimizer, loss_fn):
    model.train()
    train_losses = []
    loop1 = tqdm(train_loader)
    for batch_index, (image, label, path) in enumerate(loop1):
        image = image.to(device=config.DEVICE)
        label = [int(char) for char in label]
        label = torch.tensor(label).to(device=config.DEVICE)  # 将标签转换为浮点数张量
        with torch.cuda.amp.autocast():
            output = model(image)
            loss = loss_fn(output, label)  # 使用交叉熵损失函数
            train_losses.append(loss.item())
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop1.set_postfix(loss=loss.item())
    print(f"Train Loss -------------------------->: {sum(train_losses) / len(train_losses)}")

    model.eval()
    val_losses = []
    val_labels, val_preds = [], []
    loop2 = tqdm(val_loader)
    with torch.no_grad():
        for batch_index, (image, label, path) in enumerate(loop2):
            image = image.to(device=config.DEVICE)
            label = [int(char) for char in label]
            label = torch.tensor(label).to(device=config.DEVICE)
            output = model(image)
            softmax_output = F.softmax(output, dim=1)
            preds = softmax_output[:, 1]
            preds = (preds >= 0.5).float()
            val_labels.extend(label.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())
            loss = loss_fn(output, label)  # 使用交叉熵损失函数
            val_losses.append(loss.item())
    print(f"Val Loss ---------------------------->: {sum(val_losses) / len(val_losses)}")

    return np.mean(train_losses), np.mean(val_losses), accuracy_score(val_labels, val_preds)


def main():
    with open(config.data_path_lesion + "/PB(P)/train.txt", "r", encoding='utf8') as f:
        malignant_3 = f.readlines()
    with open(config.data_path_lesion + "/PB(N)/train.txt", "r", encoding='utf8') as f:
        benign_3 = f.readlines()

    for line in benign_3[:150]:
        malignant_3.append(line)

    malignant_3 = [config.data_path_lesion + line for line in malignant_3]
    # K折交叉验证
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_train_loss = []
    fold_val_loss = []
    LR = 0.00002
    for fold, (train_index, val_index) in enumerate(folds.split(malignant_3)):
        print(f'Fold {fold + 1}')
        train_losses = []
        val_losses = []

        # 创建数据集和数据加载器
        malignant_3 = np.array(malignant_3)
        train_dataset = Cancer_Dataset_2(total_data=malignant_3[train_index], transform=config.train_transforms_2)
        val_dataset = Cancer_Dataset_2(total_data=malignant_3[val_index], transform=config.test_transforms)

        train_loader = DataLoader(train_dataset, batch_size=80, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=60, shuffle=False, num_workers=4)

        model = resnet.ClassifierBaseModel().to(config.DEVICE)
        # 配置不同层的学习率
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=config.WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()

        best_acc = float('inf')
        best_model_state = None
        # 训练
        for epoch in range(100):
            print("第" + str(epoch + 1) + "轮")
            train_loss, val_loss, acc = train_and_validation(train_loader, val_loader, model, optimizer,
                                                             criterion)  # 训练模型
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            # 保存验证acc最高的模型
            if acc < best_acc:
                best_acc = acc
                best_model_state = model.state_dict()
            print(f"Fold {fold + 1}, ACC {acc:.4f}")

        # 每折模型评估
        predictions = []
        val_labels = []
        model.load_state_dict(best_model_state)
        model.eval()
        loop = tqdm(val_loader)
        with torch.no_grad():
            for batch_index, (image, label, path) in enumerate(loop):
                image = image.to(device=config.DEVICE)
                label = [int(char) for char in label]
                label = torch.tensor(label).to(device=config.DEVICE)
                output = model(image)
                val_labels.extend(label.cpu().numpy())
        softmax_output = F.softmax(output, dim=1)
        prediction = softmax_output[:, 1]
        predictions.extend(prediction.cpu().numpy())
        with open(config.Model_lesion_CE_Prediction_path + '/Fold/' + f"Fold_{fold + 1}_pred_" +
                  datetime.now().strftime("%m.%d.%H.%M_") + "_without.txt", 'w') as f:
            for x, y, z in zip(list(path), val_labels, predictions):
                f.write(f"{x};{y};{z}\n")
        # 保存每折模型参数
        torch.save(best_model_state,
                   config.Model_lesion_CE_weight_path + f"Fold_{fold + 1}_" +
                   datetime.now().strftime("%m.%d.%H.%M_without") + f"_ACC_{acc:.4f}_" + ".pth")

        # 保存每折loss
        fold_train_loss.append(train_losses)
        fold_val_loss.append(val_losses)

    # 绘制训练损失和验证损失
    plt.figure(figsize=(12, 12))
    # 每个折的损失
    for i in range(5):
        plt.plot(fold_train_loss[i], label=f'Train Loss Fold {i + 1}', alpha=0.5)
        plt.plot(fold_val_loss[i], label=f'Val Loss Fold {i + 1}', alpha=0.5)

    # 平均损失
    fold_train_loss = np.array(fold_train_loss)
    fold_val_loss = np.array(fold_val_loss)
    mean_train_loss = np.mean(fold_train_loss, axis=0)
    mean_val_loss = np.mean(fold_val_loss, axis=0)
    # 保存loss数据到文件
    np.save(config.Model_lesion_CE_Loss_path + 'stage2_train_loss_without_' +
            datetime.now().strftime("%m.%d.%H.%M_") + '.npy', mean_train_loss)
    np.save(config.Model_lesion_CE_Loss_path + 'stage2_val_loss_without_' +
            datetime.now().strftime("%m.%d.%H.%M_") + '.npy', mean_val_loss)

    plt.plot(mean_train_loss, label='Mean Train Loss', color='blue', linewidth=2)
    plt.plot(mean_val_loss, label='Mean Val Loss', color='orange', linewidth=2)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.savefig(config.Model_lesion_CE_Loss_path + "stage2loss_without_" +
                datetime.now().strftime("%m.%d.%H.%M_") + '_' + str(LR) + ".png")
    plt.show()


if __name__ == "__main__":
    main()
