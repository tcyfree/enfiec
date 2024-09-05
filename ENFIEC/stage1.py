import os
from datetime import datetime
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from Nets import resnet
from ENFIEC import config, contrastiveloss
from ENFIEC.utils import draw_loss
from ENFIEC.dataset import Cancer_Dataset_1


def train_model(train_loader1, model, optimizer, loss_fn, device):
    losses = []
    loop1 = tqdm(train_loader1)
    for batch_index, (malignant_3_image, malignant_4a_image, benign_3_image_1, benign_3_image_2) in enumerate(loop1):
        malignant_3_image = malignant_3_image.to(device=device)
        malignant_4a_image = malignant_4a_image.to(device=device)
        benign_3_image = torch.cat([benign_3_image_1, benign_3_image_2], dim=0).to(device=device)
        with torch.cuda.amp.autocast():
            feature_malignant_3 = model(malignant_3_image)
            feature_malignant_4a = model(malignant_4a_image)
            feature_benign_3 = model(benign_3_image)
            loss = loss_fn(feature_malignant_3, feature_malignant_4a, feature_benign_3)
            losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop1.set_postfix(loss=loss.item())
    epoch_loss = sum(losses) / len(losses)
    print(f"Loss average over epoch:{epoch_loss}")

    return epoch_loss


def main():
    with open(config.data_path_lesion + "/PB(P)/train.txt", "r", encoding='utf8') as f:
        malignant_3 = f.readlines()
    with open(config.data_path_lesion + "/LM/train.txt", "r", encoding='utf8') as f:
        malignant_4a = f.readlines()
    with open(config.data_path_lesion + "/PB(N)/train.txt", "r", encoding='utf8') as f:
        benign_3 = f.readlines()

    # 3e 由150扩至600
    extended_data_3e = []
    for line in malignant_3:
        line = config.data_path_lesion + line.strip()
        extended_data_3e.append(line)
        extended_data_3e.append(line)
        extended_data_3e.append(line)
        extended_data_3e.append(line)
    extended_data_4a = []
    # 4a 由300扩至600
    for line in malignant_4a:
        line = config.data_path_lesion + line.strip()
        extended_data_4a.append(line)
        extended_data_4a.append(line)
    # 3l 由600扩至1200
    extended_data_3l = []
    for line in benign_3:
        line = config.data_path_lesion + line.strip()
        extended_data_3l.append(line)
        extended_data_3l.append(line)

    dataset1 = Cancer_Dataset_1(
        malignant_3=extended_data_3e,
        malignant_4a=extended_data_4a,
        benign_3=extended_data_3l,
        transform1=config.train_transforms_1,
        transform2=config.train_transforms_2
    )

    train_loader1 = DataLoader(dataset=dataset1, batch_size=config.TRAIN_BATCH_SIZE, num_workers=8, shuffle=True,
                               pin_memory=True)

    # 初始化多编码器模型
    model = resnet.ShareEncoderModel(pretrained=True).to(config.DEVICE)
    # 对比学习
    loss_fn = contrastiveloss.ContrastiveLoss2Tradition(temperature=config.temp)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    draw_losses = []
    best_loss = np.inf
    for epoch in range(config.TRAIN_NUM_EPOCHS):
        print("第" + str(epoch + 1) + "轮:")
        epoch_loss = train_model(train_loader1, model, optimizer, loss_fn, config.DEVICE)
        draw_losses.append(epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            formatted_loss = "{:.4f}".format(best_loss)
            model_state_dict = model.state_dict()
            print("Save Best Loss:" + formatted_loss)
        print("Best Loss:" + str(best_loss))

    # 第一阶段模型保存
    torch.save(model_state_dict, config.Model_lesion_CL_weight_path +
               "stage1_" +
               datetime.now().strftime("%m.%d.%H.%M_") +
               str(config.TRAIN_BATCH_SIZE) + "_" +
               str(config.TRAIN_NUM_EPOCHS) + "_" +
               str(config.LEARNING_RATE) + "_" +
               str(formatted_loss) + ".pth"
               )
    # Loss值保存
    with open(config.Model_lesion_CL_Loss_path + 'stage1_loss_' + datetime.now().strftime("%m.%d.%H.%M_") + str(
            config.LEARNING_RATE) + str(
            config.TRAIN_BATCH_SIZE) + str(config.TRAIN_NUM_EPOCHS) + '.txt', 'w') as f:
        for loss in draw_losses:
            f.write(f"{loss}\n")
    # 画出stage1的loss图
    draw_loss(draw_losses, config.TRAIN_NUM_EPOCHS,
              config.Model_lesion_CL_Loss_path + "stage1_loss_" + datetime.now().strftime("%m.%d.%H.%M_") + str(
                  config.LEARNING_RATE) + str(
                  config.TRAIN_BATCH_SIZE) + str(config.TRAIN_NUM_EPOCHS) + str(config.LEARNING_RATE) + "_" + ".png")


if __name__ == "__main__":
    main()
