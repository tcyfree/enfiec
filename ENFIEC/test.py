import os
from datetime import datetime
import torch.nn.functional as F
from Nets import resnet
import ENFIEC.config as config
from torch.utils.data import DataLoader

from ENFIEC.dataset import Cancer_Dataset
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, accuracy_score, precision_score, \
    recall_score, f1_score, precision_recall_curve
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy import interp


def standard(y_true, y_pred):
    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)

    # 计算精确率 所有预测positive中真的为正类的概率
    precision = precision_score(y_true, y_pred)

    # 计算召回率
    recall = recall_score(y_true, y_pred)

    # 计算特异性
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    # 计算F1-Measure
    f1 = f1_score(y_true, y_pred)

    # 打印结果
    print(f"准确率：{accuracy * 100:.2f}%")
    print(f"精确率：{precision * 100:.2f}%")
    print(f"召回率：{recall * 100:.2f}%")
    print(f"特异性：{specificity * 100:.2f}%")
    print(f"F1-Measure：{f1 * 100:.2f}%")


def roc(y_true, y_score, pos_label=1):
    """
    y_true：真实标签
    y_score：模型预测分数
    pos_label：正样本标签，如“1”
    """
    # 统计正样本和负样本的个数
    num_positive_examples = 0
    for true in y_true:
        if true == pos_label:
            num_positive_examples += 1
    num_negative_examples = len(y_true) - num_positive_examples

    tp, fp = 0, 0
    tpr, fpr, thresholds = [], [], []
    score = max(y_score) + 1

    # 根据排序后的预测分数分别计算fpr和tpr
    for i in np.flip(np.argsort(y_score)):
        # 处理样本预测分数相同的情况
        if y_score[i] != score:
            fpr.append(fp / num_negative_examples)
            tpr.append(tp / num_positive_examples)
            thresholds.append(score)
            score = y_score[i]

        if y_true[i] == pos_label:
            tp += 1
        else:
            fp += 1

    fpr.append(fp / num_negative_examples)
    tpr.append(tp / num_positive_examples)
    thresholds.append(score)

    return fpr, tpr, thresholds


def ci(y_true, y_scores, test_path):
    # 计算原始数据的AUC和ROC曲线
    roc_auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    # 使用bootstrapping方法计算标准误差和置信区间
    n_bootstraps = 2000
    rng_seed = 42  # 固定随机数种子以获得可重复结果
    bootstrapped_scores = []
    bootstrapped_tprs = []

    rng = np.random.RandomState(rng_seed)
    mean_fpr = np.linspace(0, 1, 100)

    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_scores), len(y_scores))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_scores[indices])
        fpr_b, tpr_b, _ = roc_curve(y_true[indices], y_scores[indices])
        bootstrapped_scores.append(score)
        bootstrapped_tprs.append(interp(mean_fpr, fpr_b, tpr_b))
        bootstrapped_tprs[-1][0] = 0.0

    mean_tpr = np.mean(bootstrapped_tprs, axis=0)
    mean_tpr[-1] = 1.0

    # 填充置信区间
    std_tpr = np.std(bootstrapped_tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.figure(figsize=(6, 6))
    auc_upper = auc(mean_fpr, tprs_upper)
    auc_lower = auc(mean_fpr, tprs_lower)
    # 绘制原始数据的ROC曲线
    plt.plot(fpr, tpr, color='darkred', lw=2,
             label=r'ROC curve AUC = %0.3f (95%% CI = %0.3f-%0.3f)' % (
                 roc_auc, auc_lower, auc_upper))
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='darkred', alpha=.2)
    plt.plot([0, 1], [0, 1], color='lightgray', lw=2, linestyle='--')

    # 添加图例和标签
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=18)
    plt.xlabel('False Positive Rate', fontsize=26)
    plt.ylabel('True Positive Rate', fontsize=26)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=22)
    plt.legend(loc='lower right', fontsize=14)
    plt.savefig(
        test_path + "ROC_" +
        datetime.now().strftime("%m.%d_%H.%M.%S.jpg"))
    # 显示图形
    plt.show()


# 测试模型
def test_model(model, test_loader):
    model.eval()
    predictions = []
    all_labels = []
    with torch.no_grad():
        for image, label, lesion in test_loader:
            image = image.to(config.DEVICE)
            label = label.to(config.DEVICE)
            output = model(image)

    all_labels.extend(label.cpu().numpy())
    all_labels = np.array(all_labels)

    softmax_output = F.softmax(output, dim=1)
    prediction = softmax_output[:, 1]
    predictions.extend(prediction.cpu().numpy())

    return all_labels, predictions, lesion


# 绘制混淆矩阵
def plot_confusion_matrix(cm, matrix_path, classes, normalize=False):
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Greens')
    plt.title('Confusion Matrix', fontsize=38)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=22)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=26)
    plt.yticks(tick_marks, classes, fontsize=26, rotation=90, va='center')
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=38)
    plt.xlabel('Predicted Label', fontsize=34, labelpad=10)
    plt.ylabel('True Label', fontsize=34, labelpad=10)
    plt.savefig(
        matrix_path +
        datetime.now().strftime("%m.%d_%H.%M.%S.png"))
    plt.show()


# 绘制 ROC 曲线
def plot_roc_curve(fpr, tpr, roc_auc, ROC_path):
    plt.figure()
    plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve (AUC = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='lightgray', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def main():
    with open(config.data_path_lesion + "/PB(P)/test.txt", "r", encoding='utf8') as f:
        malignant_3 = f.readlines()
    with open(config.data_path_lesion + "/PB(N)/test.txt", "r", encoding='utf8') as f:
        benign_3 = f.readlines()

    for line in benign_3:
        malignant_3.append(line)
    malignant_3 = [config.data_path_lesion + line for line in malignant_3]
    test_ds = Cancer_Dataset(
        annotation_lines=np.array(malignant_3),
        transform=config.test_transforms,
        train=False
    )
    test_loader = DataLoader(
        dataset=test_ds, batch_size=config.TEST_BATCH_SIZE, num_workers=4, shuffle=False, pin_memory=True,
        drop_last=False
    )

    model1 = resnet.ShareEncoderModel(pretrained=True)
    model1.load_state_dict(
        torch.load(
            config.Model_lesion_CL_weight_path + "stage1_xxxxx.pth",
            map_location='cuda:0'), strict=False)
    model4 = resnet.ClassifierModel(model1.shared_encoder).to(config.DEVICE)
    model4.load_state_dict(torch.load(
        config.Model_lesion_CL_weight_path + "stage2_xxxxxx.pth",
        map_location=config.DEVICE), strict=False)

    # 进行测试
    labels, predictions, lesion = test_model(model=model4, test_loader=test_loader)
    with open(config.Model_lesion_CL_Prediction_path + 'Test_Predictions.txt', 'w') as f:
        for loss in predictions:
            f.write(f"{loss}\n")

    # 绘制 ROC 曲线和计算 AUC
    average_list = np.array(predictions)
    ci(labels, average_list, config.Model_lesion_CL_ROC_path)

    # 绘制混淆矩阵
    threshold = 0.5
    binary_output = [1 if prob > threshold else 0 for prob in average_list]
    cm = confusion_matrix(labels, binary_output, labels=[1, 0])
    plot_confusion_matrix(cm, config.Model_lesion_CL_Matrix_path, classes=["Malignant", "Benign"])

    # 计算准确率、精确率、召回率、F1-Measure，并绘制P-R曲线
    standard(labels, binary_output)

    # 输出到result.txt文件中
    result = [f"{path};{label};{pred}" for path, label, pred in zip(lesion, labels, average_list)]
    with open(config.Model_lesion_CL_Prediction_path + 'ImagePath_Label_Prediction.txt', 'w') as f:
        for line in result:
            f.write(line + '\n')


if __name__ == "__main__":
    main()
