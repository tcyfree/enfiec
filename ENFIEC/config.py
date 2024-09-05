from torchvision import transforms

# Project Root Path
projectPath = 'Your Root Path'

DEVICE = "cuda:0"
DEVICE_CPU = "cpu"

LEARNING_RATE = 0.00002  # CL-self&self
WEIGHT_DECAY = 0.000001  # 0.000001权重衰减，防止过拟合
temp = 0.1
TRAIN_BATCH_SIZE = 120
TRAIN_NUM_EPOCHS = 80
TEST_BATCH_SIZE = 220

# Data Path
data_path_lesion = projectPath + '/new_data/Lesion Data'

# Lesion Path
Model_lesion_path = projectPath + '/Test_Result/Lesion'
Model_lesion_CL_weight_path = Model_lesion_path + '/Weight/CL/'
Model_lesion_CL_Loss_path = Model_lesion_path + '/Loss/CL/'
Model_lesion_CL_Prediction_path = Model_lesion_path + '/Prediction/CL/'
Model_lesion_CL_ROC_path = Model_lesion_path + '/ROC/CL/'
Model_lesion_CL_Matrix_path = Model_lesion_path + '/Matrix/CL/'

Model_lesion_CE_weight_path = Model_lesion_path + '/Weight/CE/'
Model_lesion_CE_Loss_path = Model_lesion_path + '/Loss/CE/'
Model_lesion_CE_Prediction_path = Model_lesion_path + '/Prediction/CE/'
Model_lesion_CE_ROC_path = Model_lesion_path + '/ROC/CE/'
Model_lesion_CE_Matrix_path = Model_lesion_path + '/Matrix/CE/'


# HeatMap Save Path
heatmap_save_path_lesion = Model_lesion_path + 'HeatMap/'

train_transforms_1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
    transforms.RandomRotation(degrees=45),
    transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=0.1),
    # 随机调整图像的亮度、对比度、饱和度和色调
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

train_transforms_2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])
