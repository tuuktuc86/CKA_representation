import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.transforms.functional as TF

# e2cnn 임포트 (Equivariant 모델을 위한)
import e2cnn.gspaces as gspaces
import e2cnn.nn as enn

from src.ckatorch import CKA

# --- 1번 모델: Equivariant Steerable CNN (4 conv, 2 fc) ---
class EquivariantMNIST4Conv2FC(nn.Module):
    def __init__(self) -> None:
        super(EquivariantMNIST4Conv2FC, self).__init__()
        # 0°, 90°, 180°, 270°에 대한 discrete rotation group (N=4)
        self.r2_act = gspaces.Rot2dOnR2(N=4)
        # 입력: 1채널, trivial representation
        self.input_type = enn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        
        # conv1: 입력 -> 16 복사 (각 regular repr 차원=4 → 총 16×4 = 64 채널)
        self.conv1 = enn.R2Conv(
            in_type=self.input_type,
            out_type=enn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]),
            kernel_size=3, padding=1, bias=False
        )
        self.bn1 = enn.InnerBatchNorm(self.conv1.out_type)
        self.relu1 = enn.ReLU(self.conv1.out_type, inplace=True)
        
        # conv2: 16 복사 유지 (총 64 채널), 첫 번째 pooling: 28×28 → 14×14
        self.conv2 = enn.R2Conv(
            in_type=self.conv1.out_type,
            out_type=enn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]),
            kernel_size=3, padding=1, bias=False
        )
        self.bn2 = enn.InnerBatchNorm(self.conv2.out_type)
        self.relu2 = enn.ReLU(self.conv2.out_type, inplace=True)
        self.pool1 = enn.PointwiseMaxPool(self.conv2.out_type, kernel_size=2, stride=2)
        
        # conv3: 16 복사 → 32 복사 (총 32×4 = 128 채널), spatial: 14×14
        self.conv3 = enn.R2Conv(
            in_type=self.conv2.out_type,
            out_type=enn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
            kernel_size=3, padding=1, bias=False
        )
        self.bn3 = enn.InnerBatchNorm(self.conv3.out_type)
        self.relu3 = enn.ReLU(self.conv3.out_type, inplace=True)
        
        # conv4: 32 복사 유지 (총 128 채널), 두 번째 pooling: 14×14 → 7×7
        self.conv4 = enn.R2Conv(
            in_type=self.conv3.out_type,
            out_type=enn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
            kernel_size=3, padding=1, bias=False
        )
        self.bn4 = enn.InnerBatchNorm(self.conv4.out_type)
        self.relu4 = enn.ReLU(self.conv4.out_type, inplace=True)
        self.pool2 = enn.PointwiseMaxPool(self.conv4.out_type, kernel_size=2, stride=2)
        
        # Group Pooling: 그룹 equivariant feature map을 일반 텐서로 변환
        self.gpool = enn.GroupPooling(self.conv4.out_type)
        # conv4 출력: 원래 32 복사 (각 regular repr 4차원) → 총 128 채널,
        # 그러나 GroupPooling 후 채널 차원이 축소되어 32가 됨.
        # 공간 크기: 7×7 → fc_in_features = 32×7×7 = 1568.
        self.fc_in_features = 32 * 7 * 7
        self.fc1 = nn.Linear(self.fc_in_features, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 28, 28]
        x = enn.GeometricTensor(x, self.input_type)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool1(x)  # [B, 64, 14, 14]
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool2(x)  # [B, 128, 7, 7] (before group pooling)
        
        x = self.gpool(x)  # 결과: GeometricTensor, shape: [B, 32, 7, 7]
        x = x.tensor.view(x.tensor.size(0), -1)  # Flatten: [B, 1568]
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# --- 2번 모델: 일반 CNN (4 conv, 2 fc) ---
class MNISTCNN(nn.Module):
    def __init__(self) -> None:
        super(MNISTCNN, self).__init__()
        # conv1: [B, 1, 28, 28] -> [B, 32, 28, 28]
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # conv2: [B, 32, 28, 28] -> [B, 32, 28, 28]
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # [B, 32, 14, 14]
        # conv3: [B, 32, 14, 14] -> [B, 64, 14, 14]
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # conv4: [B, 64, 14, 14] -> [B, 64, 14, 14]
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # 두 번째 pooling: [B, 64, 14, 14] -> [B, 64, 7, 7]
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        # FC층: Flatten (64×7×7 = 3136) -> 128 -> 10
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))  # [B, 32, 28, 28]
        x = self.relu(self.conv2(x))  # [B, 32, 28, 28]
        x = self.pool(x)              # [B, 32, 14, 14]
        x = self.relu(self.conv3(x))  # [B, 64, 14, 14]
        x = self.relu(self.conv4(x))  # [B, 64, 14, 14]
        x = self.pool(x)              # [B, 64, 7, 7]
        x = x.view(x.size(0), -1)     # Flatten: [B, 3136]
        x = self.relu(self.fc1(x))    # [B, 128]
        x = self.dropout(x)
        x = self.fc2(x)               # [B, 10]
        return x

# --- 커스텀 Dataset: MNIST 이미지를 dict 형태로 반환 ---
class MNISTImagesDict(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.mnist = datasets.MNIST(root=root, train=train, transform=transform, download=download)
    def __getitem__(self, index):
        image, _ = self.mnist[index]
        return {"x": image}
    def __len__(self):
        return len(self.mnist)

# --- 데이터 전처리: 0°, 90°, 180°, 270° 중 무작위 회전 적용 ---
rotation_transform = transforms.RandomChoice([
    transforms.Lambda(lambda img: TF.rotate(img, angle=0)),
    transforms.Lambda(lambda img: TF.rotate(img, angle=90)),
    transforms.Lambda(lambda img: TF.rotate(img, angle=180)),
    transforms.Lambda(lambda img: TF.rotate(img, angle=270))
])
test_transform = transforms.Compose([
    rotation_transform,
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# MNIST 테스트 데이터셋 (dict 형태 반환)
test_dataset = MNISTImagesDict(root='./data', train=False, transform=test_transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# --- 모델 불러오기 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 1번 모델: Equivariant Steerable CNN, weight: "task_model_e2cnn.pth"
first_model = EquivariantMNIST4Conv2FC().to(device)
first_model.load_state_dict(torch.load("task_model_e2cnn.pth", map_location=device))
first_model.eval()

# 2번 모델: 일반 CNN, weight: "task1_model.pth"
second_model = MNISTCNN().to(device)
second_model.load_state_dict(torch.load("task1_model_cnn.pth", map_location=device))
second_model.eval()

# --- CKA 비교 설정 ---
# 모든 층 비교를 위해 두 모델에서 hook을 걸 층 이름을 모두 지정합니다.
# 예를 들어, Equivariant 모델의 경우: conv1, conv2, conv3, conv4, fc1, fc2
# 일반 CNN의 경우: conv1, conv2, conv3, conv4, fc1, fc2
layers_model = ["conv1", "conv2", "conv3", "conv4", "fc1", "fc2"]
shared_parameters = {
    "layers": layers_model,
    "first_name": "Equiv_MNIST_E2CNN",
    "device": device,
}

cka_same_model = CKA(
    first_model=first_model,
    second_model=first_model,
    **shared_parameters,
)
cka_different_models = CKA(
    first_model=first_model,
    second_model=second_model,
    second_layers=layers_model,  # 모델별 층 이름이 동일하다고 가정
    second_name="MNIST_CNN",
    **shared_parameters,
)

# --- CKA 계산 및 시각화 ---
print("| Computing CKA |")
cka_matrix_same = cka_same_model(test_loader)
cka_matrix_different = cka_different_models(test_loader)

plot_parameters = {
    "show_ticks_labels": True,
    "short_tick_labels_splits": 2,
    "use_tight_layout": True,
    "show_half_heatmap": True,
}
cka_same_model.plot_cka(
    cka_matrix=cka_matrix_same,
    title=f"Model {cka_same_model.first_model_info.name} compared with itself",
    **plot_parameters,
)
cka_different_models.plot_cka(
    cka_matrix=cka_matrix_different,
    title=f"{cka_different_models.first_model_info.name} vs {cka_different_models.second_model_info.name}",
    **plot_parameters,
)
