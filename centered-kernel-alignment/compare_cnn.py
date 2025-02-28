import torch
from torch import nn
from torch.functional import F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.transforms.functional as TF

from src.ckatorch import CKA

# --- MNIST CNN 모델 정의 ---
class MNISTCNN(nn.Module):
    def __init__(self) -> None:
        super(MNISTCNN, self).__init__()
        # 첫 번째 컨볼루션: 입력 채널 1, 출력 채널 32, kernel=3, padding=1 -> (B,32,28,28)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 두 번째 컨볼루션: 입력 32, 출력 32, kernel=3, padding=1 -> (B,32,28,28)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        # 2x2 max pooling: (B,32,28,28) -> (B,32,14,14)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 세 번째 컨볼루션: 입력 32, 출력 64, kernel=3, padding=1 -> (B,64,14,14)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 네 번째 컨볼루션: 입력 64, 출력 64, kernel=3, padding=1 -> (B,64,14,14)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # 또 한번 pooling: (B,64,14,14) -> (B,64,7,7)
        
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

        # 완전연결 레이어: flatten 후 (64*7*7 = 3136) -> 128
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # 출력 레이어: 128 -> 10
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x의 shape: [B, 1, 28, 28]
        x = self.relu(self.conv1(x))    # [B, 32, 28, 28]
        x = self.relu(self.conv2(x))    # [B, 32, 28, 28]
        x = self.pool(x)                # [B, 32, 14, 14]
        x = self.relu(self.conv3(x))    # [B, 64, 14, 14]
        x = self.relu(self.conv4(x))    # [B, 64, 14, 14]
        x = self.pool(x)                # [B, 64, 7, 7]
        x = x.view(x.size(0), -1)       # Flatten: [B, 3136]
        x = self.relu(self.fc1(x))      # [B, 128]
        x = self.dropout(x)
        x = self.fc2(x)                 # [B, 10]
        return x

# --- 회전 변환 (테스트용) ---
# x 확률(p)로 y 각도(angle)만큼 회전시키는 변환
class RandomRotationWithProbability:
    def __init__(self, angle: float, p: float = 1.0):
        self.angle = angle
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            return TF.rotate(img, self.angle)
        return img

# --- MNIST 데이터셋을 dict 형태로 반환하는 커스텀 Dataset ---
# 여기서 반환하는 dict의 key를 "x"로 하여, 모델의 forward 인자와 일치시킵니다.
class MNISTImagesDict(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.mnist = datasets.MNIST(root=root, train=train, transform=transform, download=download)

    def __getitem__(self, index):
        image, _ = self.mnist[index]
        return {"x": image}

    def __len__(self):
        return len(self.mnist)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 로컬에 저장된 모델 불러오기 ---
    first_cnn = MNISTCNN().to(device)
    second_cnn = MNISTCNN().to(device)
    first_cnn.load_state_dict(torch.load("task1_model_cnn.pth", map_location=device))
    second_cnn.load_state_dict(torch.load("task2_model_cnn.pth", map_location=device))
    first_cnn.eval()
    second_cnn.eval()

    # --- CKA 비교 설정 ---
    # 관찰할 레이어 이름을 CNN의 레이어 이름으로 지정합니다.
    # 여기서는 conv1, conv2, conv3, conv4, fc1, fc2를 관찰합니다.
    layers_to_observe = ["conv1", "conv2", "conv3", "conv4", "fc1", "fc2"]
    shared_parameters = {
        "layers": layers_to_observe,
        "first_name": "MNIST_CNN1",
        "device": device,
    }

    cka_same_model = CKA(
        first_model=first_cnn,
        second_model=first_cnn,
        **shared_parameters,
    )
    cka_different_models = CKA(
        first_model=first_cnn,
        second_model=second_cnn,
        second_name="MNIST_CNN2",
        **shared_parameters,
    )

    # --- MNIST 데이터셋 로드 (회전 변환 적용) ---
    # 여기서는 회전 확률을 기본값 0으로 설정 (회전 미적용)
    rotation_probability = 0.0  # 기본값 0 (회전 적용 안 함)
    rotation_angle = 90         # 원하는 회전 각도

    transform = transforms.Compose([
        transforms.ToTensor(),
        # 필요 시 아래 주석 해제해서 회전 변환 적용 가능
        # RandomRotationWithProbability(angle=rotation_angle, p=rotation_probability),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = MNISTImagesDict(root='./data', train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # --- CKA 계산 및 시각화 ---
    print("| Computing CKA |")
    cka_matrix_same = cka_same_model(dataloader)
    cka_matrix_different = cka_different_models(dataloader)  # 필요시 활성화

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
    cka_different_models.plot_cka(cka_matrix=cka_matrix_different, **plot_parameters)
