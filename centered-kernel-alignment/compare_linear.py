import torch
from torch import nn
from torch.functional import F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.transforms.functional as TF

from src.ckatorch import CKA

# --- MNIST 모델 정의 ---
class MNISTClassifier(nn.Module):
    def __init__(self) -> None:
        super(MNISTClassifier, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 10)  # 10개 클래스 (0~9)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.relu(self.fc5(x))
        x = self.dropout(x)
        x = self.fc6(x)
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
# 여기서 반환하는 dict의 key를 "x"로 하여, MNISTClassifier의 forward 인자와 일치시킵니다.
class MNISTImagesDict(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.mnist = datasets.MNIST(root=root, train=train, transform=transform, download=download)

    def __getitem__(self, index):
        # 이미지와 라벨 대신, 이미지 데이터만 dict 형태로 반환 (키를 "x"로 변경)
        image, _ = self.mnist[index]
        return {"x": image}

    def __len__(self):
        return len(self.mnist)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 로컬에 저장된 모델 불러오기 ---
    first_many_ff = MNISTClassifier().to(device)
    second_many_ff = MNISTClassifier().to(device)
    first_many_ff.load_state_dict(torch.load("task1_model.pth", map_location=device))
    second_many_ff.load_state_dict(torch.load("task2_model.pth", map_location=device))
    first_many_ff.eval()
    second_many_ff.eval()

    # --- CKA 비교 설정 ---
    # 관찰할 레이어 (여기서는 fc1 ~ fc6)
    layers_to_observe = ["fc1", "fc2", "fc3", "fc4", "fc5", "fc6"]
    shared_parameters = {
        "layers": layers_to_observe,
        "first_name": "MNIST_Model1",
        "device": device,
    }

    cka_same_model = CKA(
        first_model=first_many_ff,
        second_model=first_many_ff,
        **shared_parameters,
    )
    cka_different_models = CKA(
        first_model=first_many_ff,
        second_model=second_many_ff,
        second_name="MNIST_Model2",
        **shared_parameters,
    )

    # --- MNIST 데이터셋 로드 (회전 변환 적용) ---
    # 예시: 테스트 데이터에서 50% 확률로 90도 회전 적용
    rotation_probability = 0  # x%
    rotation_angle = 90         # y도

    transform = transforms.Compose([
        transforms.ToTensor(),
        #RandomRotationWithProbability(angle=rotation_angle, p=rotation_probability),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = MNISTImagesDict(root='./data', train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # --- CKA 계산 및 시각화 ---
    print("| Computing CKA |")
    cka_matrix_same = cka_same_model(dataloader)
    #cka_matrix_different = cka_different_models(dataloader) 여기 일단만 걸어둘게

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
    #cka_different_models.plot_cka(cka_matrix=cka_matrix_different, **plot_parameters)
