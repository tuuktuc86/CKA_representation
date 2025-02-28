import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

# e2cnn 관련 임포트
import e2cnn.gspaces as gspaces
import e2cnn.nn as enn

# --- 데이터 전처리 ---
# MNIST 테스트 데이터를 0°, 90°, 180°, 270° 중 무작위 회전 적용
rotation_transforms = transforms.RandomChoice([
    transforms.Lambda(lambda img: TF.rotate(img, angle=0)),
    transforms.Lambda(lambda img: TF.rotate(img, angle=90)),
    transforms.Lambda(lambda img: TF.rotate(img, angle=180)),
    transforms.Lambda(lambda img: TF.rotate(img, angle=270)),
])
transform = transforms.Compose([
    rotation_transforms,
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# MNIST 테스트 데이터셋 로드
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader  = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# --- Equivariant Steerable CNN 모델 정의 (4 conv, 2 fc) ---
class EquivariantMNIST4Conv2FC(nn.Module):
    def __init__(self) -> None:
        super(EquivariantMNIST4Conv2FC, self).__init__()
        # 0°, 90°, 180°, 270°에 대한 discrete rotation group (N=4)
        self.r2_act = gspaces.Rot2dOnR2(N=4)
        # 입력: 1채널, trivial representation
        self.input_type = enn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        
        # conv1: 입력 -> 16 복사 (각 복사 regular repr, 총 16×4 = 64 채널)
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
        
        # Group Pooling: 그룹 equivariant feature map → 일반 텐서
        self.gpool = enn.GroupPooling(self.conv4.out_type)
        # conv4 출력: 원래 32 복사 (각 regular repr 4차원) → 총 128 채널,
        # 하지만 GroupPooling으로 채널 차원이 축소되어 32 채널이 됩니다.
        # 공간 크기: 7×7, 따라서 fc_in_features = 32×7×7 = 1568.
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
        x = self.pool2(x)  # [B, 128, 7, 7] before group pooling
        
        x = self.gpool(x)  # 결과: GeometricTensor, shape: [B, 32, 7, 7]
        x = x.tensor.view(x.tensor.size(0), -1)  # Flatten: [B, 32*7*7] = [B, 1568]
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# --- 모델 불러오기 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EquivariantMNIST4Conv2FC().to(device)
model.load_state_dict(torch.load("task_model_e2cnn.pth", map_location=device))
model.eval()

# --- 테스트 함수 ---
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return test_loss, accuracy

criterion = nn.CrossEntropyLoss()

if __name__ == "__main__":
    test(model, device, test_loader, criterion)
