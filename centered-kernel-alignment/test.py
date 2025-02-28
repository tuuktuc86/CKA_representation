import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# 회전 변환: x 확률(p)로 y 각도만큼 회전
class RandomRotationWithProbability:
    def __init__(self, angle: float, p: float = 1.0):
        self.angle = angle
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            return TF.rotate(img, self.angle)
        return img

# MNIST 분류를 위한 6계층 피드포워드 신경망 (모델 구조)
class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 10)  # 10개 클래스 (0~9)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
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
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return test_loss, accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 사용자 지정 회전: test 데이터의 x%에 대해 y 각도 회전 적용
    rotation_probability = 100  # 예: 50%의 확률로 회전 적용
    rotation_angle = 180         # 예: 90도 회전 (원하는 각도로 변경 가능)

    transform = transforms.Compose([
        transforms.ToTensor(),
        RandomRotationWithProbability(angle=rotation_angle, p=rotation_probability),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # MNIST 테스트 데이터셋 로드 (필요시 다운로드)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 로컬에서 모델 불러오기 (저장된 weight: task1_model.pth)
    model = MNISTClassifier().to(device)
    model.load_state_dict(torch.load("task2_model.pth", map_location=device))

    criterion = nn.CrossEntropyLoss()
    # 테스트 수행 및 정확도 출력
    test(model, device, test_loader, criterion)

if __name__ == '__main__':
    main()
