import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset
import random

# 원본 테스트 셋과 180도 회전된 테스트 셋에서 1:1 비율로 샘플을 추출하기 위한 인덱스 생성 함수
def get_balanced_test_indices(dataset):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    half = len(indices) // 2
    return indices[:half], indices[half:half*2]

# 원본과 회전된 MNIST 테스트 데이터셋을 로드한 후, 1:1 비율로 섞인 DataLoader 생성
def create_test_loaders(batch_size=64):
    # 원본 이미지는 ToTensor만 적용
    transform_orig = transforms.ToTensor()
    # 회전된 이미지는 180도 회전 후 ToTensor 적용
    transform_rot = transforms.Compose([
        transforms.Lambda(lambda img: transforms.functional.rotate(img, 180)),
        transforms.ToTensor()
    ])
    
    # MNIST 테스트 데이터셋 로드 (train=False)
    test_dataset_orig = MNIST(root='./data', train=False, download=True, transform=transform_orig)
    test_dataset_rot = MNIST(root='./data', train=False, download=True, transform=transform_rot)
    
    # 두 데이터셋에서 동일한 수의 샘플을 랜덤하게 추출 (1:1 비율)
    orig_indices, rot_indices = get_balanced_test_indices(test_dataset_orig)
    subset_orig = Subset(test_dataset_orig, orig_indices)
    subset_rot = Subset(test_dataset_rot, rot_indices)
    
    loader_orig = DataLoader(subset_orig, batch_size=batch_size, shuffle=False)
    loader_rot = DataLoader(subset_rot, batch_size=batch_size, shuffle=False)
    
    return loader_orig, loader_rot

# 단일 DataLoader를 사용해 모델 평가 (평균 손실과 정확도 출력)
def test_model(model, device, loader, criterion, description=""):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
    test_loss /= total
    accuracy = 100. * correct / total
    print(f"{description} Test - Avg Loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)")
    return test_loss, accuracy

# 테스트용 main 코드 예제
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 여기에 여러분의 모델을 로드하세요. (아래는 예시로 간단한 CNN 모델입니다.)
    class SimpleCNN(torch.nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2)
            )
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(32 * 14 * 14, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 10)
            )
        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = SimpleCNN().to(device)
    # 모델 파라미터 로드 (예: torch.load('model.pth'))—여기서는 이미 학습된 모델이라고 가정합니다.
    # model.load_state_dict(torch.load('your_trained_model.pth'))
    
    criterion = torch.nn.CrossEntropyLoss()
    
    # 원본과 회전된 테스트 DataLoader 생성 (각각 동일한 수의 이미지 포함)
    loader_orig, loader_rot = create_test_loaders(batch_size=64)
    
    print("=== Original MNIST Test Subset ===")
    test_model(model, device, loader_orig, criterion, description="Original")
    
    print("=== 180° Rotated MNIST Test Subset ===")
    test_model(model, device, loader_rot, criterion, description="Rotated")
