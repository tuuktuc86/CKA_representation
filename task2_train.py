import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np
import json

# --- 모델 정의 (SimpleCNN) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 28x28 -> 14x14
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# --- 각 클래스별로 동일 샘플 수만 추출하는 함수 ---
def get_subset_by_class(dataset, num_samples_per_class):
    targets = np.array(dataset.targets)
    indices = []
    for c in np.unique(targets):
        c_indices = np.where(targets == c)[0]
        if len(c_indices) < num_samples_per_class:
            raise ValueError(f"클래스 {c}에 대해 {num_samples_per_class}개의 샘플을 추출할 수 없습니다.")
        selected = np.random.choice(c_indices, num_samples_per_class, replace=False)
        indices.extend(selected)
    return Subset(dataset, indices)

# --- Training 함수 ---
def train_task(model, device, dataloader, optimizer, criterion, epoch, task_name="Task2"):
    model.train()
    running_loss = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f"{task_name} Epoch {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)}] Loss: {loss.item():.6f}")
    return running_loss / (batch_idx + 1)

# --- 평가 함수 ---
def test(model, device, dataloader, criterion, task_name="Task2"):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
    avg_loss = test_loss / total
    accuracy = 100. * correct / total
    print(f"{task_name} Test - Avg Loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)")
    return avg_loss, accuracy

# --- 메인 함수: Training data만 180° 회전하여 학습하고, Task2 로그/모델 저장 ---
def main(task_samples_per_class=1000, batch_size=64, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    
    # 학습 로그 기록용 딕셔너리 (Task2 이름으로 기록)
    training_history = {"epoch": [], "test_loss": [], "test_acc": []}
    
    # training data에는 180° 회전 transform 적용, test data는 원본 사용
    transform_train = transforms.Compose([
        transforms.Lambda(lambda img: transforms.functional.rotate(img, 180)),
        transforms.ToTensor()
    ])
    transform_test = transforms.ToTensor()
    
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transform_test)
    
    # 각 클래스에서 동일한 수의 샘플 선택
    train_subset = get_subset_by_class(train_dataset, task_samples_per_class)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("=== Task2 Training (Training data rotated 180°) ===")
    for epoch in range(1, epochs + 1):
        avg_train_loss = train_task(model, device, train_loader, optimizer, criterion, epoch, task_name="Task2")
        test_loss, test_acc = test(model, device, test_loader, criterion, task_name="Task2")
        training_history["epoch"].append(epoch)
        training_history["test_loss"].append(test_loss)
        training_history["test_acc"].append(test_acc)
    
    # 모델 저장
    torch.save(model.state_dict(), "model_task2.pth")
    print("Model saved as model_task2.pth")
    
    # 학습 로그 저장
    with open("training_history_task2.json", "w") as f:
        json.dump(training_history, f, indent=4)
    print("Training history saved as training_history_task2.json")

if __name__ == '__main__':
    main(task_samples_per_class=1000, batch_size=64, epochs=5)
