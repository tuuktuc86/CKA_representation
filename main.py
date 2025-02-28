import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np
import json

# -------------------------------
# 1. 모델 정의 (SimpleCNN)
# -------------------------------
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)  # 28x28 -> 14x14
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(32 * 14 * 14, 128),
#             nn.ReLU(),
#             nn.Linear(128, 10)
#         )

#     def forward(self, x):
#         x = self.conv(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x
    
class ExpandedCNN(nn.Module):
    def __init__(self):
        super(ExpandedCNN, self).__init__()
        self.features = nn.Sequential(
            # 첫 번째 컨볼루션 블록
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 출력: 28x28x64
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # 28x28x64
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 14x14x64

            # 두 번째 컨볼루션 블록
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 14x14x128
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),# 14x14x128
            nn.ReLU(),
            nn.MaxPool2d(2)                              # 7x7x128
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# --------------------------------------------------
# 2. 각 클래스별 동일 샘플 수 추출 함수 (get_subset_by_class)
# --------------------------------------------------
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

# --------------------------------------------------
# 3. Task1 학습을 위한 training 함수
# --------------------------------------------------
def train_task(model, device, dataloader, optimizer, criterion, epoch, task_name="Task1"):
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

# --------------------------------------------------
# 4. 평가 함수 (테스트 셋 평가 및 loss, accuracy 반환)
# --------------------------------------------------
def test(model, device, dataloader, criterion, task_name="Task"):
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

# --------------------------------------------------
# 5. Task1 데이터로부터 OGD basis 계산 함수
# --------------------------------------------------
def compute_ogd_basis(model, device, dataloader, criterion, num_batches=10):
    model.eval()  # basis 계산 시에는 모델 고정
    ogd_basis = {}
    # 각 파라미터별로 gradient 벡터를 저장할 리스트 초기화
    for name, param in model.named_parameters():
        ogd_basis[name] = []
    
    batches_used = 0
    for data, target in dataloader:
        if batches_used >= num_batches:
            break
        data, target = data.to(device), target.to(device)
        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_flat = param.grad.detach().view(-1).clone()
                ogd_basis[name].append(grad_flat)
        batches_used += 1

    # 각 파라미터별로 수집된 gradient들의 orthonormal basis 계산 (QR 분해)
    for name in ogd_basis:
        if len(ogd_basis[name]) > 0:
            grads = torch.stack(ogd_basis[name], dim=1)  # shape: (param_size, num_vectors)
            Q, _ = torch.linalg.qr(grads)
            ogd_basis[name] = Q
        else:
            ogd_basis[name] = None
    print("OGD basis computed from Task1 data.")
    return ogd_basis

# --------------------------------------------------
# 6. Task2 학습 (OGD 적용) 함수
# --------------------------------------------------
def train_task2_with_ogd(model, device, dataloader, optimizer, criterion, ogd_basis, epoch, task_name="Task2"):
    model.train()
    running_loss = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # 각 파라미터별로 Task1의 subspace에 대한 gradient 성분 제거 (OGD)
        for name, param in model.named_parameters():
            if param.grad is not None and ogd_basis.get(name) is not None:
                grad_flat = param.grad.view(-1)
                basis = ogd_basis[name].to(grad_flat.device)
                proj = basis @ (basis.t() @ grad_flat)
                new_grad = grad_flat - proj
                param.grad.data.copy_(new_grad.view_as(param.grad))
                
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f"{task_name} Epoch {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)}] Loss: {loss.item():.6f}")
    return running_loss / (batch_idx + 1)

# --------------------------------------------------
# 7. 메인 함수: Task1과 Task2 연속 학습 후 모델 및 로그 저장
# --------------------------------------------------
def main(task1_samples_per_class=1000, task2_samples_per_class=1000, batch_size=64,
         epochs_task1=5, epochs_task2=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    
    # 학습 로그 기록용 딕셔너리
    training_history = {"Task1": {"epoch": [], "test_loss": [], "test_acc": []},
                        "Task2": {"epoch": [], "test_loss": [], "test_acc": []},
                        "Task1_after_Task2": {}}
    
    # ===== Task1: 일반 MNIST 학습 =====
    transform_task1 = transforms.ToTensor()  # data augmentation 없음
    train_dataset_task1 = MNIST(root='./data', train=True, download=True, transform=transform_task1)
    test_dataset_task1 = MNIST(root='./data', train=False, download=True, transform=transform_task1)
    
    train_subset_task1 = get_subset_by_class(train_dataset_task1, task1_samples_per_class)
    train_loader_task1 = DataLoader(train_subset_task1, batch_size=batch_size, shuffle=True)
    test_loader_task1 = DataLoader(test_dataset_task1, batch_size=batch_size, shuffle=False)
    
    model = ExpandedCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("=== Task1 Training ===")
    for epoch in range(1, epochs_task1 + 1):
        avg_train_loss = train_task(model, device, train_loader_task1, optimizer, criterion, epoch, task_name="Task1")
        test_loss, test_acc = test(model, device, test_loader_task1, criterion, task_name="Task1")
        training_history["Task1"]["epoch"].append(epoch)
        training_history["Task1"]["test_loss"].append(test_loss)
        training_history["Task1"]["test_acc"].append(test_acc)
    
    # Task1 학습 후 OGD basis 계산
    ogd_basis = compute_ogd_basis(model, device, train_loader_task1, criterion, num_batches=10)
    
    # ===== Task2: 180도 회전된 MNIST 학습 (OGD 적용) =====
    transform_task2 = transforms.Compose([
        transforms.Lambda(lambda img: transforms.functional.rotate(img, 180)),
        transforms.ToTensor()
    ])
    train_dataset_task2 = MNIST(root='./data', train=True, download=True, transform=transform_task2)
    test_dataset_task2 = MNIST(root='./data', train=False, download=True, transform=transform_task2)
    
    train_subset_task2 = get_subset_by_class(train_dataset_task2, task2_samples_per_class)
    train_loader_task2 = DataLoader(train_subset_task2, batch_size=batch_size, shuffle=True)
    test_loader_task2 = DataLoader(test_dataset_task2, batch_size=batch_size, shuffle=False)
    
    print("=== Task2 Training (with OGD) ===")
    for epoch in range(1, epochs_task2 + 1):
        avg_train_loss = train_task2_with_ogd(model, device, train_loader_task2, optimizer, criterion, ogd_basis, epoch, task_name="Task2")
        test_loss, test_acc = test(model, device, test_loader_task2, criterion, task_name="Task2")
        training_history["Task2"]["epoch"].append(epoch)
        training_history["Task2"]["test_loss"].append(test_loss)
        training_history["Task2"]["test_acc"].append(test_acc)
    
    # Task2 학습 후, Task1 테스트셋에 대해 평가 (forgetting 확인)
    print("=== Evaluating on Task1 test set after Task2 training ===")
    test_loss, test_acc = test(model, device, test_loader_task1, criterion, task_name="Task1_after_Task2")
    training_history["Task1_after_Task2"]["test_loss"] = test_loss
    training_history["Task1_after_Task2"]["test_acc"] = test_acc
    
    # 모델 저장
    torch.save(model.state_dict(), "model_final.pth")
    print("Model saved as model_final.pth")
    
    # 학습 로그 저장 (JSON 파일)
    with open("training_history.json", "w") as f:
        json.dump(training_history, f, indent=4)
    print("Training history saved as training_history.json")

if __name__ == '__main__':
    main(task1_samples_per_class=1000, task2_samples_per_class=1000, batch_size=64,
         epochs_task1=10, epochs_task2=10)
