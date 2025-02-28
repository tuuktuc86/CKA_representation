import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 데이터 변환: 현재는 180도 회전 (향후 각도 변경 가능)
transform = transforms.Compose([
    # PIL 이미지에 회전 적용 (여기서 각도 변경 가능)
    transforms.Lambda(lambda img: transforms.functional.rotate(img, angle=0)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# MNIST 데이터셋 다운로드 및 로드
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# --- CNN 모델 정의 ---
class MNISTCNN(nn.Module):
    def __init__(self) -> None:
        super(MNISTCNN, self).__init__()
        # 컨볼루션 레이어
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # [B, 32, 28, 28]
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1) # [B, 32, 28, 28]
        self.pool = nn.MaxPool2d(2, 2)                           # [B, 32, 14, 14]
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # [B, 64, 14, 14]
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # [B, 64, 14, 14]
        # 두 번째 pooling: [B, 64, 14, 14] -> [B, 64, 7, 7]
        
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
        # 완전연결(FC) 레이어
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10개 클래스

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x의 shape: [B, 1, 28, 28]
        x = self.relu(self.conv1(x))     # [B, 32, 28, 28]
        x = self.relu(self.conv2(x))     # [B, 32, 28, 28]
        x = self.pool(x)                 # [B, 32, 14, 14]
        x = self.relu(self.conv3(x))     # [B, 64, 14, 14]
        x = self.relu(self.conv4(x))     # [B, 64, 14, 14]
        x = self.pool(x)                 # [B, 64, 7, 7]
        x = x.view(x.size(0), -1)        # Flatten: [B, 3136]
        x = self.relu(self.fc1(x))       # [B, 128]
        x = self.dropout(x)
        x = self.fc2(x)                  # [B, 10]
        return x

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]  Loss: {loss.item():.4f}')
            
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    print(f'==> Epoch: {epoch} Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

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
    print(f'==> Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return test_loss, accuracy

if __name__ == '__main__':
    # GPU 사용 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델, 손실함수, 옵티마이저 초기화 (CNN 모델 사용)
    model = MNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 5
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion)
    
    # 학습 완료 후 모델 weight 저장
    torch.save(model.state_dict(), 'task1_model_cnn.pth')
    print("모델의 weight가 task2_model.pth에 저장되었습니다.")
