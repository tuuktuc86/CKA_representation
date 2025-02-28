# import torch
# from torch import nn
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms

# # ckatorch 모듈에서 CKA 클래스를 import (이미 ckatorch 패키지가 설치되어 있다고 가정)
# from src.ckatorch import CKA

# # SimpleCNN 모델 정의
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, padding=1),  # conv.0
#             nn.ReLU(),                                  # conv.1
#             nn.MaxPool2d(2)                             # conv.2 (28x28 -> 14x14)
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(32 * 14 * 14, 128),               # fc.0
#             nn.ReLU(),                                  # fc.1
#             nn.Linear(128, 10)                          # fc.2
#         )

#     def forward(self, x):
#         x = self.conv(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x
# # if __name__ == '__main__':
# #     # 모델 로드, 데이터셋 구성, CKA 객체 생성 등 모든 실행 코드를 이 블록 안에 넣어주세요.
# #     # 예:
# #     model_task1.load_state_dict(torch.load("model_task1.pth", map_location=device))
# #     model_task2.load_state_dict(torch.load("model_task2.pth", map_location=device))
    
# #     # 데이터셋 구성 및 dataloader 생성
# #     dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# #     dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    
# #     # CKA 객체 생성 및 CKA 계산
# #     cka_matrix_same = cka_same_model(dataloader, epochs=1)
# #     cka_matrix_different = cka_different_models(dataloader, epochs=1)
    
# #     # 결과 시각화 등
# #     cka_same_model.plot_cka(cka_matrix=cka_matrix_same, title="Model Task1 (Self Comparison)", **plot_parameters)
# #     cka_different_models.plot_cka(cka_matrix=cka_matrix_different, title="SimpleCNN_Task1 vs SimpleCNN_Task2", **plot_parameters)

# # 디바이스 설정 (GPU가 있으면 사용)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # 모델 인스턴스 생성 및 가중치 로드
# model_task1 = SimpleCNN().to(device)
# model_task2 = SimpleCNN().to(device)

# # model_task1.pth: 원본 MNIST로 학습된 모델, model_task2.pth: 180도 회전된 MNIST로 학습된 모델
# model_task1.load_state_dict(torch.load("model_task1.pth", map_location=device))
# model_task2.load_state_dict(torch.load("model_task2.pth", map_location=device))

# # MNIST 데이터셋 구성  
# # [옵션1] 원본 MNIST를 사용: 두 모델에 동일한 입력(원본 이미지)을 제공하여 비교  
# # [옵션2] 혹은 회전된 MNIST를 사용하거나, 두 버전을 모두 포함하는 복합 데이터셋을 구성할 수 있음.
# transform = transforms.Compose([
#     transforms.ToTensor(),  # MNIST 이미지는 [0,1] 범위의 Tensor로 변환
#     # 추가 전처리(예: 정규화)가 필요하면 여기에 추가할 수 있음.
# ])

# # MNIST 테스트 데이터셋 로드
# dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

# # 비교할 레이어 지정 (SimpleCNN의 경우, 예시로 conv의 첫 번째 layer와 fc의 첫 번째 및 마지막 layer)
# # named_modules()로 확인하면, 이름은 "conv.0", "fc.0", "fc.2" 등이 될 것입니다.
# layers = ["conv.0", "fc.0", "fc.2"]

# # CKA 객체 생성
# # 1. 동일 모델 내 비교 (타당성 검사용)
# cka_same_model = CKA(
#     first_model=model_task1,
#     second_model=model_task1,
#     layers=layers,
#     first_name="SimpleCNN_Task1",
#     device=device
# )

# # 2. 서로 다른 모델 비교 (model_task1 vs model_task2)
# cka_different_models = CKA(
#     first_model=model_task1,
#     second_model=model_task2,
#     layers=layers,
#     first_name="SimpleCNN_Task1",
#     second_name="SimpleCNN_Task2",
#     device=device
# )

# # CKA 값 계산 (여기서는 epochs=1로 간단히 실행)
# cka_matrix_same = cka_same_model(dataloader, epochs=1)
# cka_matrix_different = cka_different_models(dataloader, epochs=1)

# # CKA 결과를 시각화 (heatmap 형태)
# plot_parameters = {
#     "use_tight_layout": True,
#     "show_half_heatmap": True,
# }

# cka_same_model.plot_cka(
#     cka_matrix=cka_matrix_same,
#     title="Model Task1 (Self Comparison)",
#     **plot_parameters,
# )

# cka_different_models.plot_cka(
#     cka_matrix=cka_matrix_different,
#     title="SimpleCNN_Task1 vs SimpleCNN_Task2",
#     **plot_parameters,
# )


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    # 모델, 데이터셋, DataLoader 및 CKA 객체 생성 등 모든 코드를 여기에 포함합니다.
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from src.ckatorch import CKA

    # SimpleCNN 모델 정의
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),  # conv.0
                nn.ReLU(),                                  # conv.1
                nn.MaxPool2d(2)                             # conv.2
            )
            self.fc = nn.Sequential(
                nn.Linear(32 * 14 * 14, 128),               # fc.0
                nn.ReLU(),                                  # fc.1
                nn.Linear(128, 10)                          # fc.2
            )

        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
        
    def f_extract(batch, **kwargs):
        # 배치는 (images, labels) 형태이므로, images만 추출합니다.
        images, labels = batch
        return {"x": images}  # "x"는 모델 forward의 인자 이름과 일치해야 합니다.

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_task1 = SimpleCNN().to(device)
    model_task2 = SimpleCNN().to(device)

    # 모델 가중치 로드 (파일 경로에 맞게 수정)
    model_task1.load_state_dict(torch.load("model_task1.pth", map_location=device))
    model_task2.load_state_dict(torch.load("model_task2.pth", map_location=device))

    # MNIST 데이터셋 구성 (여기서는 원본 MNIST 사용)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # num_workers를 0으로 설정하면 multiprocessing 관련 문제를 회피할 수 있음
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    # 비교할 레이어 지정 (예: conv.0, fc.0, fc.2)
    layers = ["conv.0", "fc.0", "fc.2"]

    # CKA 객체 생성
    cka_same_model = CKA(
        first_model=model_task1,
        second_model=model_task1,
        layers=layers,
        first_name="SimpleCNN_Task1",
        device=device
    )

    cka_different_models = CKA(
        first_model=model_task1,
        second_model=model_task2,
        layers=layers,
        first_name="SimpleCNN_Task1",
        second_name="SimpleCNN_Task2",
        device=device
    )

    # # CKA 값 계산 (epochs=1)
    # cka_matrix_same = cka_same_model(dataloader, epochs=1)
    # cka_matrix_different = cka_different_models(dataloader, epochs=1)
    cka_matrix_same = cka_same_model(dataloader, epochs=1, f_extract=f_extract, f_args={})
    cka_matrix_different = cka_different_models(dataloader, epochs=1, f_extract=f_extract, f_args={})

    # CKA 결과 시각화
    plot_parameters = {
        "use_tight_layout": True,
        "show_half_heatmap": True,
    }
    cka_same_model.plot_cka(
        cka_matrix=cka_matrix_same,
        title="Model Task1 (Self Comparison)",
        **plot_parameters,
    )
    cka_different_models.plot_cka(
        cka_matrix=cka_matrix_different,
        title="SimpleCNN_Task1 vs SimpleCNN_Task2",
        **plot_parameters,
    )
