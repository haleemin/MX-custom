# File: microxcaling-main/run_mx_cnn.py
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as D

# MX 옵션 파싱
from mx import add_mx_args, get_mx_specs
# 사용자 정의 CNN 모델
from cnn_model import CNN
# 양자화 호출 (modules.py가 torch.nn.Conv2d를 오버라이드하므로 직접 호출은 선택적)
from mx.mx_ops import quantize_mx_op as quantize_tensor

# MXConv2d 래퍼 (modules.py를 사용한다면 이 부분은 생략 가능)
class MXConv2d(nn.Conv2d):
    def __init__(self, *args, mx_specs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mx_specs = mx_specs

    def forward(self, x):
        x = quantize_tensor(x, self.mx_specs, mode="input")
        out = super().forward(x)
        return quantize_tensor(out, self.mx_specs, mode="output")

# 모델 내부 Conv2d→MXConv2d 교체 (modules.py로 전역 교체 시 생략 가능)
def replace_convs_with_mx(module, mx_specs):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d) and not isinstance(child, MXConv2d):
            new_conv = MXConv2d(
                child.in_channels, child.out_channels,
                mx_specs=mx_specs,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                bias=(child.bias is not None)
            )
            new_conv.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new_conv.bias.data.copy_(child.bias.data)
            setattr(module, name, new_conv)
        else:
            replace_convs_with_mx(child, mx_specs)


def main():
    parser = argparse.ArgumentParser(description="MX-Quantized CNN Inference")
    parser = add_mx_args(parser)

    # CNN 생성자 인자
    parser.add_argument("--model_code",  type=str,   default="VGG11",
                        help="VGG11/VGG13/VGG16/VGG19 중 선택")
    parser.add_argument("--in_channels", type=int,   default=3,
                        help="입력 채널 수")
    parser.add_argument("--out_dim",     type=int,   default=10,
                        help="출력 클래스 수")
    parser.add_argument("--act",         type=str,   default="relu",
                        choices=["relu","sigmoid","tanh"],
                        help="활성화 함수 코드")
    parser.add_argument("--use_bn",      action="store_true", default=True,
                        help="BatchNorm 사용 여부")

    # 기타 인자
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="가중치 파일 경로 (없으면 랜덤 초기화)")
    args = parser.parse_args()

    # MX 스펙 생성
    mx_specs = get_mx_specs(args)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 초기화
    model = CNN(
        model_code  = args.model_code,
        in_channels = args.in_channels,
        out_dim     = args.out_dim,
        act         = args.act,
        use_bn      = args.use_bn
    ).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # Conv2d→MXConv2d 교체
    replace_convs_with_mx(model, mx_specs)
    model.eval()

    # 데이터 로드
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = D.CIFAR10(root="./data", train=False, download=True, transform=transform)
    loader  = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    # 추론 및 정확도 계산
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds   = outputs.argmax(dim=1)
            total  += labels.size(0)
            correct += (preds == labels).sum().item()

    print(f"MX-quantized accuracy: {100 * correct / total:.2f}%")

if __name__ == '__main__':
    main()

