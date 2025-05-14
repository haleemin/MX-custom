import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as D

# ① MX 옵션 파싱
from mx import add_mx_args, get_mx_specs

# ② 여러분의 CNN 모델
from cnn_model import CNN

# ③ MX 양자화에 사용할 함수
from mx.ops import quantize_tensor


# ── 1) MXConv2d 래퍼 ─────────────────────────────────────────
class MXConv2d(nn.Conv2d):
    """
    nn.Conv2d 를 상속하여
    forward() 전후에 quantize_tensor 를 적용합니다.
    """
    def __init__(self, *args, mx_specs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mx_specs = mx_specs

    def forward(self, x):
        # 입력 활성화 양자화
        x = quantize_tensor(x, self.mx_specs, mode="input")
        # 원래 Conv2d 연산
        out = super().forward(x)
        # 출력 활성화 양자화
        return quantize_tensor(out, self.mx_specs, mode="output")


# ── 2) 모델 내부의 모든 nn.Conv2d → MXConv2d 교체 ─────────────
def replace_convs_with_mx(module, mx_specs):
    """
    module 내의 모든 nn.Conv2d 인스턴스를
    MXConv2d(wrapper) 인스턴스로 교체합니다.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            # 기존 Conv2d 파라미터 복사
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


# ── 3) main() 함수 ───────────────────────────────────────────
def main():
    # 3.1) 인자 파싱
    parser = argparse.ArgumentParser(description="MX-Quantized CNN Inference")
    parser = add_mx_args(parser)                               # --w_elem_format 등 MX 옵션 추가
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--checkpoint", type=str, default="cnn.pth")
    args = parser.parse_args()

    # 3.2) MX 스펙 생성
    mx_specs = get_mx_specs(args)

    # 3.3) 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # 3.4) Conv2d → MXConv2d 교체
    replace_convs_with_mx(model, mx_specs)
    model.eval()

    # 3.5) 데이터셋 준비
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = D.CIFAR10(root="./data", train=False, download=True, transform=transform)
    loader  = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    # 3.6) 추론 루프
    correct = 0
    total   = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds   = outputs.argmax(dim=1)
            total  += labels.size(0)
            correct += (preds == labels).sum().item()

    print(f"MX-quantized accuracy: {100 * correct / total:.2f}%")


# ── 4) 스크립트 실행부 ────────────────────────────────────────
if __name__ == "__main__":
    main()
