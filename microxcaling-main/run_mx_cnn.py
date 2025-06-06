# File: microxcaling-main/run_mx_cnn.py
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as D
import torch.nn.functional as F

# MX 옵션 파싱
from mx import add_mx_args, get_mx_specs
# 사용자 정의 CNN 모델
from cnn_model import CNN
# 양자화 호출 (modules.py가 torch.nn.Conv2d를 오버라이드하므로 직접 호출은 선택적)
from mx.mx_ops import quantize_mx_op 
# MXConv2d 래퍼 (modules.py를 사용한다면 이 부분은 생략 가능)

class MXConv2d(nn.Conv2d):

    _print_counter = 0
    _max_prints    = 3
    
    def __init__(self, *args, mx_specs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mx_specs = mx_specs

    def forward(self, x):
         # 1) 입력 activation 양자화
       # print(x)
        if MXConv2d._print_counter < MXConv2d._max_prints:
            print(f"[MXConv2d #{MXConv2d._print_counter+1}] {self}")
            print(f"  입력 원본   shape={tuple(x.shape)}, min={x.min():.4f}, max={x.max():.4f}")
            print(x)

        x_q = quantize_mx_op(
            x,
            self.mx_specs,
            elem_format=self.mx_specs["a_elem_format"],
            axes=[1]
        )

        if MXConv2d._print_counter < MXConv2d._max_prints:
            print(f"  입력 양자화 shape={tuple(x_q.shape)}, min={x_q.min():.4f}, max={x_q.max():.4f}")
            print(x_q)

      #  print(x_q)
        # 2) weight 양자화
    #    print("W fmt", self.mx_specs["w_elem_format"],
     #   "axes", [0], "block_size", self.mx_specs["block_size"],
      #  "scale_bits", self.mx_specs["scale_bits"])

     #   print(self.weight)
        if MXConv2d._print_counter < MXConv2d._max_prints:
            print(f"  가중치 원본 min={self.weight.min():.4f}, max={self.weight.max():.4f}")
            print(self.weight)
        
        w_q = quantize_mx_op(
            self.weight,
            self.mx_specs,
            elem_format=self.mx_specs["w_elem_format"],
            axes=[0]
        )

        if MXConv2d._print_counter < MXConv2d._max_prints:
            diff_w = (self.weight - w_q).abs().max().item()
            print(f"  가중치 양자화 min={w_q.min():.4f}, max={w_q.max():.4f}, max|ΔW|={diff_w:.4f}")
            print(w_q)

      #  print(w_q)
       # print("  max |W-Wq| =", (self.weight - w_q).abs().max().item())

        # 3) bias 양자화 (bias가 있는 경우만)
        b_q =  quantize_mx_op(
                self.bias,
                self.mx_specs,
                elem_format=self.mx_specs["w_elem_format"],
                axes=[0]
            ) if self.bias is not None else None

             
        out = F.conv2d(
            x_q,
            w_q,
            b_q,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
      #  print(out)
      #  print("????")
        if MXConv2d._print_counter < MXConv2d._max_prints:
            print("──────────────────────────────────────────")
            MXConv2d._print_counter += 1
        
        return quantize_mx_op(
        out, self.mx_specs,
        elem_format=self.mx_specs["a_elem_format"],
        axes=[1],                     # 단일 축
        block_size=self.mx_specs["block_size"]
        ) 

# 모델 내부 Conv2d→MXConv2d 교체 (modules.py로 전역 교체 시 생략 가능)
def replace_convs_with_mx(module, mx_specs):
    for name, child in list(module.named_children()):
         # (1) 방문하는 모듈 정보 출력
    #    print(f"Visiting {name}: {child.__class__.__name__}")
        
        if isinstance(child, nn.Conv2d) and not isinstance(child, MXConv2d):
             # (2) 교체 전 Conv2d 정보 출력
     #       print(f"  → Before replace ({name}) weight.shape={tuple(child.weight.shape)}")
      #      print(f"    weight data:\n{child.weight.data}")
       #     if child.bias is not None:
        #        print(f"    bias data:\n{child.bias.data}")
                
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
          
            # 교체 후 파라미터
       #     print(f"  → After replace ({name}) weight.shape={tuple(new_conv.weight.shape)}")
        #    print(f"    weight data:\n{new_conv.weight.data}")
         #   if new_conv.bias is not None:
          #      print(f"    bias data:\n{new_conv.bias.data}")
            # (3) 교체 후 MXConv2d 정보 출력
            
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
    print(mx_specs)
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
    model = model.to(device)
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

