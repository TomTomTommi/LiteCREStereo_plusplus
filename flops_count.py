import sys
sys.path.append('core')
import torch
from thop import profile, clever_format
from core.crestereo import CREStereo_plusplus


if __name__ == '__main__':

    img = torch.randn(1, 3, 544, 960)
    # img = torch.randn(1, 3, 384, 1248)
    img = img.cuda()

    model = CREStereo_plusplus()
    model.cuda()

    macs, params = profile(model, inputs=(img, img))
    macs, params = clever_format([macs, params], "%.3f")  # to be consistent with neapeak

    print("Input size:", img.size())
    print("Macs:", macs)
    print("params:", params)