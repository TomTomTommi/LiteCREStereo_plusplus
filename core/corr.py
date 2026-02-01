import torch
import torch.nn.functional as F
from .utils.utils import bilinear_sampler, coords_grid


class AGCL:
    """
    Implementation of Adaptive Group Correlation Layer (AGCL).
    """
    def __init__(self, fmap1, fmap2, deform=None):
        self.fmap1 = fmap1
        self.fmap2 = fmap2

        self.coords = coords_grid(fmap1)
        self.deform = deform

    def __call__(self, flow, extra_offset, var, small_patch=False, iter_mode=False):
        corr = self.corr_v1(self.fmap1, self.fmap2, flow, small_patch, var)
        return corr

    def corr_v1(self, left_feature, right_feature, flow, small_patch, var):

        coords = self.coords + flow
        coords = coords.permute(0, 2, 3, 1)
        right_feature_warp = bilinear_sampler(right_feature, coords)
        if self.deform is not None:
            right_feature_warp = self.deform(right_feature, right_feature_warp, left_feature, flow, var)

        if small_patch:
            psize_list = [(3, 3), (3, 3), (3, 3), (3, 3)]
            dilate_list = [(1, 1), (1, 1), (1, 1), (1, 1)]
        else:
            psize_list = [(1, 9), (1, 9), (1, 9), (1, 9)]
            dilate_list = [(1, 1), (1, 1), (1, 1), (1, 1)]

        N, C, H, W = left_feature.size()
        lefts = torch.split(left_feature, [C // 4] * 4, dim=1)
        rights = torch.split(right_feature_warp, [C // 4] * 4, dim=1)

        corrs = []
        for i in range(len(psize_list)):
            corr = self.get_correlation(lefts[i], rights[i], psize_list[i], dilate_list[i])
            corrs.append(corr)

        final_corr = torch.cat(corrs, dim=1)

        return final_corr

    def get_correlation(self, left_feature, right_feature, psize=(3, 3), dilate=(1, 1)):

        N, C, H, W = left_feature.size()

        di_y, di_x = dilate[0], dilate[1]
        pady, padx = psize[0] // 2 * di_y, psize[1] // 2 * di_x

        right_pad = F.pad(right_feature, [padx, padx, pady, pady], mode='replicate')

        corr_list = []
        for h in range(0, pady * 2 + 1, di_y):
            for w in range(0, padx * 2 + 1, di_x):
                right_crop = right_pad[:, :, h:h + H, w:w + W]
                assert right_crop.size() == left_feature.size()
                corr = (left_feature * right_crop).mean(dim=1, keepdim=True)
                corr_list.append(corr)

        corr_final = torch.cat(corr_list, dim=1)

        return corr_final
