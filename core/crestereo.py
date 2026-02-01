import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock
from .extractor import BasicEncoder
from .corr import AGCL
from .DCN import DCNv2PackFlowGuided

autocast = torch.cuda.amp.autocast


class CREStereo_plusplus(nn.Module):
    def __init__(self, max_disp=192, mixed_precision=False, test_mode=False):
        super(CREStereo_plusplus, self).__init__()

        self.max_flow = max_disp
        self.mixed_precision = mixed_precision
        self.test_mode = test_mode

        self.hidden_dim = 128 // 4
        self.context_dim = 128 // 4
        self.dropout = 0

        # feature network and update block
        self.fnet = BasicEncoder(output_dim=256 // 4, norm_fn='instance', dropout=self.dropout)
        self.update_block = BasicUpdateBlock(hidden_dim=self.hidden_dim, cor_planes=4*9, mask_size=4)
        self.deform = DCNv2PackFlowGuided(64, 64, 3, padding=1, deformable_groups=4, max_residue_magnitude=10, pa_frames=2)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def patch_var(self, corr):
        var = torch.var(corr, dim=1, keepdim=True)
        var = 1 - torch.sigmoid(var)
        return var

    def upsample_flow(self, flow, mask, rate=4):
        """ Upsample flow field [H/rate, W/rate, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, rate, rate, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(rate * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, rate * H, rate * W)

    def zero_init_flow(self, fmap):
        N, C, H, W = fmap.shape
        flow_u = torch.zeros([N, 1, H, W], dtype=torch.float)
        flow_v = torch.zeros([N, 1, H, W], dtype=torch.float)
        flow = torch.cat([flow_u, flow_v], dim=1).to(fmap.device)
        return flow

    def forward(self, image1, image2=None, iters=8, flow_init=None):

        if image2 is None:
            image1, image2 = torch.split(image1, [3, 3], dim=1)

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # feature network
        with autocast(enabled=self.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        with autocast(enabled=self.mixed_precision):

            # 1/4 -> 1/8
            # feature
            s_fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            s_fmap2 = F.avg_pool2d(fmap2, 2, stride=2)

            # context
            net, inp = torch.split(fmap1, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
            s_net = F.avg_pool2d(net, 2, stride=2)
            s_inp = F.avg_pool2d(inp, 2, stride=2)

            # 1/4 -> 1/16
            # feature
            ss_fmap1 = F.avg_pool2d(fmap1, 4, stride=4)
            ss_fmap2 = F.avg_pool2d(fmap2, 4, stride=4)

            # context
            ss_net = F.avg_pool2d(net, 4, stride=4)
            ss_inp = F.avg_pool2d(inp, 4, stride=4)

        # Adaptive Group Correlation Layer
        corr_fn = AGCL(fmap1, fmap2, deform=self.deform)
        s_corr_fn = AGCL(s_fmap1, s_fmap2, deform=self.deform)
        ss_corr_fn_att = AGCL(ss_fmap1, ss_fmap2, deform=self.deform)

        # cascaded refinement (1/16 + 1/8 + 1/4)
        flow_predictions = []
        flow = None
        flow_up = None
        ss_var = torch.ones([ss_fmap1.size(0), 1, ss_fmap1.size(2), ss_fmap1.size(3)]).to(ss_fmap1.device)
        s_var = torch.ones([s_fmap1.size(0), 1, s_fmap1.size(2), s_fmap1.size(3)]).to(s_fmap1.device)
        var = torch.ones([fmap1.size(0), 1, fmap1.size(2), fmap1.size(3)]).to(fmap1.device)

        if flow_init is not None:
            scale = fmap1.shape[2] / flow_init.shape[2]
            flow = - scale * F.interpolate(flow_init, size=(fmap1.shape[2], fmap1.shape[3]), mode='bilinear', align_corners=True)
        else:
            # init flow
            ss_flow = self.zero_init_flow(ss_fmap1)

            # Recurrent Update Module
            # RUM: 1/16
            for itr in range(iters//4):
                if itr % 2 == 0:
                    small_patch = False
                else:
                    small_patch = True

                ss_flow = ss_flow.detach()
                out_corrs = ss_corr_fn_att(ss_flow, None, ss_var, small_patch=small_patch, iter_mode=True)
                ss_var = self.patch_var(out_corrs).to(ss_fmap1.device)

                with autocast(enabled=self.mixed_precision):
                    ss_net, up_mask, delta_flow = self.update_block(ss_net, ss_inp, out_corrs, ss_flow)

                ss_flow = ss_flow + delta_flow
                flow = self.upsample_flow(ss_flow, up_mask, rate=4)
                flow_up = - 4 * F.interpolate(flow, size=(4 * flow.shape[2], 4 * flow.shape[3]), mode='bilinear', align_corners=True)
                flow_predictions.append(flow_up)

            scale = s_fmap1.shape[2] / flow.shape[2]
            s_flow = - scale * F.interpolate(flow, size=(s_fmap1.shape[2], s_fmap1.shape[3]), mode='bilinear', align_corners=True)

            # RUM: 1/8
            for itr in range(iters//2):
                if itr % 2 == 0:
                    small_patch = False
                else:
                    small_patch = True

                s_flow = s_flow.detach()
                out_corrs = s_corr_fn(s_flow, None, s_var, small_patch=small_patch, iter_mode=True)
                s_var = self.patch_var(out_corrs).to(s_fmap1.device)

                with autocast(enabled=self.mixed_precision):
                    s_net, up_mask, delta_flow = self.update_block(s_net, s_inp, out_corrs, s_flow)

                s_flow = s_flow + delta_flow
                flow = self.upsample_flow(s_flow, up_mask, rate=4)
                flow_up = - 2 * F.interpolate(flow, size=(2 * flow.shape[2], 2 * flow.shape[3]), mode='bilinear', align_corners=True)
                flow_predictions.append(flow_up)

            scale = fmap1.shape[2] / flow.shape[2]
            flow = - scale * F.interpolate(flow, size=(fmap1.shape[2], fmap1.shape[3]), mode='bilinear', align_corners=True)

        # RUM: 1/4
        for itr in range(6):
            if itr % 2 == 0:
                small_patch = False
            else:
                small_patch = True

            flow = flow.detach()
            out_corrs = corr_fn(flow, None, var, small_patch=small_patch, iter_mode=True)
            var = self.patch_var(out_corrs).to(fmap1.device)

            with autocast(enabled=self.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, out_corrs, flow)

            flow = flow + delta_flow
            flow_up = - self.upsample_flow(flow, up_mask, rate=4)
            flow_predictions.append(flow_up)

        if self.test_mode:
            return flow_up[:,:1]

        return flow_predictions
