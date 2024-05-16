import torch
import torch.nn as nn

from .fusion import make_fusion
from .head import make_head
from .text_net import make_text_net
from .video_net import make_video_net
import numpy as np




class PtTransformer(nn.Module):
    """
    Transformer based model for single-stage sentence grounding
    """
    def __init__(self, opt):
        super().__init__()

        # backbones
        self.text_net = make_text_net(opt['text_net'])
        self.vid_net = make_video_net(opt['vid_net'])

        # thong: clip_mapping
        self.clip_act_fn = nn.ReLU()
        self.clip_alpha_beta_conv = nn.Conv1d(512, 2, kernel_size=3, padding=1)
        self.clip_mapping_conv = nn.Conv1d(512, opt['vid_net']['embd_dim'], kernel_size=3, padding=1)

        # fusion and prediction heads
        self.fusion = make_fusion(opt['fusion'])
        self.cls_head = make_head(opt['cls_head'])
        self.reg_head = make_head(opt['reg_head'])

        # thong: add layernorm
        self.clip_layernorm = nn.LayerNorm(opt['vid_net']['embd_dim'])

    def encode_text(self, tokens, token_masks):
        text, text_masks = self.text_net(tokens, token_masks)
        return text, text_masks

    def encode_video(self, vid, vid_masks):
        fpn, fpn_masks = self.vid_net(vid, vid_masks)
        return fpn, fpn_masks

    def fuse_and_predict(self, fpn, fpn_masks, text, text_masks, text_size=None):
        fused_fpn, fpn_masks = self.fusion(fpn, fpn_masks, text, text_masks, text_size)
        fpn_logits, _ = self.cls_head(fused_fpn, fpn_masks)
        fpn_offsets, fpn_masks = self.reg_head(fused_fpn, fpn_masks)
        return fpn_logits, fpn_offsets, fpn_masks, fused_fpn

    def forward(self, vid, vid_masks, text, text_masks, text_size=None, clip_text_feats=None, clip_vis_feats=None):
        # pack text features
        if text.ndim == 4:
            text = torch.cat([t[:k] for t, k in zip(text, text_size)])
        if text_masks.ndim == 3:
            text_masks = torch.cat(
                [t[:k] for t, k in zip(text_masks, text_size)]
            )
        
        text, text_masks = self.encode_text(text, text_masks)
        fpn, fpn_masks = self.encode_video(vid, vid_masks)

        for i in range(len(fpn)):
            fpn[i] = fpn[i].repeat_interleave(text_size, 0)
            fpn_masks[i] = fpn_masks[i].repeat_interleave(text_size, 0)
            total_len = 0
            for j in range(len(clip_text_feats)):
                for k in range(len(clip_text_feats[j])):
                    clip_feats = torch.cat([clip_text_feats[j][k], clip_vis_feats[j][k].mean(0)[None,:]], 0)
                    clip_feats = torch.mean(clip_feats, 0)[None,:]
                    clip_alpha_beta = self.clip_alpha_beta_conv(clip_feats.transpose(0,1))

                    clip_alpha = clip_alpha_beta[0,:]
                    clip_beta = clip_alpha_beta[1,:]
                    clip_res_feats = clip_alpha * clip_feats + clip_beta
                    clip_feats = self.clip_act_fn(self.clip_mapping_conv(clip_res_feats.transpose(0,1)))

                    fpn[i][total_len + k] = self.clip_layernorm((fpn[i][total_len + k] + 0.1 * clip_feats).transpose(0,1)).transpose(0,1)
                    # fpn[i][total_len + k] = self.clip_layernorm((fpn[i][total_len + k] + 0.5 * clip_feats).transpose(0,1)).transpose(0,1)
                total_len += len(clip_text_feats[j])


        fpn = tuple(fpn)
        fpn_masks = tuple(fpn_masks)
        fpn_logits, fpn_offsets, fpn_masks, fused_fpn = \
            self.fuse_and_predict(fpn, fpn_masks, text, text_masks, text_size)

        return fpn_logits, fpn_offsets, fpn_masks, fused_fpn


class BufferList(nn.Module):

    def __init__(self, buffers):
        super().__init__()

        for i, buf in enumerate(buffers):
            self.register_buffer(str(i), buf, persistent=False)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


class PtGenerator(nn.Module):
    """
    A generator for candidate points from specified FPN levels.
    """
    def __init__(
        self,
        max_seq_len,        # max sequence length
        num_fpn_levels,     # number of feature pyramid levels
        regression_range=4, # normalized regression range
        sigma=1,            # controls overlap between adjacent levels
        use_offset=False,   # whether to align points at the middle of two tics
    ):
        super().__init__()

        self.num_fpn_levels = num_fpn_levels
        assert max_seq_len % 2 ** (self.num_fpn_levels - 1) == 0
        self.max_seq_len = max_seq_len

        # derive regression range for each pyramid level
        self.regression_range = ((0, regression_range), )
        assert sigma > 0 and sigma <= 1
        for l in range(1, self.num_fpn_levels):
            assert regression_range <= max_seq_len
            v_min = regression_range * sigma
            v_max = regression_range * 2
            if l == self.num_fpn_levels - 1:
                v_max = max(v_max, max_seq_len + 1)
            self.regression_range += ((v_min, v_max), )
            regression_range = v_max

        self.use_offset = use_offset

        # generate and buffer all candidate points
        self.buffer_points = self._generate_points()

    def _generate_points(self):
        # tics on the input grid
        tics = torch.arange(0, self.max_seq_len, 1.0)

        points_list = tuple()
        for l in range(self.num_fpn_levels):
            stride = 2 ** l
            points = tics[::stride][:, None]                    # (t, 1)
            if self.use_offset:
                points += 0.5 * stride

            reg_range = torch.as_tensor(
                self.regression_range[l], dtype=torch.float32
            )[None].repeat(len(points), 1)                      # (t, 2)
            stride = torch.as_tensor(
                stride, dtype=torch.float32
            )[None].repeat(len(points), 1)                      # (t, 1)
            points = torch.cat((points, reg_range, stride), 1)  # (t, 4)
            points_list += (points, )

        return BufferList(points_list)

    def forward(self, fpn_n_points):
        """
        Args:
            fpn_n_points (int list [l]): number of points at specified levels.

        Returns:
            fpn_point (float tensor [l * (p, 4)]): candidate points from speficied levels.
        """
        assert len(fpn_n_points) == self.num_fpn_levels

        fpn_points = tuple()
        for n_pts, pts in zip(fpn_n_points, self.buffer_points):
            assert n_pts <= len(pts), (
                'number of requested points {:d} cannot exceed max number '
                'of buffered points {:d}'.format(n_pts, len(pts))
            )
            fpn_points += (pts[:n_pts], )

        return fpn_points