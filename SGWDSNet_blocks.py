# -----------------------------------------------------------
# SGWDSNet_blocks.py - MSAS / SC2A / SGWDS / GASG / GradientFlowHead
# -----------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import math



def get_norm_layer(norm_type, num_channels, num_groups=8):
    
    if norm_type == 'group':
        return nn.GroupNorm(num_groups, num_channels)
    elif norm_type == 'instance':
        return nn.InstanceNorm2d(num_channels)
    elif norm_type == 'batch':
        return nn.BatchNorm2d(num_channels)
    else:
        raise ValueError(f'unknown norm: {norm_type}')


class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last or channels_first."""
    def __init__(self, normalized_shape, eps=1e-5, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x, dummy_tensor=False):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class OutBlock(nn.Module):
    
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.conv_out = nn.ConvTranspose2d(in_channels, n_classes, kernel_size=1)

    def forward(self, x, dummy_tensor=None):
        return self.conv_out(x)




class MedNeXtBlock(nn.Module):
    

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 exp_r: int = 4,
                 kernel_size: int = 7,
                 do_res: int = True,
                 norm_type: str = 'group',
                 n_groups: int or None = None,
                 grn = False
                 ):
        super().__init__()
        self.do_res = do_res

        self.conv1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = 1,
            padding = kernel_size//2,
            groups = in_channels if n_groups is None else n_groups,
        )

        if norm_type == 'group':
            self.norm = nn.GroupNorm(
                num_groups=in_channels,
                num_channels=in_channels
            )
        elif norm_type == 'layer':
            self.norm = LayerNorm(
                normalized_shape=in_channels,
                data_format='channels_first'
            )

        self.conv2 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = exp_r*in_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )

        self.act = nn.GELU()

        self.grn_enabled = grn
        if grn:
            self.grn_beta = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1), requires_grad=True)
            self.grn_gamma = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1), requires_grad=True)

        self.conv3 = nn.Conv2d(
            in_channels = exp_r*in_channels,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )

    def forward(self, x, dummy_tensor=None):
        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))

        if self.grn_enabled:
            gx = torch.norm(x1, p=2, dim=(-2, -1), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True)+1e-6)
            x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1

        x1 = self.conv3(x1)
        if self.do_res:
            x1 = x + x1
        return x1


class Down(MedNeXtBlock):
    

    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7,
                do_res=False, norm_type='group', grn=False):
        super().__init__(in_channels, out_channels, exp_r, kernel_size,
                        do_res=False, norm_type=norm_type,
                        grn=grn)

        self.resample_do_res = do_res
        if do_res:
            self.res_conv = nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 2
            )

        self.conv1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = 2,
            padding = kernel_size//2,
            groups = in_channels,
        )

    def forward(self, x, dummy_tensor=None):
        x1 = super().forward(x)
        if self.resample_do_res:
            res = self.res_conv(x)
            x1 = x1 + res
        return x1




class SampleAdaptiveSASA(nn.Module):
    

    def __init__(self, channels, gamma=2, b=1, num_scales=4):
        super().__init__()
        self.channels = channels
        self.num_scales = num_scales

        t = int(abs((math.log(channels, 2) + b) / gamma))
        self.kernel_size = t if t % 2 else t + 1

        k = 4
        while k > 1:
            if channels % k == 0:
                break
            k -= 1
        self.k = k if k > 1 else 1
        self.group_channels = channels // self.k

        mid_channels = max(channels // 16, 32)
        self.sample_router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, 1),
            nn.GroupNorm(num_groups=min(8, mid_channels), num_channels=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_scales, 1)
        )

        self.conv_h_3 = nn.Conv2d(self.group_channels, self.group_channels,
                                  (self.kernel_size, 1), padding=(self.kernel_size // 2, 0),
                                  groups=self.group_channels, bias=False)
        self.conv_h_5 = nn.Conv2d(self.group_channels, self.group_channels,
                                  (self.kernel_size + 2, 1), padding=((self.kernel_size + 2) // 2, 0),
                                  groups=self.group_channels, bias=False)
        self.conv_h_7 = nn.Conv2d(self.group_channels, self.group_channels,
                                  (self.kernel_size + 4, 1), padding=((self.kernel_size + 4) // 2, 0),
                                  groups=self.group_channels, bias=False)
        self.conv_h_9 = nn.Conv2d(self.group_channels, self.group_channels,
                                  (self.kernel_size + 6, 1), padding=((self.kernel_size + 6) // 2, 0),
                                  groups=self.group_channels, bias=False)

        self.conv_w_3 = nn.Conv2d(self.group_channels, self.group_channels,
                                  (1, self.kernel_size), padding=(0, self.kernel_size // 2),
                                  groups=self.group_channels, bias=False)
        self.conv_w_5 = nn.Conv2d(self.group_channels, self.group_channels,
                                  (1, self.kernel_size + 2), padding=(0, (self.kernel_size + 2) // 2),
                                  groups=self.group_channels, bias=False)
        self.conv_w_7 = nn.Conv2d(self.group_channels, self.group_channels,
                                  (1, self.kernel_size + 4), padding=(0, (self.kernel_size + 4) // 2),
                                  groups=self.group_channels, bias=False)
        self.conv_w_9 = nn.Conv2d(self.group_channels, self.group_channels,
                                  (1, self.kernel_size + 6), padding=(0, (self.kernel_size + 6) // 2),
                                  groups=self.group_channels, bias=False)

        self.avg_pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.gn = nn.GroupNorm(self.k, channels)
        self.sigmoid = nn.Sigmoid()
        self.temperature = 1.0

    def forward(self, x):
        B, C, H, W = x.shape
        sample_weights = self.sample_router(x)
        sample_weights = F.softmax(sample_weights / self.temperature, dim=1)

        x_h = self.avg_pool_h(x)
        x_w = self.avg_pool_w(x)

        h_outputs, w_outputs = [], []
        for i in range(self.k):
            start = i * self.group_channels
            end = (i + 1) * self.group_channels

            x_h_g = x_h[:, start:end, :, :]
            h_3 = self.conv_h_3(x_h_g)
            h_5 = self.conv_h_5(x_h_g)
            h_7 = self.conv_h_7(x_h_g)
            h_9 = self.conv_h_9(x_h_g)
            h_weighted = (sample_weights[:, 0:1, :, :] * h_3 +
                          sample_weights[:, 1:2, :, :] * h_5 +
                          sample_weights[:, 2:3, :, :] * h_7 +
                          sample_weights[:, 3:4, :, :] * h_9)
            h_outputs.append(h_weighted)

            x_w_g = x_w[:, start:end, :, :]
            w_3 = self.conv_w_3(x_w_g)
            w_5 = self.conv_w_5(x_w_g)
            w_7 = self.conv_w_7(x_w_g)
            w_9 = self.conv_w_9(x_w_g)
            w_weighted = (sample_weights[:, 0:1, :, :] * w_3 +
                          sample_weights[:, 1:2, :, :] * w_5 +
                          sample_weights[:, 2:3, :, :] * w_7 +
                          sample_weights[:, 3:4, :, :] * w_9)
            w_outputs.append(w_weighted)

        h_concat = torch.cat(h_outputs, dim=1)
        w_concat = torch.cat(w_outputs, dim=1)
        h_attn = self.sigmoid(self.gn(h_concat))
        w_attn = self.sigmoid(self.gn(w_concat))
        spatial_weights = h_attn * w_attn

        return x * spatial_weights, sample_weights


class PSVCA(nn.Module):
    

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels
        self.reduction = reduction

        self.progressive_pool = nn.AdaptiveAvgPool2d(7)

        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        channel_weights = self._compute_channel_weights(x)
        return identity + x * channel_weights

    def _compute_channel_weights(self, x):
        x_pooled = self.progressive_pool(x)
        channel_attn = self.channel_mlp(x_pooled)
        channel_attn = self.sigmoid(channel_attn)
        channel_attn = F.interpolate(channel_attn, size=x.shape[2:], mode='bilinear', align_corners=True)
        return channel_attn


class SC2A(nn.Module):
    """Spatial-Channel Cross Attention (2D)"""

    def __init__(self, channels, gamma=2, b=1, reduction_ratio=16):
        super().__init__()
        self.sasa = SampleAdaptiveSASA(channels, gamma=gamma, b=b)
        self.psvca = PSVCA(channels, reduction=reduction_ratio)

    def forward(self, x):
        x, _ = self.sasa(x)
        x = self.psvca(x)
        return x


class TriplePoolAttention(nn.Module):
   

    def __init__(self, channels: int, reduction_ratio: int = 16, use_l2_pool: bool = True):
        super().__init__()
        self.channels = channels
        self.use_l2_pool = use_l2_pool

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        pool_dim = channels * 3 if use_l2_pool else channels * 2

        mid_channels = max(channels // reduction_ratio, 8)
        self.weight_gen = nn.Sequential(
            nn.Conv2d(pool_dim, mid_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 4, kernel_size=1),
            nn.Sigmoid()
        )

        self.sc2a = SC2A(channels // 4, reduction_ratio=reduction_ratio)

    def forward(self, x):
        B, C4, H, W = x.shape

        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)

        if self.use_l2_pool:
            l2_out = torch.norm(x.view(B, C4, -1), p=2, dim=2)
            l2_out = l2_out.view(B, C4, 1, 1)
            pool_cat = torch.cat([avg_out, max_out, l2_out], dim=1)
        else:
            pool_cat = torch.cat([avg_out, max_out], dim=1)

        branch_weights = self.weight_gen(pool_cat)

        x_branches = torch.chunk(x, 4, dim=1)
        w1, w3, w5, w7 = torch.chunk(branch_weights, 4, dim=1)
        x_weighted = (x_branches[0] * w1) + (x_branches[1] * w3) + \
                     (x_branches[2] * w5) + (x_branches[3] * w7)

        sc2a_enhanced = self.sc2a(x_weighted)
        return branch_weights, sc2a_enhanced




class SGWDSBlock(nn.Module):
    

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2,
                 norm_type: str = 'group',
                 groups: int = 4, edge_max_boost: float = 2.0):
        super().__init__()
        self.scale_factor = scale_factor
        self.groups = groups
        self.edge_max_boost = edge_max_boost
        self.in_channels = in_channels
        self.out_channels = out_channels

        offset_channels = groups * 2

        self.base_upsample = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            get_norm_layer(norm_type, in_channels),
            nn.GELU()
        )

        self.residual_offset_gen = nn.Sequential(
            nn.Conv2d(in_channels, max(in_channels // 2, 32), 3, padding=1),
            get_norm_layer(norm_type, max(in_channels // 2, 32)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(in_channels // 2, 32), offset_channels, 3, padding=1)
        )

        self.range_predictor = nn.Sequential(
            nn.Conv2d(in_channels, max(in_channels // 4, 16), 3, padding=1),
            get_norm_layer(norm_type, max(in_channels // 4, 16)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(in_channels // 4, 16), groups, 3, padding=1),
            nn.Sigmoid()
        )

        self.semantic_avg = nn.AdaptiveAvgPool2d(1)
        self.semantic_max = nn.AdaptiveMaxPool2d(1)
        self.semantic_fc = nn.Sequential(
            nn.Conv2d(in_channels * 2, groups, 1),
            nn.Sigmoid()
        )

        for m in self.semantic_fc.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.edge_det_3 = nn.Conv2d(in_channels, groups * 2, 3, padding=1)
        self.edge_det_5 = nn.Conv2d(in_channels, groups * 2, 5, padding=2)
        self.edge_fusion = nn.Conv2d(groups * 4, groups * 2, 1)

        nn.init.zeros_(self.edge_det_3.weight)
        nn.init.zeros_(self.edge_det_3.bias)
        nn.init.zeros_(self.edge_det_5.weight)
        nn.init.zeros_(self.edge_det_5.bias)

        self.group_attention = nn.Sequential(
            nn.Conv2d(in_channels, groups, 1),
            nn.Softmax(dim=1)
        )

        self.feature_reconstruct = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            get_norm_layer(norm_type, out_channels),
            nn.GELU()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        H_up, W_up = H * self.scale_factor, W * self.scale_factor

        x_up = F.interpolate(x, size=(H_up, W_up), mode='bilinear', align_corners=True)

        x_base = self.base_upsample(x_up)

        residual_offsets = self.residual_offset_gen(x_base)
        range_mask = self.range_predictor(x_base)

        semantic_avg = self.semantic_avg(x_base)
        semantic_max = self.semantic_max(x_base)
        semantic_concat = torch.cat([semantic_avg, semantic_max], dim=1)
        semantic_weight = self.semantic_fc(semantic_concat)

        edge_3 = self.edge_det_3(x_base)
        edge_5 = self.edge_det_5(x_base)
        edge_concat = torch.cat([edge_3, edge_5], dim=1)
        edge_feat = self.edge_fusion(edge_concat)

        edge_weight = torch.sigmoid(edge_feat)
        edge_weight = 1 + edge_weight * (self.edge_max_boost - 1)
        edge_weight = torch.clamp(edge_weight, 1.0, self.edge_max_boost)

        semantic_weight = semantic_weight.repeat_interleave(2, dim=1)
        range_mask = range_mask.repeat_interleave(2, dim=1)

        max_residual_offset = 0.5 / self.scale_factor
        offsets_normalized = torch.tanh(residual_offsets)

        modulation = range_mask * semantic_weight * edge_weight
        modulated_offsets = offsets_normalized * modulation * max_residual_offset

        grid = self._create_2d_grid(B, H_up, W_up, x.device)
        sampled_features = self._apply_2d_sampling_weighted(
            x_up, grid, modulated_offsets, B, C, H_up, W_up, x_base
        )

        output = self.feature_reconstruct(sampled_features)
        return output

    def _create_2d_grid(self, B, H, W, device):
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        mesh_y, mesh_x = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([mesh_x, mesh_y], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
        return grid

    def _apply_2d_sampling_weighted(self, x, grid, offsets, B, C, H, W, x_base):
        offsets_view = offsets.view(B, self.groups, 2, H, W).permute(0, 3, 4, 1, 2)
        grid_expanded = grid.unsqueeze(3).repeat(1, 1, 1, self.groups, 1)
        adjusted_grid = (grid_expanded + offsets_view).view(B, H, W * self.groups, 2)

        sampled = F.grid_sample(x, adjusted_grid, mode='bilinear',
                                padding_mode='border', align_corners=True)
        sampled = sampled.view(B, C, H, W, self.groups)

        group_weights = self.group_attention(x_base)
        group_weights = group_weights.permute(0, 2, 3, 1)
        group_weights = group_weights.unsqueeze(1)

        sampled_weighted = (sampled * group_weights).sum(dim=-1, keepdim=False)
        return sampled_weighted



class GroupAdaptiveSkipGate(nn.Module):


    def __init__(self, channels: int, groups: int = 4, bottleneck_channels: int = None):
        super().__init__()
        self.groups = groups

        if bottleneck_channels is None:
            bottleneck_channels = channels * 2

        up_group_size = channels // groups
        bn_group_size = bottleneck_channels // groups

        
        self.sub_modules = nn.ModuleList([
            nn.Sequential(
                nn.Linear(bn_group_size * 2 + up_group_size * 2, 1),
                nn.Sigmoid()
            ) for _ in range(groups)
        ])

    def forward(self, upsampled, skip, bottleneck_feat):
        
        B, C, H, W = upsampled.shape

        
        bn_avg = F.adaptive_avg_pool2d(bottleneck_feat, 1).view(B, -1)
        bn_max = F.adaptive_max_pool2d(bottleneck_feat, 1).view(B, -1)
        bn_global = torch.cat([bn_avg, bn_max], dim=1)  

        up_avg = F.adaptive_avg_pool2d(upsampled, 1).view(B, -1)
        up_max = F.adaptive_max_pool2d(upsampled, 1).view(B, -1)
        up_global = torch.cat([up_avg, up_max], dim=1)  

        
        bn_group_size = bottleneck_feat.shape[1] // self.groups
        up_group_size = C // self.groups

        
        proposals = []
        for g in range(self.groups):
            
            bn_avg_g = bn_global[:, g*bn_group_size:(g+1)*bn_group_size]  
            bn_max_g = bn_global[:, bottleneck_feat.shape[1] + g*bn_group_size:bottleneck_feat.shape[1] + (g+1)*bn_group_size]  
            up_avg_g = up_global[:, g*up_group_size:(g+1)*up_group_size]  
            up_max_g = up_global[:, C + g*up_group_size:C + (g+1)*up_group_size]  

            
            input_g = torch.cat([bn_avg_g, bn_max_g, up_avg_g, up_max_g], dim=1)
            proposal_g = self.sub_modules[g](input_g)  
            proposals.append(proposal_g)

        
        proposals = torch.cat(proposals, dim=1)  
        final_ratio = proposals.mean(dim=1)      

        
        if skip.shape[2:] != upsampled.shape[2:]:
            skip = F.interpolate(skip, size=(H, W), mode='bilinear', align_corners=True)

        
        output = upsampled + final_ratio.view(B, 1, 1, 1) * skip

        return output



class MSASLayer(nn.Module):
    

    def __init__(self, channels: int, norm_type: str = 'group',
                 reduction_ratio: int = 16, use_l2_pool: bool = True):
        super().__init__()
        self.channels = channels

        self.branch1 = self._make_branch(channels, 1, norm_type)
        self.branch3 = self._make_branch(channels, 3, norm_type)
        self.branch5 = self._make_branch(channels, 5, norm_type)
        self.branch7 = self._make_branch(channels, 7, norm_type)

        self.triple_pool_attn = TriplePoolAttention(
            channels * 4, reduction_ratio, use_l2_pool=use_l2_pool)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            get_norm_layer(norm_type, channels),
            nn.GELU()
        )

    def _make_branch(self, in_channels, kernel_size, norm_type):
        dilation_rate = {
            1: 1,
            3: 1,
            5: 1,
            7: 2,
            9: 2
        }[kernel_size]

        padding = dilation_rate * (kernel_size // 2)

        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size,
                     padding=padding,
                     dilation=dilation_rate,
                     groups=in_channels,
                     bias=False),
            nn.Conv2d(in_channels, in_channels, 1),
            get_norm_layer(norm_type, in_channels),
            nn.GELU()
        )

    def forward(self, x):
        residual = x

        x1 = self.branch1(x)
        x3 = self.branch3(x)
        x5 = self.branch5(x)
        x7 = self.branch7(x)

        x_cat = torch.cat([x1, x3, x5, x7], dim=1)

        _, sc2a_enhanced = self.triple_pool_attn(x_cat)

        x_out = self.fusion_conv(sc2a_enhanced)

        x_out = x_out + residual

        return x_out


class MSASBlock(nn.Module):
    

    def __init__(self, in_channels: int, out_channels: int,
                 norm_type: str = 'group',
                 use_residual: bool = True, reduction_ratio: int = 16,
                 expansion_ratio: int = 2, use_l2_pool: bool = True):
        super().__init__()
        self.use_residual = use_residual
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion_ratio = expansion_ratio

        expanded_channels = int(in_channels * expansion_ratio)
        expanded_channels = (expanded_channels // 8) * 8

        self.conv_expand = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, 1),
            get_norm_layer(norm_type, expanded_channels),
            nn.GELU()
        )

        self.msas_layer = MSASLayer(
            expanded_channels, norm_type, reduction_ratio, use_l2_pool=use_l2_pool)

        self.conv_reduce = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1),
            get_norm_layer(norm_type, out_channels)
        )

        if use_residual and in_channels != out_channels:
            self.residual_adapter = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                get_norm_layer(norm_type, out_channels)
            )
        else:
            self.residual_adapter = None

    def forward(self, x, dummy=None):
        residual = x

        x = self.conv_expand(x)
        x = self.msas_layer(x)
        x = self.conv_reduce(x)

        if self.use_residual:
            if self.residual_adapter is not None:
                residual = self.residual_adapter(residual)
            x = x + residual

        return x




class GradientFlowHead(nn.Module):
   

    def __init__(self, in_channels: int, hidden_channels: int = 64,
                 norm_type: str = 'group', predict_prob: bool = False):
        super().__init__()
        self.predict_prob = predict_prob
        out_channels = 3 if predict_prob else 2  

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            get_norm_layer(norm_type, hidden_channels),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            get_norm_layer(norm_type, hidden_channels),
            nn.GELU()
        )
        self.output = nn.Conv2d(hidden_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.output(x)

        
        flow = out[:, :2, :, :]  
        flow_norm = torch.sqrt(flow[:, 0:1]**2 + flow[:, 1:2]**2 + 1e-8)
        flow_normalized = flow / flow_norm

        if self.predict_prob:
            prob = torch.sigmoid(out[:, 2:3, :, :])
            return flow_normalized, prob
        else:
            return flow_normalized




__all__ = [
    'get_norm_layer', 'LayerNorm', 'OutBlock',
    'MedNeXtBlock', 'Down',
    'SampleAdaptiveSASA', 'PSVCA', 'SC2A', 'TriplePoolAttention',
    'GroupAdaptiveSkipGate',
    'SGWDSBlock',
    'MSASLayer', 'MSASBlock',
    'GradientFlowHead'
]
