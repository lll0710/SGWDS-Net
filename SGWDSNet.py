import torch

import torch.nn as nn

import sys

import os


sys.path.append(os.path.dirname(__file__))


from SGWDSNet_blocks import (

    get_norm_layer,

    OutBlock,

    Down,

    GroupAdaptiveSkipGate,

    SGWDSBlock,

    MSASBlock,

    GradientFlowHead

)


class SGWDSNet(nn.Module):


    def __init__(

            self,

            in_channels: int = 1,

            n_channels: int = 32,

            n_classes: int = 1,

            norm_type: str = 'group',

            deep_supervision: bool = False,

            dropout_rate: float = 0.2,

            use_l2_pool: bool = True,

            predict_flow: bool = True,

            predict_cell_prob: bool = False,

            flow_weight: float = 0.5,

            **kwargs

    ):

        super().__init__()

        self.do_ds = deep_supervision

        self.use_l2_pool = use_l2_pool

        self.predict_flow = predict_flow

        self.predict_cell_prob = predict_cell_prob

        self.flow_weight = flow_weight


        n_channels = (n_channels // 8) * 8

        if n_channels < 8:

            n_channels = 8


        self.stem = nn.Sequential(

            nn.Conv2d(in_channels, n_channels // 2, 3, padding=1),

            get_norm_layer(norm_type, n_channels // 2),

            nn.GELU(),

            nn.Conv2d(n_channels // 2, n_channels, 3, padding=1),

            get_norm_layer(norm_type, n_channels),

            nn.GELU()

        )

        self.stem_dropout = nn.Dropout2d(p=dropout_rate * 0.5)


        self.enc_block_0 = nn.Sequential(

            MSASBlock(n_channels, n_channels, norm_type, expansion_ratio=1, use_l2_pool=use_l2_pool),

            MSASBlock(n_channels, n_channels, norm_type, expansion_ratio=2, use_l2_pool=use_l2_pool),

            MSASBlock(n_channels, n_channels, norm_type, expansion_ratio=2, use_l2_pool=use_l2_pool)

        )

        self.enc_dropout_0 = nn.Dropout2d(p=dropout_rate * 0.75)

        self.down_0 = Down(

            n_channels, 2 * n_channels, exp_r=4, kernel_size=7,

            do_res=True, norm_type=norm_type

        )


        self.enc_block_1 = nn.Sequential(

            MSASBlock(2 * n_channels, 2 * n_channels, norm_type, expansion_ratio=2, use_l2_pool=use_l2_pool),

            MSASBlock(2 * n_channels, 2 * n_channels, norm_type, expansion_ratio=2, use_l2_pool=use_l2_pool),

            MSASBlock(2 * n_channels, 2 * n_channels, norm_type, expansion_ratio=2, use_l2_pool=use_l2_pool),

            MSASBlock(2 * n_channels, 2 * n_channels, norm_type, expansion_ratio=2, use_l2_pool=use_l2_pool)

        )

        self.enc_dropout_1 = nn.Dropout2d(p=dropout_rate)

        self.down_1 = Down(

            2 * n_channels, 4 * n_channels, exp_r=4, kernel_size=7,

            do_res=True, norm_type=norm_type

        )


        self.bottleneck = nn.Sequential(

            MSASBlock(4 * n_channels, 4 * n_channels, norm_type, expansion_ratio=2, use_l2_pool=use_l2_pool),

            MSASBlock(4 * n_channels, 4 * n_channels, norm_type, expansion_ratio=2, use_l2_pool=use_l2_pool),

            MSASBlock(4 * n_channels, 4 * n_channels, norm_type, expansion_ratio=2, use_l2_pool=use_l2_pool),

            MSASBlock(4 * n_channels, 4 * n_channels, norm_type, expansion_ratio=1, use_l2_pool=use_l2_pool)

        )

        self.bottleneck_dropout = nn.Dropout2d(p=dropout_rate * 1.5)


        self.up_1 = SGWDSBlock(

            4 * n_channels, 2 * n_channels, scale_factor=2, norm_type=norm_type

        )

        self.adaptive_skip_1 = GroupAdaptiveSkipGate(

            channels=2 * n_channels,

            groups=4,

            bottleneck_channels=4 * n_channels

        )

        self.dec_block_1 = nn.Sequential(

            MSASBlock(2 * n_channels, 2 * n_channels, norm_type, expansion_ratio=2, use_l2_pool=use_l2_pool),

            MSASBlock(2 * n_channels, 2 * n_channels, norm_type, expansion_ratio=2, use_l2_pool=use_l2_pool),

            MSASBlock(2 * n_channels, 2 * n_channels, norm_type, expansion_ratio=2, use_l2_pool=use_l2_pool),

            MSASBlock(2 * n_channels, 2 * n_channels, norm_type, expansion_ratio=2, use_l2_pool=use_l2_pool)

        )


        self.up_0 = SGWDSBlock(

            2 * n_channels, n_channels, scale_factor=2, norm_type=norm_type

        )

        self.adaptive_skip_0 = GroupAdaptiveSkipGate(

            channels=n_channels,

            groups=4,

            bottleneck_channels=4 * n_channels

        )

        self.dec_block_0 = nn.Sequential(

            MSASBlock(n_channels, n_channels, norm_type, expansion_ratio=2, use_l2_pool=use_l2_pool),

            MSASBlock(n_channels, n_channels, norm_type, expansion_ratio=2, use_l2_pool=use_l2_pool),

            MSASBlock(n_channels, n_channels, norm_type, expansion_ratio=2, use_l2_pool=use_l2_pool),

            MSASBlock(n_channels, n_channels, norm_type, expansion_ratio=1, use_l2_pool=use_l2_pool)

        )


        self.out_seg = OutBlock(n_channels, n_classes)


        if predict_flow:

            self.out_flow = GradientFlowHead(

                n_channels,

                hidden_channels=64,

                norm_type=norm_type,

                predict_prob=predict_cell_prob

            )


        if deep_supervision:

            self.out_seg_1 = OutBlock(2 * n_channels, n_classes)

            if predict_flow:

                self.out_flow_1 = GradientFlowHead(

                    2 * n_channels,

                    hidden_channels=32,

                    norm_type=norm_type,

                    predict_prob=predict_cell_prob

                )


    def forward(self, x):

        x = self.stem(x)

        x = self.stem_dropout(x)


        x0 = self.enc_block_0(x)

        x0 = self.enc_dropout_0(x0)

        x = self.down_0(x0)


        x1 = self.enc_block_1(x)

        x1 = self.enc_dropout_1(x1)

        x = self.down_1(x1)


        bottleneck_feat = self.bottleneck(x)

        bottleneck_feat = self.bottleneck_dropout(bottleneck_feat)


        x = self.up_1(bottleneck_feat)

        x = self.adaptive_skip_1(x, x1, bottleneck_feat)

        x = self.dec_block_1(x)


        if self.do_ds:

            x_ds_seg_1 = self.out_seg_1(x)

            if self.predict_flow:

                x_ds_flow_1 = self.out_flow_1(x)


        del x1


        x = self.up_0(x)

        x = self.adaptive_skip_0(x, x0, bottleneck_feat)

        x = self.dec_block_0(x)

        del x0


        output = {'seg': self.out_seg(x)}


        if self.predict_flow:

            flow_out = self.out_flow(x)

            if self.predict_cell_prob:

                flow, prob = flow_out

                output['flow'] = flow

                output['cell_prob'] = prob

            else:

                output['flow'] = flow_out


        if self.do_ds:

            output['seg_ds'] = x_ds_seg_1

            if self.predict_flow:

                if self.predict_cell_prob:

                    output['flow_ds'] = x_ds_flow_1[0]

                    output['cell_prob_ds'] = x_ds_flow_1[1]

                else:

                    output['flow_ds'] = x_ds_flow_1


        return output


def compute_gradient_flow_from_mask(mask):


    from scipy.ndimage import distance_transform_edt

    import numpy as np


    mask = mask.astype(np.float32)


    dist_inside = distance_transform_edt(mask)

    dist_outside = distance_transform_edt(1 - mask)

    dist_combined = dist_inside - dist_outside


    dy = np.gradient(dist_combined, axis=0)

    dx = np.gradient(dist_combined, axis=1)


    magnitude = np.sqrt(dy**2 + dx**2) + 1e-8

    dy = dy / magnitude

    dx = dx / magnitude


    return np.stack([dy, dx]).astype(np.float32)


def flow_to_mask(flow, niter=200, eps=0.01):


    import numpy as np


    if isinstance(flow, torch.Tensor):

        flow = flow.detach().cpu().numpy()


    dy, dx = flow[0], flow[1]

    H, W = dy.shape


    y = np.linspace(0, 1, H)

    x = np.linspace(0, 1, W)

    Y, X = np.meshgrid(y, x, indexing='ij')


    for _ in range(niter):

        y_idx = np.clip((Y * (H - 1)).astype(int), 0, H - 1)

        x_idx = np.clip((X * (W - 1)).astype(int), 0, W - 1)


        grad_y = dy[y_idx, x_idx]

        grad_x = dx[y_idx, x_idx]


        Y_new = Y + grad_y * eps

        X_new = X + grad_x * eps


        Y = np.clip(Y_new, 0, 1)

        X = np.clip(X_new, 0, 1)


    final_y = (Y * (H - 1)).astype(int)

    final_x = (X * (W - 1)).astype(int)


    from scipy.ndimage import label

    positions = final_y * W + final_x

    unique_positions = np.unique(positions)


    mask = np.zeros((H, W), dtype=np.uint8)

    for pos in unique_positions:

        mask[positions == pos] = 1


    return mask


__all__ = [

    'SGWDSNet',

    'compute_gradient_flow_from_mask',

    'flow_to_mask'

]


if __name__ == "__main__":

    print("=" * 60)

    print("SGWDS-Net")

    print("=" * 60)


    model = SGWDSNet(

        in_channels=1,

        n_channels=32,

        n_classes=1,

        predict_flow=True,

        predict_cell_prob=False

    )


    x = torch.randn(2, 1, 256, 256)


    with torch.no_grad():

        out = model(x)


    print(f"\nInput shape: {x.shape}")

    print(f"Output:")

    for k, v in out.items():

        print(f"  {k}: {v.shape}")


    print("\n" + "=" * 60)

    print("Test gradient flow GT generation")

    print("=" * 60)


    import numpy as np

    mask = np.zeros((256, 256), dtype=np.float32)

    mask[100:150, 100:150] = 1


    flow_gt = compute_gradient_flow_from_mask(mask)

    print(f"Mask shape: {mask.shape}")

    print(f"Flow GT shape: {flow_gt.shape}")


    recovered_mask = flow_to_mask(flow_gt)

    print(f"Recovered mask shape: {recovered_mask.shape}")
