import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchinfo import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpectrumAttention(nn.Module):
    def __init__(self, in_channels, expansion_ratio=3):
        super(SpectrumAttention, self).__init__()
        self.avg_pool = nn.AvgPool3d(kernel_size=(1, 8, 9), stride=(1, 8, 9), padding=0)
        self.max_pool = nn.MaxPool3d(kernel_size=(1, 8, 9), stride=(1, 8, 9), padding=0)
        self.fc1 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels * expansion_ratio, kernel_size=1)
        self.fc2 = nn.Conv3d(in_channels=in_channels * expansion_ratio, out_channels=in_channels, kernel_size=1)
        self.ReLU = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.ReLU(self.fc2(self.ReLU(self.fc1(self.avg_pool(x)))))
        max_out = self.ReLU(self.fc2(self.ReLU(self.fc1(self.max_pool(x)))))
        out = avg_out + max_out
        attention = self.sigmoid(out)
        return attention

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.avg_pool = nn.AvgPool3d(kernel_size=(5, 1, 1))
        self.max_pool = nn.MaxPool3d(kernel_size=(5, 1, 1))
        self.fc = nn.Conv3d(in_channels=in_channels, out_channels=1, kernel_size=1)
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x.transpose(1, 2)).transpose(1, 2)
        max_out = self.max_pool(x.transpose(1, 2)).transpose(1, 2)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        attention = self.sigmoid(out)
        return attention

class SpectrumSpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3, expansion_ratio=2,spe_spa=True,parallel=False,only=''):
        super(SpectrumSpatialAttention, self).__init__()
        self.spe_spa=spe_spa
        self.parallel=parallel
        self.Only=only
        self.spectrum_attention = SpectrumAttention(in_channels, expansion_ratio)
        self.spatial_attention = SpatialAttention(in_channels, kernel_size)

    def forward(self, x):
        if self.parallel:
            spatial_attention = self.spatial_attention(x)
            spectrum_attention = self.spectrum_attention(x)
            x1 = x * spectrum_attention
            x2 = x * spatial_attention
            x=x1+x2
            return x, spectrum_attention, spatial_attention
        elif self.Only == 'spa':
            spatial_attention = self.spatial_attention(x)
            x2 = x * spatial_attention
            x =x2
            return x, spatial_attention
        elif self.Only == 'spe':
            spectrum_attention = self.spectrum_attention(x)
            x1 = x * spectrum_attention
            x = x1
            return x, spectrum_attention
        else:
            if self.spe_spa:
                spectrum_attention = self.spectrum_attention(x)
                x = x * spectrum_attention
                spatial_attention = self.spatial_attention(x)
                x = x * spatial_attention
            else:
                spatial_attention = self.spatial_attention(x)
                x = x * spatial_attention
                spectrum_attention = self.spectrum_attention(x)
                x = x * spectrum_attention
            return x, spectrum_attention, spatial_attention

class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)

class MultiTimeScaleFusion(nn.Module):
    def __init__(self, in_branches, block, num_blocks, num_channels, time_scales, out_branches):
        super(MultiTimeScaleFusion, self).__init__()
        self.in_branches = in_branches
        self.out_branches = out_branches
        self.branches = nn.ModuleList([
            self._make_branch(block, num_blocks[i], num_channels[i]) for i in range(in_branches)
        ])
        self.fuse_layers = nn.ModuleList([
            nn.ModuleList([
                self._make_fuse_layer(i, j, num_channels[i], num_channels[j], time_scales)
                for j in range(in_branches)
            ])
            for i in range(in_branches)
        ])
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

    def _make_branch(self, block, num_blocks, num_channels):
        layers = [block(num_channels, num_channels) for _ in range(num_blocks)]
        return nn.Sequential(*layers)

    def _make_fuse_layer(self, i, j, num_inchannels, num_outchannels, time_scales):
        if i == j:
            return nn.Identity()
        elif i < j:
            return nn.Sequential(
                nn.Conv3d(num_outchannels, num_inchannels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm3d(num_inchannels)
            )
        else:
            layers = []
            for k in range(i - j):
                layers.append(nn.Conv3d(num_inchannels, num_inchannels, kernel_size=(time_scales[i], 1, 1), stride=1, bias=False))
                layers.append(nn.BatchNorm3d(num_inchannels))
                layers.append(nn.ReLU())
            layers.append(nn.Conv3d(num_outchannels, num_inchannels, kernel_size=1, stride=1, bias=False))
            layers.append(nn.BatchNorm3d(num_inchannels))
            return nn.Sequential(*layers)

    def forward(self, inputs):
        outputs = [branch(x) for branch, x in zip(self.branches, inputs)]
        fused_outputs = []
        for i in range(self.in_branches):
            fused_output = outputs[i]
            for j in range(self.in_branches):
                if i != j:
                    if i < j:
                        scale_factor = (
                            outputs[i].size(2) / outputs[j].size(2),
                            outputs[i].size(3) / outputs[j].size(3),
                            outputs[i].size(4) / outputs[j].size(4)
                        )
                        scale_output = F.interpolate(self.fuse_layers[i][j](outputs[j]), scale_factor=scale_factor, mode='trilinear', align_corners=True)
                    else:
                        scale_output = self.fuse_layers[i][j](outputs[j])
                    fused_output = fused_output + scale_output
            fused_outputs.append(self.LeakyReLU(fused_output))
            if self.out_branches == 1:
                break
        return fused_outputs

class SSAMTSNet(nn.Module):
    def __init__(self, num_classes, dropout_rate,use_attention=True,spe_spa=True,parallel=False,only=''):
        super(SSAMTSNet, self).__init__()
        self.use_Attention=use_attention
        self.parallel=parallel
        self.Spe_spa=spe_spa
        self.Only=only

        # Stage 1 Transition
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(5, 32, kernel_size=(k, 1, 1), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                nn.Dropout3d(dropout_rate),  # Add Dropout3d here
            ) for k in [1, 2]
        ])

        # Stage 1 Fusion
        self.stage1 = MultiTimeScaleFusion(
            in_branches=2,
            block=BasicBlock3D,
            num_blocks=[4, 4],
            num_channels=[32, 32],
            time_scales=[1, 2],
            out_branches=2
        )

        # Stage 2  Transition
        self.transition2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(32, 64, kernel_size=(k, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.Dropout3d(dropout_rate),  # Add Dropout3d here
            ) for k in [1, 1, 2]
        ])

        # Stage 2  Fusion
        self.stage2 = MultiTimeScaleFusion(
            in_branches=3,
            block=BasicBlock3D,
            num_blocks=[4, 4, 4],
            num_channels=[64, 64, 64],
            time_scales=[1, 2, 2],
            out_branches=3
        )

        # Stage 3 Transition
        self.transition3 = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(64, 32, kernel_size=(k, 3, 3), stride=(1, 3, 3), padding=(0, 1, 1), bias=False),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                nn.Dropout3d(dropout_rate),  # Add Dropout3d here
            ) for k in [1, 1, 1, 2]
        ])

        # Stage3 Fusion
        self.stage3 = MultiTimeScaleFusion(
            in_branches=4,
            block=BasicBlock3D,
            num_blocks=[4, 4, 4, 4],
            num_channels=[32, 32, 32, 32],
            time_scales=[1, 2, 2, 2],
            out_branches=4
        )

        # Final Fusion
        self.stage4 = MultiTimeScaleFusion(
            in_branches=4,
            block=BasicBlock3D,
            num_blocks=[4, 4, 4, 4],
            num_channels=[32, 32, 32, 32],
            time_scales=[1, 2, 2, 2],
            out_branches=1
        )
        if self.use_Attention:
            self.Attention = SpectrumSpatialAttention(in_channels=5,spe_spa=self.Spe_spa,parallel=self.parallel,only=self.Only)
        self.lstm = nn.LSTM(input_size=32 * 2 * 2, hidden_size=64, batch_first=True)
        self.Linear1 = nn.Linear(64, num_classes)

        self.dropout = nn.Dropout(dropout_rate)  # Add Dropout here
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x,forward_until_attention=False):
        t_step = x.size(2)

        # 注意力
        if self.use_Attention:
            if self.Only:
                x, _ = self.Attention(x)
            else:
                x, _, _ = self.Attention(x)

        # Stage 1
        branch_outputs = []
        for idx, branch in enumerate(self.transition1):
            if idx != len(self.transition1) - 1:
                branch_outputs.append(branch(x))
            else:
                branch_outputs.append(branch(x))
        x = self.stage1(branch_outputs)

        # Stage 2
        branch_outputs = []
        for idx, branch in enumerate(self.transition2):
            if idx != len(self.transition2) - 1:
                branch_outputs.append(branch(x[idx]))
            else:
                branch_outputs.append(branch(x[-1]))
        x = self.stage2(branch_outputs)

        # Stage 3
        branch_outputs = []
        for idx, branch in enumerate(self.transition3):
            if idx != len(self.transition3) - 1:
                branch_outputs.append(branch(x[idx]))
            else:
                branch_outputs.append(branch(x[-1]))
        x = self.stage3(branch_outputs)

        # Final fusion
        x = self.stage4(x)

        x = x[0].transpose(1, 2).reshape(x[0].size(0), t_step, -1)

        #LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        lstm_out_last = lstm_out[:, -1, :]
        output = self.Linear1(self.dropout(lstm_out_last))  # Add Dropout here
        return output

if __name__ == '__main__':
    model = SSAMTSNet(num_classes=3, dropout_rate=0.1211).to(device)
    X = np.load('../Features/SEED/Segment/X/X89_SelfDE_Si_segmented.npy')[0, 0]
    input_data = torch.tensor(X, dtype=torch.float32).to(device)
    summary(model, input_size=(128,6,8,9,5), col_names=["input_size", "output_size", "num_params"])
