import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPDecoder(nn.Module):
    def __init__(self, channel_dims = [96, 192, 384, 768], hidden_dim = 256, out_channels = 1):
        super(MLPDecoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = channel_dims[0], out_channels = hidden_dim, kernel_size = 1, stride = 1, padding = 0)
        self.conv2 = nn.Conv2d(in_channels = channel_dims[1], out_channels = hidden_dim, kernel_size = 1, stride = 1, padding = 0)
        self.conv3 = nn.Conv2d(in_channels = channel_dims[2], out_channels = hidden_dim, kernel_size = 1, stride = 1, padding = 0)
        self.conv4 = nn.Conv2d(in_channels = channel_dims[3], out_channels = hidden_dim, kernel_size = 1, stride = 1, padding = 0)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels = 4 * hidden_dim, out_channels = hidden_dim, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(num_features = hidden_dim),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = hidden_dim, out_channels = out_channels, kernel_size = 1, stride = 1, padding = 0)
        )
        
    def forward(self, multiscale_features):
        stage_0 = self.conv1(multiscale_features[0])
        stage_1 = self.conv2(multiscale_features[1])
        stage_2 = self.conv3(multiscale_features[2])
        stage_3 = self.conv4(multiscale_features[3])
        
        size = stage_0.shape[2:]
        
        upsampled_stage_1 = F.interpolate(stage_1, size = size, mode = 'bilinear', align_corners = False)
        upsampled_stage_2 = F.interpolate(stage_2, size = size, mode = 'bilinear', align_corners = False)
        upsampled_stage_3 = F.interpolate(stage_3, size = size, mode = 'bilinear', align_corners = False)
        
        fused_output = torch.cat([stage_0, upsampled_stage_1, upsampled_stage_2, upsampled_stage_3], dim = 1)
        
        return self.final_conv(fused_output)
    

class BrainTumorModel(nn.Module):
    def __init__(self, num_classes = 4, out_channels = 1, dropout_ratio = 0.1):
        super(BrainTumorModel, self).__init__()
        # Backbone with features_only=True to extract multiscale intermediate features
        self.backbone = timm.create_model('davit_tiny.msft_in1k', pretrained = True, features_only = True)
        
        # Dynamically query the backbone's stage channel dimensions to adapt the decoder and head
        channel_dims = self.backbone.feature_info.channels()
        num_features = channel_dims[-1]
        
        # Segmentation decoder (configured with out_channels parameter)
        self.decoder = MLPDecoder(channel_dims = channel_dims, hidden_dim = 256, out_channels = out_channels)
        
        # Classification head (comparable to native DaViT head structure)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.clf_norm = nn.LayerNorm(num_features)
        self.dropout = nn.Dropout(p = dropout_ratio)
        self.classifier = nn.Linear(in_features = num_features, out_features = num_classes)
        
    def forward(self, x):
        # Extract features from the backbone:
        # Expected outputs: [B, 96, 56, 56], [B, 192, 28, 28], [B, 384, 14, 14], [B, 768, 7, 7]
        features = self.backbone(x)
        
        # 1. Segmentation / Reconstruction Output
        seg_out = self.decoder(features)
        # Upsample output logits to original input resolution
        seg_out = F.interpolate(seg_out, size = x.shape[2:], mode = 'bilinear', align_corners = False)
        
        # 2. Classification Output
        cls_feat = self.global_pool(features[-1])
        cls_feat = cls_feat.flatten(1)
        cls_feat = self.clf_norm(cls_feat)
        cls_feat = self.dropout(cls_feat)
        cls_out = self.classifier(cls_feat)
        
        return cls_out, seg_out