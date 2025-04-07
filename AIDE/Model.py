#%%writefile Model.py
import torch.nn as nn
import torch
import open_clip
import numpy as np
from dict import DCTFeatureExtractor
from PIL import Image

class HPF(nn.Module):
    def __init__(self):
        super(HPF, self).__init__()
        # Simplified HPF - single learnable 3x3 filter
        self.hpf = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False, groups=3)
        nn.init.kaiming_normal_(self.hpf.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, input):
        return self.hpf(input)


class ResNetBlock(nn.Module):
    #Simplified ResNet Block
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNetFeatureExtractor(nn.Module):
  def __init__(self, in_channels=3, out_channels=256):
      super().__init__()
      self.layer1 = ResNetBlock(in_channels, 64, stride=2)  # Downsample
      self.layer2 = ResNetBlock(64, 128, stride=2)      # Downsample
      self.layer3 = ResNetBlock(128, out_channels, stride=2) # Downsample
      self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


  def forward(self, x):
      x = self.layer1(x)
      x = self.layer2(x)
      x = self.layer3(x)
      x = self.avgpool(x)  # Global average pooling
      return x.view(x.size(0), -1)


class AIDE_Model(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=True):
        super(AIDE_Model, self).__init__()
        self.num_classes = num_classes
        self.hpf = HPF()
        self.dct_extractor = DCTFeatureExtractor()

        # Use open_clip.create_model_and_transforms consistently
        # self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
        #     "convnext_xxlarge", pretrained='laion2b_s34b_b82k_augreg_soup'
        # )
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            "RN50", pretrained='openai'
        )
        self.clip_model = self.clip_model.visual
        # self.clip_model.head = nn.Identity()  # Remove classification head

        # Freeze CLIP
        # if freeze_backbone:
        #     for param in self.clip_model.parameters():
        #         param.requires_grad = False

        self.resnet_top = ResNetFeatureExtractor()
        self.resnet_bottom = ResNetFeatureExtractor()

        # Feature fusion and classification.
       # self.fc = nn.Sequential(
       #     nn.Linear(3584, 1024),  # 1024 from CLIP, 256 from each ResNet
       #     nn.ReLU(inplace=True),
       #     nn.Linear(1024, num_classes)
       # )

        # self.fc = nn.Sequential(
        #     nn.Linear(1024, 512),  
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, num_classes)
        # )

        self.fc = nn.Sequential(
            nn.Linear(1536, 1024),  
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.num_classes)
        )


    def forward(self, x, original_pil_image=None):
        if len(x.shape) == 4:
            N, C, H, W = x.shape
        else:
            N, C, H, W, _ = x.shape
        
        with torch.no_grad():
            
            clip_mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073])
            clip_mean = clip_mean.to(tokens, non_blocking=True).view(3, 1, 1)
            clip_std = torch.Tensor([0.26862954, 0.26130258, 0.27577711])
            clip_std = clip_std.to(tokens, non_blocking=True).view(3, 1, 1)
            dinov2_mean = torch.Tensor([0.485, 0.456, 0.406]).to(tokens, non_blocking=True).view(3, 1, 1)
            dinov2_std = torch.Tensor([0.229, 0.224, 0.225]).to(tokens, non_blocking=True).view(3, 1, 1)

            local_convnext_image_feats = self.openclip_convnext_xxl(
                tokens * (dinov2_std / clip_std) + (dinov2_mean - clip_mean) / clip_std
            ) #[b, 3072, 8, 8]
            assert local_convnext_image_feats.size()[1:] == (3072, 8, 8)
            local_convnext_image_feats = self.avgpool(local_convnext_image_feats).view(tokens.size(0), -1)
            x_0 = self.convnext_proj(local_convnext_image_feats)
            
               # print(f"CLIP features shape: {clip_features.shape}")  # Keep this for now

        dct_top_patches, dct_bottom_patches = self.dct_extractor(x)
        # print(f"DCT Top Patches shape: {dct_top_patches.shape}")
        # print(f"DCT Bottom Patches shape: {dct_bottom_patches.shape}")

        dct_top_patches = self.hpf(dct_top_patches.view(-1, C, self.dct_extractor.window_size, self.dct_extractor.window_size))
        dct_bottom_patches = self.hpf(dct_bottom_patches.view(-1, C, self.dct_extractor.window_size, self.dct_extractor.window_size))

        dct_top_features = self.resnet_top(dct_top_patches).view(N, -1, 256)
        dct_bottom_features = self.resnet_bottom(dct_bottom_patches).view(N, -1, 256)
        # print(f"DCT Top Features shape: {dct_top_features.shape}")
        # print(f"DCT Bottom Features shape: {dct_bottom_features.shape}")

        dct_top_features = dct_top_features.mean(dim=1)
        dct_bottom_features = dct_bottom_features.mean(dim=1)
        # print(f"DCT Top Features shape (after mean): {dct_top_features.shape}")
        # print(f"DCT Bottom Features shape (after mean): {dct_bottom_features.shape}")

        combined_features = torch.cat([clip_features, dct_top_features, dct_bottom_features], dim=1)
        # print(f"Combined features shape: {combined_features.shape}") # And this!

        logits = self.fc(combined_features)
        return logits
    
