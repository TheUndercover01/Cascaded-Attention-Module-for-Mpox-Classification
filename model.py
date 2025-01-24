import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as transforms

import torch
import torch.nn as nn
import torch.nn.functional as F


class CascadedAttentionModule(nn.Module):
    """
    Multi-head attention module with configurable architecture.
    """

    def __init__(
            self,
            in_channels: int,
            heads: int = 3,
            dim: int = 8,
            head_dim: int = 16
    ):
        super().__init__()

        # Configuration parameters
        self.heads = heads
        self.dim = dim
        self.head_dim = head_dim
        self.scale = dim ** -0.5

        # Initial feature extraction
        self.get_begin = nn.Conv2d(
            in_channels,
            self.heads * self.head_dim,
            kernel_size=3,
            padding=1
        )

        # QKV generation
        self.get_qkv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    self.head_dim,
                    self.head_dim,
                    kernel_size=3,
                    padding=1
                ),
                nn.Conv2d(self.head_dim, 3 * self.dim, kernel_size=1)
            ) for _ in range(2)
        ])

        # Cross-scale feature mixing and projection
        self.mix = nn.Sequential(
            nn.Conv2d(self.dim, self.dim * 3, kernel_size=1),
            nn.ReLU()
        )

        self.proj = nn.Conv2d(
            self.heads * self.dim * 2,
            in_channels,
            kernel_size=1
        )

        self.norm = nn.BatchNorm2d(
            num_features=in_channels,
            affine=False
        )

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute scaled dot-product attention.
        """
        q = q * self.scale
        att_map = q.transpose(-2, -1) @ k
        att_map = att_map.softmax(dim=-1)
        return v @ att_map.transpose(-2, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-head attention module.
        """
        B, C, H, W = x.shape
        x_copy = x

        # Initial feature extraction
        all_heads = self.get_begin(x)
        multi_layer = all_heads.split([self.head_dim] * self.heads, dim=1)

        # Multi-head processing
        all_final = []
        for i in range(self.heads):
            out_all = []
            for op in self.get_qkv:
                # QKV generation
                head_feat = op(multi_layer[i])
                q, k, v = head_feat.split([self.dim, self.dim, self.dim], dim=1)

                # Attention computation
                q, k, v = map(lambda t: t.flatten(2), (q, k, v))
                out = self.attention(q, k, v)
                out = out.view(B, self.dim, H, W)

                # Feature mixing
                out = F.interpolate(out, size=(H, W), mode='bilinear')
                out_all.append(out)

            # Aggregate outputs
            out_all_one = torch.cat(out_all, dim=1)
            all_final.append(out_all_one)

        # Final projection and normalization
        all_concat = torch.cat(all_final, dim=1)
        x_final = self.proj(all_concat) + x_copy
        return self.norm(x_final)



def freeze_features(model):
    for param in model.parameters(): #Freezing all the layer
        param.requires_grad = False





def change_classifier(model, model_name, num_classes=4, dropout=0.5,
                     neurons1=4096, neurons2=1024, neurons3=256, neurons4=512, n_layers=2):
    """
    Change the classifier head of various vision models

    Args:
        model: The base model to modify
        model_name: Name/type of the model to determine input features
        num_classes: Number of output classes
        dropout: Dropout rate
        neurons1-4: Number of neurons in each layer
        n_layers: Number of layers in classifier (1-4)
    """
    # Define input features based on model architecture
    input_features = {
        'resnet50': 2048,
        'deit3_small_patch16_224': 384,
        'swin_base_patch4_window7_224': 1024,
        'mobilenetv3_large_100.ra_in1k': 1280
    }

    in_features = input_features.get(model_name, 3072)  # Default to 3072 if model not found

    # Create the classifier based on number of layers
    if n_layers == 1:
        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, neurons1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(neurons1, num_classes),
            nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
        )

    elif n_layers == 2:
        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, neurons1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(neurons1, neurons2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(neurons2, num_classes),
            nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
        )

    elif n_layers == 3:
        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, neurons1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(neurons1, neurons2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(neurons2, neurons3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(neurons3, num_classes),
            nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
        )

    else:  # 4 layers
        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, neurons1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(neurons1, neurons2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(neurons2, neurons3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(neurons3, neurons4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(neurons4, num_classes),
            nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
        )

    # Determine where to attach the classifier based on model type
    if hasattr(model, 'head'):
        model.head.classifier = classifier

    else:
        raise AttributeError("Model structure not supported. Cannot find classifier or head attribute.")

    return model

    return model

def init_weights(m):  #initilizew the classifier with xavier initializer
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
    model.apply(init_weights)

def unfreeze_classifier(model):
    for param in model.head.parameters(): # allow the classifier tto learn
        param.requires_grad = True