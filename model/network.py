import torch
import torchvision
from torch import nn
from model.metric_learning import ArcMarginProduct


class LandmarkNet(nn.Module):
    def __init__(self,
                 args,
                 num_classes):
        super(LandmarkNet, self).__init__()
        self.encoder = get_encoder(args)
        self.loss_module = args.loss_module
        if self.loss_module == 'arcface':
            self.final_layer = ArcMarginProduct(args.features_dim, num_classes,
                                                device=args.device,
                                                s=args.arcface_s,
                                                m=args.arcface_margin,
                                                easy_margin=False,
                                                ls_eps=args.arcface_ls_eps)  # label smoothing
        else:
            self.final_layer = nn.Linear(args.features_dim, num_classes)

    def forward(self, x, labels=None):
        x = self.encoder(x)
        if self.loss_module == 'arcface':
            logits = self.final_layer(x.squeeze(), labels)
        else:
            logits = self.final_layer(x.squeeze())
        return logits


def get_encoder(args):
    """
    Function that returns the selected model as a feature extractor
    """
    if args.arch.startswith("r"):  # It's a ResNet
        if args.arch.startswith("r18"):
            encoder = torchvision.models.resnet18(pretrained=True)
        elif args.arch.startswith("r50"):
            encoder = torchvision.models.resnet50(pretrained=True)
        elif args.arch.startswith("r101"):
            encoder = torchvision.models.resnet101(pretrained=True)
    elif args.arch == "vgg16":
        encoder = torchvision.models.vgg16(pretrained=True)
    # Drop original final linear layer
    encoder = nn.Sequential(*(list(encoder.children())[:-1]))
    # Dynamically obtain number of channels in output
    args.features_dim = get_output_channels_dim(encoder)
    return encoder


def get_output_channels_dim(model):
    """Return the number of channels in the output of a model."""
    return model(torch.ones([1, 3, 224, 224])).shape[1]
