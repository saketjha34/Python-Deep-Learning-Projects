import torch
import torch.nn as nn
import torchvision.models as models


class ImageStyler(nn.Module):
    """
    A feature extractor model based on VGG19 pretrained network for Neural Style Transfer.

    Attributes:
        chosen_features (list): Layer indices to extract features from.
        model (nn.Sequential): Truncated VGG19 model up to layer 28.
    """
    def __init__(self):
        """
        Initializes the ImageStyler model by loading pretrained VGG19 and selecting layers.
        """
        super(ImageStyler,self).__init__()
        self.chosen_features = ["0", "5", "10", "19", "28"]
        self.model = models.vgg19(pretrained = True ).features[:29]

    def forward(self,x):
        """
        Forward pass to extract features from specific layers of the VGG19 model.

        Args:
            x (Tensor): Input image tensor.

        Returns:
            list: Feature maps from the selected layers.
        """
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            if str(layer_num) in self.chosen_features:
                features.append(x)

        return features



class ContentLoss(nn.Module):
    """
    Computes the Mean Squared Error loss between content features.
    """
    def __init__(self,):
        """
        Initializes the ContentLoss module.
        """
        super(ContentLoss, self).__init__()
    
    def forward(self, input , target):
        """
        Computes the content loss.

        Args:
            input (Tensor): Features from the generated image.
            target (Tensor): Features from the content image.

        Returns:
            Tensor: Scalar content loss.
        """
        loss = nn.functional.mse_loss(input, target)
        return loss


class StyleLoss(nn.Module):
    """
    Computes style loss using the Gram matrix of feature maps.
    """
    def __init__(self,):
        """
        Initializes the StyleLoss module.
        """
        super(StyleLoss, self).__init__()

    def forward(self, input ,target_feature):
        """
        Computes the style loss between input and target style features.

        Args:
            input (Tensor): Features from the generated image.
            target_feature (Tensor): Features from the style image.

        Returns:
            Tensor: Scalar style loss.
        """
        G = self._gram_matrix(input)
        A = self._gram_matrix(target_feature).detach()
        loss = nn.functional.mse_loss(G, A)
        return loss
    
    def _gram_matrix(self,input):
        """
        Computes the Gram matrix from feature maps.

        Args:
            input (Tensor): Feature map tensor of shape (B, C, H, W).

        Returns:
            Tensor: Gram matrix of shape (B*C, B*C).
        """
        batch_size, num_feature_maps, height, width = input.size()
        features = input.view(batch_size * num_feature_maps, height * width)
        G = torch.mm(features, features.t())
        return G
