import torch
import torch.nn as nn
import torchvision.models as models


class ImageStyler(nn.Module):
    def __init__(self):
        super(self,ImageStyler).__init__()
        self.chosen_features = ["0", "5", "10", "19", "28"]
        self.model = models.vgg19(pretrained = True ).features[:29]

    def forward(self,x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            if str(layer_num) in self.chosen_features:
                features.append(x)

        return features


class ContentLoss(nn.Module):
    def __init__(self,):
        super(ContentLoss, self).__init__()
    
    def forward(self, input , target):
        loss = nn.functional.mse_loss(input, target)
        return loss


class StyleLoss(nn.Module):
    def __init__(self,):
        super(StyleLoss, self).__init__()

    def forward(self, input ,target_feature):
        G = self._gram_matrix(input)
        A = self._gram_matrix(target_feature).detach()
        loss = nn.functional.mse_loss(G, A)
        return loss
    
    def _gram_matrix(self,input):
        batch_size, num_feature_maps, height, width = input.size()
        features = input.view(batch_size * num_feature_maps, height * width)
        G = torch.mm(features, features.t())
        return G