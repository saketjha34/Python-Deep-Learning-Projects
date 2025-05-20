import torch
from tqdm import tqdm
from utils.config import StyleTransferConfig
StyleTransferConfig = StyleTransferConfig()

def train_model(model : torch.nn.Module,
                generated_img : torch.Tensor,
                original_img : torch.Tensor,
                style_img : torch.Tensor,
                optimizer : torch.optim.Optimizer,
                content_loss_fn : torch.nn.Module,
                style_loss_fn : torch.nn.Module,
                weights : tuple[float , float],
                save_image: callable,
                epochs : int) -> dict[str, list[float]]:
    """
    Trains a neural network model for style transfer, optimizing the generated image to match the style of a given style image while preserving the content of an original image.

    Parameters:
    model (torch.nn.Module): The neural network model used to extract features from images.
    generated_img (torch.Tensor): The image being generated and optimized.
    original_img (torch.Tensor): The original image whose content is to be preserved.
    style_img (torch.Tensor): The style image whose style is to be applied to the generated image.
    optimizer (torch.optim.Optimizer): The optimizer used to update the generated image.
    content_loss_fn (callable): The function used to compute the content loss between the generated image and the original image.
    weights (tuple[float, float]): A tuple containing two weights: the content weight and the style weight.
    style_loss_fn (callable): The function used to compute the style loss between the generated image and the style image.
    epochs (int): The number of epochs for training.

    Returns:
    dict[str, list[float]]: A dictionary containing lists of content loss, style loss, and total loss recorded at each epoch.
    """
    
    results = {
        'content_loss': [],
        'style_loss': [],
        'total_loss': [],
    }

    for epoch in tqdm(range(epochs+1)):

        content_weight = weights[0]
        style_weight = weights[1]

        generated_img_features = model(generated_img)
        original_img_features = model(original_img)
        style_img_features = model(style_img)

        content_loss = 0
        style_loss = 0
        total_loss = 0

        for gen_feature, orig_feature, style_feature in zip(generated_img_features, 
                                                            original_img_features, 
                                                            style_img_features):

            content_loss += content_loss_fn(gen_feature, orig_feature)
            style_loss += style_loss_fn(gen_feature, style_feature)

        total_loss += content_weight * content_loss + style_weight * style_loss
        
        results['content_loss'].append(content_loss.detach().cpu().numpy())
        results['style_loss'].append(style_loss.detach().cpu().numpy())
        results['total_loss'].append(total_loss.detach().cpu().numpy())
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            
            print('------------------ EPOCH {} -------------------'.format(epoch))
            print('Content Loss: {:.6f}, Style Loss: {:.6f}'.format(content_loss, style_loss))
            print('Total Loss: {:.6f}'.format(total_loss))
            print()

            save_image(generated_img , f'output/generated_{StyleTransferConfig.IMG_NAME}.png')

    return results