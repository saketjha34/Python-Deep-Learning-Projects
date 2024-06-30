![generated-images-0059](https://github.com/saketjha34/Python-Deep-Learning-Projects/assets/148564188/d9a419d8-3f2a-4ad5-97a1-5bb52282e21a)# Mapping Satellite Images using Pix2Pix GAN | Image Translation

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Usage](#usage)
- [Results](#results)
- [Model Architecture](#model-architecture)
- [Pix2PixGAN](#pix2pixgan)
- [Deployment](#deployment)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

This project leverages the Pix2Pix GAN (Generative Adversarial Network) model to map satellite images to corresponding map images.
The Pix2Pix GAN is a type of conditional GAN designed for image-to-image translation tasks.

Generated Maps:
![1.](https://github.com/saketjha34/Python-Deep-Learning-Projects/blob/main/Mapping%20Satellite%20Images%20using%20Pix2Pix%20GAN/generated%20maps/generated-image-0003.png)

![2.](https://github.com/saketjha34/Python-Deep-Learning-Projects/blob/main/Mapping%20Satellite%20Images%20using%20Pix2Pix%20GAN/generated%20maps/generated-image-0074.png)

![3.](https://github.com/saketjha34/Python-Deep-Learning-Projects/blob/main/Mapping%20Satellite%20Images%20using%20Pix2Pix%20GAN/generated%20maps/generated-image-0036.png)

![4.](https://github.com/saketjha34/Python-Deep-Learning-Projects/blob/main/Mapping%20Satellite%20Images%20using%20Pix2Pix%20GAN/generated%20maps/generated-image-0035.png)

![5.](https://github.com/saketjha34/Python-Deep-Learning-Projects/blob/main/Mapping%20Satellite%20Images%20using%20Pix2Pix%20GAN/generated%20maps/generated-image-0100.png)

![6.](https://github.com/saketjha34/Python-Deep-Learning-Projects/blob/main/Mapping%20Satellite%20Images%20using%20Pix2Pix%20GAN/generated%20maps/generated-image-0127.png)


## Project Structure

```
.
├── dataset
│   ├── maps/train
│   ├── maps/val
├── utils
│   ├── utils.py
│   ├── model.py
│   ├── config.py
├── pytorch saved models
│   ├── MapPix2PixGAN.pth
├── Generated Images
│   ├── images generated during training and evaluation
├── src
│   ├── train.py
│   ├── dataset.py
├── README.md
├── Pix2PixGAN PAPER.pdf
├── MapsPix2PixGAN.ipynb
└── requirements.txt
```


## Dataset

The dataset consists of paired satellite and map images, with each pair combined into a single 1200x600 pixel image.
The left half (600x600) is the satellite image, and the right half (600x600) is the corresponding map image.
The dataset can be accesed through `dataset` directory

- **Train Folder**: Contains 1096 images.
- **Validation Folder**: Contains 1098 images.
  

## Usage

To train the model on your own custom images.

1. Clone the repository:
    ```bash
    git clone https://github.com/saketjha34/Python-Deep-Learning-Projects.git
    cd Python-Deep-Learning-Projects/Artistic%20Image%20Styling%20(NST)
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv pix2pixvenv
    source pix2pixvenv/bin/activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Configure `config.py` located in the `utils/` directory:
    ```bash
    cd utils/
    ```
    
   ```python
   IMG_SIZE = 512
   IMG_CHANNEL = 3
   BATCH_SIZE = 8
   LEARNING_RATE = 2e-4
   EPOCHS = 200
   L1_LAMBDA = 1000
   NUM_WORKERS = os.cpu_count()
    ```

5. Train the model from the `src/` directory:
    ```bash
    cd ../src/
    ```
    Once the settings and image paths have been configured, run:
    ```bash
    python train.py
    ``` 

For more detailed instructions, refer to the `utils/utils.py` , `utils/config.py` and `utils/model.py` directory, which contains all the necessary scripts for data preprocessing, training, and evaluation.

## Results

During training, 50 images were generated and saved in the `generated maps` folder. These images represent the progression and improvements of the generator model over time.


## Model Architecture

The Pix2Pix GAN consists of two main components: the Generator and the Discriminator.

### Generator
The Generator is designed to convert satellite images into map images. It uses a U-Net architecture, which consists of an encoder-decoder network with skip connections. The encoder extracts features from the input image and progressively reduces its spatial dimensions, while the decoder generates the output image from these features, restoring the spatial dimensions. The skip connections help preserve fine details by connecting corresponding layers in the encoder and decoder.

### Discriminator
The Discriminator is a PatchGAN classifier, which classifies whether each N×N patch in an image is real or fake. It aims to ensure that the generated images are locally realistic. By focusing on small patches rather than the entire image, the PatchGAN discriminator helps in capturing high-frequency structures and textures.

### Training Process
The training process involves the following steps:
1. **Generator Loss**: The generator aims to minimize a combination of adversarial loss and L1 loss. The adversarial loss ensures that the generated images are realistic, while the L1 loss ensures they are close to the ground truth images.
2. **Discriminator Loss**: The discriminator aims to correctly classify real images as real and generated images as fake.

## Pix2PixGAN

The Pix2Pix GAN is a type of conditional Generative Adversarial Network (cGAN) designed for image-to-image translation tasks. Unlike traditional GANs, which generate images from random noise, the Pix2Pix GAN generates images based on an input image, making it highly suitable for tasks such as translating satellite images into map images.

### Generator

The generator in Pix2Pix GAN is based on the U-Net architecture, which is a type of convolutional neural network designed for image segmentation tasks. The U-Net architecture consists of two main parts: an encoder and a decoder, connected by skip connections.

#### Encoder
The encoder is responsible for extracting features from the input image and progressively reducing its spatial dimensions. It consists of a series of convolutional layers, each followed by batch normalization and a LeakyReLU activation function. The encoder's output is a compressed representation of the input image, capturing its essential features.

#### Decoder
The decoder takes the compressed representation produced by the encoder and generates the output image. It consists of a series of transposed convolutional layers (also known as deconvolutional layers), each followed by batch normalization and a ReLU activation function. The decoder progressively increases the spatial dimensions of the feature maps to reconstruct the output image.

#### Skip Connections
Skip connections are added between corresponding layers of the encoder and decoder. These connections help preserve spatial information by allowing high-resolution features from the encoder to be directly passed to the decoder. This is crucial for generating detailed and accurate images.

### Discriminator

The discriminator in Pix2Pix GAN is a PatchGAN, which classifies each N×N patch in an image as real or fake rather than the entire image. This approach ensures that the discriminator focuses on high-frequency details and textures, making it more effective in capturing local structures.

#### Architecture
The PatchGAN discriminator consists of a series of convolutional layers, each followed by batch normalization and a LeakyReLU activation function. The final layer outputs a matrix of probabilities, indicating whether each patch in the input image is real or fake.

### Loss Functions

Pix2Pix GAN uses a combination of adversarial loss and L1 loss to train the generator and discriminator.

#### Adversarial Loss
The adversarial loss encourages the generator to produce realistic images. It is defined as:

\[ \mathcal{L}_{GAN}(G, D) = \mathbb{E}_{x,y} [\log D(x, y)] + \mathbb{E}_{x, z} [\log (1 - D(x, G(x, z)))] \]

where \( G \) is the generator, \( D \) is the discriminator, \( x \) is the input image, \( y \) is the ground truth image, and \( z \) is the noise vector. The generator aims to minimize this loss, while the discriminator aims to maximize it.

#### L1 Loss
The L1 loss ensures that the generated images are close to the ground truth images. It is defined as:

\[ \mathcal{L}_{L1}(G) = \mathbb{E}_{x,y,z} [\| y - G(x, z) \|_1] \]

The total loss for the generator is a weighted sum of the adversarial loss and the L1 loss:

\[ \mathcal{L}_{G} = \mathcal{L}_{GAN}(G, D) + \lambda \mathcal{L}_{L1}(G) \]

where \( \lambda \) is a weight parameter that balances the two losses.

### Training Process

The training process involves the following steps:

1. **Initialize Parameters**: Initialize the weights of the generator and discriminator.
2. **Forward Pass**: For each training iteration, pass a batch of input images through the generator to produce output images.
3. **Compute Losses**:
   - Calculate the adversarial loss and L1 loss for the generator.
   - Calculate the adversarial loss for the discriminator.
4. **Backpropagation**: Perform backpropagation to update the weights of the generator and discriminator based on their respective losses.
5. **Repeat**: Repeat the process for a specified number of epochs or until the generator produces sufficiently realistic images.

### Parameters

Several hyperparameters are crucial for training the Pix2Pix GAN:

- **Batch Size**: The number of images processed in each iteration. Common values are 1, 16, or 32.
- **Learning Rate**: The step size for updating the weights. A typical value is 0.0002.
- **Epochs**: The number of times the entire training dataset is passed through the model. Common values are 100 or 200.
- **Lambda (λ)**: The weight parameter for the L1 loss. A typical value is 100.

### Datasets

The Pix2Pix GAN can be trained on various image-to-image translation datasets. The original Pix2Pix paper demonstrated its effectiveness on several tasks:

- **Cityscapes Dataset**: Contains images of urban street scenes and their corresponding semantic labels.
- **Facade Dataset**: Contains images of building facades and their corresponding architectural labels.
- **Maps Dataset**: Contains pairs of aerial photographs and maps.

For this project, a custom dataset containing paired satellite and map images is used. The dataset consists of 1096 training images and 1098 validation images, each 1200x600 pixels in size, with the left half being the satellite image and the right half being the map image.

### Resources

The original Pix2Pix GAN paper provides detailed insights into the architecture, training process, and various applications of the model. Key resources include:

- **Paper**: [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) by Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros.
- **Implementation**: The paper provides a TensorFlow implementation, which can be found on the authors' [GitHub repository](https://github.com/phillipi/pix2pix).
- **Additional Resources**: Keras documentation and tutorials on GANs, such as the official Keras GAN tutorial, provide further guidance on implementing and training GANs.

## Deployment

The trained image generator is saved as `.pth` file in the `pytorch saved models` directory. These files can be used for further deployment purposes. You can load the models in PyTorch using the following code:

```python
import torch
import torch.nn as nn
from  utils.config import IMAGE_SIZE, IMG_CHANNELS, NUM_WORKERS, BATCH_SIZE , NOISE_DIM, LEARNING_RATE, EPOCHS

class Generator(nn.Module):
    def __init__(self , noise_channels , img_channels):
        super(Generator , self).__init__()

        self.Network = nn.Sequential(

            self._create_block(in_channels=noise_channels, out_channels=512, kernel_size=4, padding=0 , stride=1),
            self._create_block(in_channels=512, out_channels=256, kernel_size=4, padding=1 , stride=2),   
            self._create_block(in_channels=256, out_channels=128, kernel_size=4, padding=1 , stride=2),   
            self._create_block(in_channels=128, out_channels=64, kernel_size=4, padding=1 , stride=2), 

            nn.ConvTranspose2d(in_channels=64, out_channels=img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )


    def _create_block(self,in_channels, out_channels , kernel_size , padding ,stride):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride, padding=padding,bias= False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.Network(x)

# Load Generator model
generator = Generator( noise_channels = NOISE_DIM , img_channels = IMG_CHANNELS )
generator.load_state_dict(torch.load('pytorch saved models/AnimeFaceDCGANs.pth'))
```

## References

- Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks." arXiv preprint arXiv:1511.06434 (2015). [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- Kaggle Dataset [link](https://www.kaggle.com/datasets/splcher/animefacedataset)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! If you have any improvements or new models to add, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for more details.

## Contact

For any questions or suggestions, please open an issue or contact me @ saketjha0324@gmail.com. or [Linkedin](https://www.linkedin.com/in/saketjha34/)

---

Happy coding!
