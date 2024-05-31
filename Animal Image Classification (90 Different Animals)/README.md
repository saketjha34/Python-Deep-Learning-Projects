# Animal Image Classification (90 Different Animals)

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Testing](#testing)
- [Deployment](#deployment)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction
This project focuses on classifying images of 90 different animal species using deep learning techniques. The aim is to build a robust image classification model that can accurately identify the species from the given images.

## Project Structure

```
.
├── dataset
│   ├── train
│   ├── test
├── models
│   ├── ModelResNet50.py
├── jupyter notebooks
│   ├── AnimalImageClassificationResNet50.ipynb
├── utils
│   ├── utils.py
│   ├── models.py
├── pytorch saved models
│   ├── AnimalImageClassificationResnet50.pth
├── testing
│   ├── Images/
│   ├── test.ipynb
├── README.md
└── requirements.txt
└── Animal Names.txt
```

## Dataset

The dataset used for training and evaluation includes Animal Images (90 different Animal & Birds)  Details on how to obtain and preprocess the dataset can be found in the `dataset/` directory.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/saketjha34/Python-Deep-Learning-Projects.git
cd Python-Deep-Learning-Projects/Animal Image Classification (90 Different Animals)
pip install -r requirements.txt
```

## Usage

To train the model, run:

```bash
python models/ModelResNet50.py 
```

To view model performances and benchmarks check out:

```bash
python Jupyter Notebooks/AnimalImageClassificationResNet50.ipynb
```

For more detailed instructions, refer to the `utils/utils.py` directory, which contains all the necessary scripts for data preprocessing, training, and evaluation.

## Model Architecture

The model is built from scratch using convolutional neural network ResNet50. The architecture details are available in the `utils/model.py` directory.


### ResNet50

ResNet50 Architecture Overview

ResNet50, a deep convolutional neural network, was introduced by He et al. in 2015 to tackle the vanishing gradient problem. Utilizing residual learning, it incorporates skip connections that allow gradients to flow more smoothly during backpropagation. The architecture comprises an initial convolutional layer, followed by four stages of convolutional blocks with three bottleneck layers each. These blocks include identity and convolutional shortcuts. ResNet50 concludes with a fully connected layer and a softmax activation for classification, making it effective for image classification and object detection tasks.

### References

1. **He, K., Zhang, X., Ren, S., & Sun, J. (2016).** Deep Residual Learning for Image Recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR)* (pp. 770-778). [Paper](https://arxiv.org/abs/1512.03385)

2. **Adaptations and Applications:**
   - **He, K., Zhang, X., Ren, S., & Sun, J. (2016).** Identity Mappings in Deep Residual Networks. In *European Conference on Computer Vision (ECCV)* (pp. 630-645). [Paper](https://arxiv.org/abs/1603.05027)
   - **Hu, J., Shen, L., & Sun, G. (2018).** Squeeze-and-Excitation Networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR)* (pp. 7132-7141). [Paper](https://arxiv.org/abs/1709.01507)

These references provide in-depth insights into the architecture and its enhancements.

## Results

| Model    | Accuracy |
|----------|----------|
| ResNet50 |   90.2%  |

## Testing 
### Testing the Trained Model

The project includes a testing folder designed for evaluating the trained model on custom images. This folder contains:

- **images/**: This subdirectory holds the custom images on which the model will be tested. Each image should represent one of the 90 different animal species that the model is trained to classify.

- **test.ipynb**: This Jupyter Notebook is used to load the trained model and run it on the images in the `images/` folder. It includes code for preprocessing the images, loading the model, making predictions, and displaying the results.

To test the model using your own images:

1. Place your images in the `images/` folder.
2. Open the `test.ipynb` notebook.
3. Follow the instructions in the notebook to run the model and view the classification results.

This setup allows for easy evaluation and demonstration of the model's performance on new, unseen data.

## Deployment

The trained models are saved as `.pth` files in the `pytorch saved models` directory. These files can be used for further deployment purposes. You can load the models in PyTorch using the following code:

```python
import torch
from utils.model import ResNet50

resnet50 = ResNet50(img_channel=3, num_classes=90)
resnet50.load_state_dict(torch.load('pytorch saved models/AnimalImageClassificationResnet50.pth'))
```

<-------------------------------------------xxxxxxxxxxxxx--------------------------------------------->

```python
import torch
import timm
from utils.utils import device

model = timm.create_model('resnet50',
                          pretrained=True,
                          num_classes = 90).to(device)

MODEL_SAVE_PATH = '../pytorch saved model/AnimalImageClassificationResnet50.pth'

loaded_model = model
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH ,map_location=torch.device('cpu')))
loaded_model = loaded_model.to(device)
```

## References

- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. *Advances in Neural Information Processing Systems*, 25, 1097-1105. [PDF](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. *arXiv preprint arXiv:1512.03385*. [PDF](https://arxiv.org/abs/1512.03385)
- Kaggle Dataset : [link](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals)

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
