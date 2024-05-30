
# Bone Fracture Classification (Multi-Region)

This repository contains a deep learning project focused on the classification of bone fractures from medical images. The model is designed to identify and classify fractures across multiple regions of interest in X-ray images.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

Bone fractures are a common injury and accurate detection is crucial for effective treatment. This project utilizes deep learning techniques to automate the detection and classification of bone fractures in X-ray images, improving diagnostic accuracy and efficiency.

## Dataset

The dataset used for training and evaluation includes X-ray images with labeled regions indicating the presence of fractures. Details on how to obtain and preprocess the dataset can be found in the `dataset/` directory.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/saketjha34/Python-Deep-Learning-Projects.git
cd Python-Deep-Learning-Projects/Bone\ Fracture\ Classification\(Multi\ Region\)
pip install -r requirements.txt
```

## Usage

To train the model, run:

```bash
python models/ModelAlexNet.py 
python models/ModelResNet18.py 
```

To view model performances and benchmarks check out:
```bash
python Jupyter Notebooks/BoneXRayFractureClassificationAlexNet.ipynb
python Jupyter Notebooks/BoneXRayFractureClassificationResNet18.ipynb
```

For more detailed instructions, refer to the `utils/utils.py` directory, which contains all the necessary scripts for data preprocessing, training, and evaluation.

## Model Architecture

The model is built from scratch using convolutional neural networks like ResNet18 and AlexNet. The architecture details are available in the `utils/models.py` directory.

### AlexNet

AlexNet is a convolutional neural network that is 8 layers deep. It was designed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton and won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2012. AlexNet significantly outperformed previous models and was the first to use ReLU activations and dropout layers to improve performance.

For more details, refer to the original paper: [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).

### ResNet18

ResNet18 is a variant of the ResNet model, which introduced the concept of residual learning. This model has 18 layers and is part of the family of Residual Networks (ResNets) that won the ImageNet competition in 2015. The key innovation in ResNet is the use of skip connections, or shortcuts, to jump over some layers.

## Results

| Model   | Accuracy |
|---------|----------|
| AlexNet | 96%      |
| ResNet18| 97.4%    |

## References

- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. *Advances in Neural Information Processing Systems*, 25, 1097-1105. [PDF](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. *arXiv preprint arXiv:1512.03385*. [PDF](https://arxiv.org/abs/1512.03385)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request with your changes. Ensure your code follows the repository's style guidelines and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for more details.

## Contact

For any questions or suggestions, please open an issue or contact me at saketjha34@gmail.com.

---

Happy coding!

