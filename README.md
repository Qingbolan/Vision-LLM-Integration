# Concrete Crack Image Classification

[![python](https://img.shields.io/badge/-Python_3.7_%7C_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_1.9+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)

This project focuses on the **automatic classification of concrete crack images**, an essential task in structural health monitoring of infrastructure. By accurately identifying cracks in concrete structures, we can facilitate timely maintenance and prevent potential failures.

We implement and compare several advanced deep learning models, including:

- **ResNet50**
- **AlexNet**
- **VGG16**
- **Vision Transformer (ViT)**
- **EfficientNet**
- **Deep Convolutional Autoencoder (DCAE)**
- **Deep Convolutional Variational Autoencoder (DCVAE)**
- **Anomaly Detection using Vision Transformers**

All models are trained and evaluated on the "Concrete Crack Images for Classification" dataset to solve this binary classification problem.

## Table of Contents

- [Project Motivation](#project-motivation)
- [Code Structure](#code-structure)
- [Project Setup](#project-setup)
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
  - [ResNet50](#resnet50)
  - [AlexNet](#alexnet)
  - [VGG16](#vgg16)
  - [Vision Transformer (ViT)](#vision-transformer-vit)
  - [EfficientNet](#efficientnet)
  - [Deep Convolutional Autoencoder (DCAE)](#deep-convolutional-autoencoder-dcae)
  - [Deep Convolutional Variational Autoencoder (DCVAE)](#deep-convolutional-variational-autoencoder-dcvae)
  - [Anomaly Detection using Vision Transformers](#anomaly-detection-using-vision-transformers)
- [Experimentation and Results](#experimentation-and-results)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Configuration](#configuration)
- [Acknowledgements](#acknowledgements)
- [Contributors](#contributors)
- [License](#license)

## Project Motivation

Crack detection in concrete structures is vital for ensuring the safety and longevity of infrastructure. Traditional manual inspection methods are time-consuming and prone to human error. Automating this process using deep learning can significantly enhance inspection efficiency and accuracy.

In this project, we aim to:

- **Explore a variety of deep learning models** for image classification, ranging from classical CNN architectures to modern transformer-based models.
- **Evaluate and compare the performance** of these models on the concrete crack detection task.
- **Implement both supervised and unsupervised learning approaches**, including anomaly detection methods.
- **Reflect on the strengths and weaknesses** of each model to provide insights into their practical applicability.

This comprehensive exploration not only deepens our understanding of deep learning techniques but also contributes to practical solutions in the field of structural engineering.

## Code Structure

- **Data Loading and Preprocessing**: `src/data/dataset.py`, `src/data/preprocess.py`
- **Model Definitions**:
  - `src/models/resnet_model.py`
  - `src/models/alexnet_model.py`
  - `src/models/vgg_model.py`
  - `src/models/vit_model.py`
  - `src/models/efficientnet_model.py`
  - `src/models/autoencoder.py` (DCAE)
  - `src/models/variational_autoencoder.py` (DCVAE)
- **Training Logic**:
  - `src/training/trainer.py`
  - `src/training/autoencoder_trainer.py`
  - `src/training/variational_autoencoder_trainer.py`
- **Evaluation Code**:
  - `src/evaluation/evaluator.py`
  - `src/evaluation/autoencoder_evaluator.py`
  - `src/evaluation/variational_autoencoder_evaluator.py`
- **Main Script**: `main.py`
- **Configuration Files**: `config/config.yaml`, various files under `configs/`

## Project Setup

Follow these steps to set up the project environment:

```bash
# Clone the repository
git clone https://github.com/Qingbolan/cs5242-for-Concrete-Crack.git
cd cs5242-for-Concrete-Crack

# Create a virtual environment
conda create -n concrete_crack python=3.9
conda activate concrete_crack

# Install PyTorch according to your system specifications
# Visit https://pytorch.org/get-started/locally/ for installation commands

# Install project dependencies
pip install -r requirements.txt

# Create outputs directory
mkdir outputs
```

Make sure you have met the prerequisites for [PyTorch](https://pytorch.org/) and installed the corresponding version compatible with your CUDA or CPU setup.

## Dataset

Please download the dataset from [Concrete Crack Images for Classification](https://data.mendeley.com/datasets/5y9wdsg2zt/2) and place the images in the `data/raw/` directory following this structure:

```
data/
├── raw/
│   ├── Negative/
│   └── Positive/
```

- **Negative**: Images without cracks.
- **Positive**: Images with cracks.

## Models Implemented

### ResNet50

ResNet50 introduces residual learning to ease the training of deep neural networks. It helps in avoiding vanishing gradient problems.

![ResNet Architecture][]
![image-20241015193834961](./assets/image-20241015193834961.png)

### AlexNet

AlexNet is one of the pioneering models in deep learning, known for its success in the ImageNet competition.

![AlexNet Architecture][]
![WhatsApp 图像2024-10-15于19.29.43_6d3c2c53](./assets/WhatsApp%20%E5%9B%BE%E5%83%8F2024-10-15%E4%BA%8E19.29.43_6d3c2c53.jpg)


### VGG16

VGG16 uses very small (3x3) convolution filters, which showed that the depth of the network is a critical component for good performance.

![VGG16 Architecture][]

![WhatsApp 图像2024-10-15于19.25.41_303f991c](./assets/WhatsApp%20%E5%9B%BE%E5%83%8F2024-10-15%E4%BA%8E19.25.41_303f991c.jpg)

### Vision Transformer (ViT)

ViT applies the Transformer architecture directly to image recognition, with great success when pre-trained on large datasets.

![Vision Transformer Architecture][]

![image-20241015194113619](./assets/image-20241015194113619.png)

### EfficientNet

EfficientNet scales up networks using a compound coefficient, balancing network depth, width, and input resolution.

![EfficientNet Architecture][]

### Deep Convolutional Autoencoder (DCAE)

DCAE is an unsupervised learning method that learns efficient data codings in an unsupervised manner. It is effective for dimensionality reduction and feature learning.

![DCAE Structure][]

### Deep Convolutional Variational Autoencoder (DCVAE)

DCVAE extends the autoencoder architecture with a probabilistic approach to model the data distribution, enabling the generation of new data samples.

![DCVAE Structure][]

### Anomaly Detection using Vision Transformers

Using ViT for anomaly detection leverages its ability to capture global relationships in data, making it suitable for detecting subtle irregularities in images.

![ViT Anomaly Detection][]

## Experimentation and Results

We conducted extensive experiments with the aforementioned models to evaluate their performance in the task of concrete crack detection.

**Performance Metrics**:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

**Experimental Findings**:

1. **ResNet50** achieved high accuracy due to its deep architecture and residual connections, which help in learning complex features.

2. **AlexNet**, despite being one of the earlier models, performed reasonably well but was outperformed by deeper networks.

3. **VGG16** showed strong performance thanks to its depth, but at the cost of increased computational resources.

4. **Vision Transformer (ViT)** demonstrated that transformer architectures can be effectively applied to image classification tasks, achieving competitive performance.

5. **EfficientNet** provided an excellent balance between accuracy and computational efficiency due to its scalable architecture.

6. **DCAE and DCVAE** were used for feature extraction and anomaly detection. While not directly classifying images, they helped in understanding the underlying data representation.

7. **Anomaly Detection using ViT** highlighted the model's capacity to detect anomalies without explicit labels, which is valuable in scenarios with limited annotated data.

**Observations**:

- **Data Augmentation**: Implementing data augmentation techniques improved model generalization.

- **Hyperparameter Tuning**: Careful tuning of learning rates, batch sizes, and optimizers was crucial for model convergence.

- **Computational Resources**: Transformer-based models required more computational power, highlighting the need for efficient training strategies.

- **Unsupervised Learning**: Including unsupervised methods like autoencoders provided additional insights into the data and potential for anomaly detection.

## Usage

### Training

To train a model, modify the `config/config.yaml` file to set the desired model and parameters.

Example: To train with ResNet50:

```yaml
model:
  name: resnet50
  pretrained: true
  num_classes: 2
  # Other model-specific parameters
```

Then run:

```bash
python main.py
```

### Evaluation

After training, the model will automatically be evaluated on the validation set, and metrics will be displayed. To perform additional evaluation:

```bash
python main.py --evaluate --ckpt_path path/to/checkpoint.pth
```

### Configuration

All configurations are managed through YAML files for clarity and flexibility. Parameters can also be overridden via command-line arguments.

Example:

```bash
python main.py model.name=efficientnet model.pretrained=False
```

## Acknowledgements

We are grateful to the providers of the [Concrete Crack Images for Classification](https://data.mendeley.com/datasets/5y9wdsg2zt/2) dataset.

**Citations**:

- Özgenel, Ç.F., & Gönenç Sorguç, A. (2018). *Performance Comparison of Pretrained Convolutional Neural Networks on Crack Detection in Buildings*. ISARC 2018, Berlin.
- Zhang, L., Yang, F., Zhang, Y. D., & Zhu, Y. J. (2016). *Road Crack Detection Using Deep Convolutional Neural Network*. IEEE International Conference on Image Processing (ICIP). http://doi.org/10.1109/ICIP.2016.7533052

## Contributors

- **HU SILAN** ([Qingbolan](https://github.com/Qingbolan)) - Project Lead, Model Implementation, Experimentation
- **Tan Kah Xuan** - Data Preparation, Model Training, Evaluation

We would also like to acknowledge the open-source community for providing invaluable resources and inspiration for this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
*Note*: This project is part of an academic exploration into deep learning techniques for image classification. The implementations and findings reflect extensive work and critical thinking in applying state-of-the-art models to a practical problem. We have extensively compared different architectures, implemented both supervised and unsupervised learning methods, and provided thorough documentation to aid in understanding and reproducibility.

Feel free to explore the codebase, run experiments, and contribute to the project!

**Project GitHub Link**: [https://github.com/Qingbolan/cs5242-for-Concrete-Crack](https://github.com/Qingbolan/cs5242-for-Concrete-Crack)

<table>
  <tr>
    <td align="center"><a href="https://github.com/Qingbolan"><img src="https://github.com/Qingbolan.png" width="100px;" alt=""/><br /><sub><b>HU SILAN</b></sub></a><br /><a href="https://github.com/Qingbolan/cs5242-for-Concrete-Crack" title="Code">💻</a></td>
    <td align="center"><a href="#"><img src="https://via.placeholder.com/100" width="100px;" alt=""/><br /><sub><b>Tan Kah Xuan</b></sub></a><br /><a href="https://github.com/Qingbolan/cs5242-for-Concrete-Crack" title="Code">💻</a></td>
  </tr>
</table>
