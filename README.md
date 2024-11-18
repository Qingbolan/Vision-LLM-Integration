# VisionaryLLM: An Extensible Framework for Enhancing Large Language Models with Doman-Specific Vision Tasks Using Deep Learning

[![python](https://img.shields.io/badge/-Python_3.7_%7C_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_1.9+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)

This project focuses on the **Deep Learning Visual Analysis with LLM Integration**, aiming to develop a general framework that integrates traditional deep learning models with Large Language Models (LLMs) to enhance image understanding and visualization. By transforming complex visual tasks into efficient classification problems, we strive to create a system that not only automates detection processes but also provides interpretable and interactive insights through natural language.

please see project website to learn more: [https://cs5242-demo.silan.tech](https://cs5242-demo.silan.tech)

## Table of Contents

- [Project Motivation](#project-motivation)
- [Code Structure](#code-structure)
- [Project Setup](#project-setup)
- [Dataset](#dataset)
- [Intelligent Segmentation Strategy](#intelligent-segmentation-strategy)
- [Models Implemented](#models-implemented)
  - [Supervised Learning](#supervised-learning)
    - [ResNet50](#resnet50)
    - [AlexNet](#alexnet)
    - [Vision Transformer (ViT)](#vision-transformer-vit)
  - [Unsupervised Learning](#unsupervised-learning)
    - [Deep Convolutional Autoencoder (DCAE)](#deep-convolutional-autoencoder-dcae)
    - [Deep Convolutional Variational Autoencoder (DCVAE)](#deep-convolutional-variational-autoencoder-dcvae)
    - [Anomaly Detection using Vision Transformers](#anomaly-detection-using-vision-transformers)
- [Experimentation and Results](#experimentation-and-results)
- [LLM Integration](#llm-integration)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Configuration](#configuration)
- [Limitations](#limitations)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Project Motivation

The rapid advancement of Large Language Models (LLMs) has revolutionized many aspects of artificial intelligence, yet their application in professional domains, particularly those requiring specialized visual analysis, remains significantly constrained. This limitation creates a critical gap between the theoretical capabilities of LLMs and their practical utility in professional settings.

## Current Challenges and Research Gap

The integration of LLMs with domain-specific visual tasks presents several compelling challenges that motivate our research:

1. **Lack of Professional-Grade Analysis**

   - Current LLMs, while capable of basic image interpretation, fall short in providing the quantifiable confidence levels required in professional settings
   - Professional domains such as structural engineering and medical imaging demand precise measurements and reliable metrics that meet industry standards
   - The absence of these capabilities limits the practical adoption of LLMs in critical professional applications
2. **Black-Box Decision Making**

   - Existing LLM implementations operate as black boxes, providing conclusions without transparent reasoning
   - Professionals cannot verify the analysis process or understand how specific visual elements influence decisions
   - This opacity poses significant risks in domains where decision validation is crucial for safety and compliance
3. **Integration Complexity**

   - The current landscape lacks a standardized framework for integrating LLMs with domain-specific visual tasks
   - Organizations must develop custom solutions for each domain, leading to:
     - Inconsistent implementations across different fields
     - High development and maintenance costs
     - Limited scalability and reusability

## Key Objectives

These challenges present a significant opportunity for innovation in the field of AI and computer vision. By developing VisionaryLLM, we aim to bridge the gap between general-purpose LLMs and specialized visual analysis requirements. Our framework addresses these limitations through:

- **Standardized Integration**: A flexible architecture that simplifies the combination of vision models with LLMs
- **Transparent Analysis**: Implementation of gradient-weighted class activation mapping (Grad-CAM) for result visualization
- **Cross-Domain Applicability**: Demonstrated effectiveness in diverse fields, from structural engineering to medical imaging

## Code Structure

- **Backend**: `app`, run `main.py` to start the backend on https://localhost:5100
- **Data Set and Backend photo stores:**
  - `data/_input`: all the input data get from the frontend
  - `data/_output`: all the deep learning models classification process generation by GradCAM
  - `data/raw`: the crack dataset
  - `data/{}other_data_set_name}`
- **AI Agorithm and utils support:**
  - **Data Loading and Preprocessing**: `src/data/dataset.py`, `src/data/preprocess.py`
  - **Model Definitions**:
    - `src/models/resnet_model.py`
    - `src/models/alexnet_model.py`
    - `src/models/vit_model.py`
    - `src/models/autoencoder.py` (DCAE)
    - `src/models/variational_autoencoder.py` (DCVAE)
    - `src/models/vit_anomaly.py` (ViT Anomaly Detection)
  - **Training Logic**:
    - `src/training/trainer.py`
    - `src/training/autoencoder_trainer.py`
    - `src/training/variational_autoencoder_trainer.py`
    - `src/training/vit_anomaly_trainer.py`
  - **Evaluation Code**:
    - `src/evaluation/evaluator.py`
    - `src/evaluation/autoencoder_evaluator.py`
    - `src/evaluation/variational_autoencoder_evaluator.py`
    - `src/evaluation/vit_anomaly_evaluator.py`
  - **LLM Integration**: `src/llm/agent.py`
- **Configuration Files**: `config/config.yaml`, various files under `configs/`

## Project Setup

Follow these steps to set up the project environment:

```bash
# Clone the repository
git clone https://github.com/Qingbolan/deep-learning-visual-analysis.git
cd deep-learning-visual-analysis

# Create a virtual environment
conda create -n visual_analysis python=3.9
conda activate visual_analysis

# Install PyTorch according to your system specifications
# Visit https://pytorch.org/get-started/locally/ for installation commands

# Install project dependencies
pip install -r requirements.txt

# Create outputs directory
mkdir outputs
```

Ensure that you have the prerequisites for [PyTorch](https://pytorch.org/) installed, compatible with your CUDA or CPU setup.

## Dataset

For demonstration, we utilize the **Concrete Crack Images for Classification** dataset to validate our Deep Learning Visual Analysis framework. Please download the dataset from [Concrete Crack Images for Classification](https://data.mendeley.com/datasets/5y9wdsg2zt/2) and organize the images in the `data/raw/` directory as follows:

```
data/
├── raw/
│   ├── Negative/
│   └── Positive/
```

- **Negative**: Images without cracks.
- **Positive**: Images with cracks.

*Note*: The framework is designed to accommodate various datasets, enabling its application to diverse visual tasks beyond crack detection.

```bash
# also you can run the following script to get other dataset
python BreastMNIST_download.py
python ChestMNIST_download.py
```

## Intelligent Segmentation Strategy

### Innovative Image Segmentation: Transforming Complex Visual Tasks into Efficient Classification Problems

Our **Intelligent Segmentation** strategy revolutionizes the approach to complex visual tasks by decomposing them into manageable classification units. This method enhances processing efficiency, accuracy, and scalability, making it suitable for a wide range of applications.

#### Core Principles:

1. **Image Partitioning**: Dividing high-resolution or complex images into smaller, uniform patches to simplify the classification process.
2. **Local Feature Extraction**: Applying deep learning models to each patch to identify the presence or absence of specific features or anomalies.
3. **Result Integration**: Aggregating patch-level classifications to form a comprehensive understanding of the entire image, enabling precise localization and analysis.

#### Implementation Steps:

1. **Image Segmentation**:

   - **Grid-Based Partitioning**: Splitting the image into fixed-size patches (e.g., 16x16 pixels) to ensure uniformity.
   - **Adaptive Segmentation**: Utilizing algorithms that adjust patch sizes based on image content for better feature representation.
2. **Feature Extraction and Classification**:

   - **Supervised Learning Models**: Employing architectures like ResNet50, AlexNet, and Vision Transformer (ViT) to classify each patch.
   - **Unsupervised Learning Models**: Utilizing Autoencoder, Variational Autoencoder (VAE), and ViT-based anomaly detection to identify deviations from normal patterns.
3. **Result Aggregation and Visualization**:

   - **Matrix Mapping**: Creating a matrix that maps the classification results of each patch, highlighting areas of interest (e.g., cracks) with distinct markers.
   - **LLM Integration**: Leveraging LLMs to interpret and describe the aggregated results, providing natural language summaries and insights.

#### Advantages:

- **Enhanced Efficiency**: Parallel processing of patches reduces computational load and accelerates overall analysis.
- **Improved Accuracy**: Localized classification ensures detailed detection of subtle anomalies.
- **Scalability**: The framework can easily scale to accommodate varying image sizes and multiple visual tasks.
- **Flexibility**: Applicable to diverse domains such as structural health monitoring, industrial inspection, and medical imaging.

#### Limitations and Mitigations:

- **Edge Effects**: Potential loss of contextual information at patch boundaries, mitigated by overlapping segmentation and contextual fusion techniques.
- **Global Context Loss**: Addressed by integrating global features through transformer-based models and post-processing methods.
- **Resource Intensity**: Optimized through efficient model architectures and distributed computing strategies.
- **Data Annotation**: Increased complexity in labeling patch-level data, especially for unsupervised tasks, addressed by leveraging automated or semi-automated annotation tools.

Our Intelligent Segmentation strategy streamlines complex visual tasks, setting the foundation for integrating advanced analytical capabilities through LLMs, and paving the way for intelligent and interactive systems.

## Models Implemented

### Supervised Learning

We implemented and evaluated several traditional deep learning models to classify segmented image patches effectively.

#### ResNet50

ResNet50 introduces residual learning to facilitate the training of deep neural networks, effectively addressing the vanishing gradient problem.

![ResNet Architecture](./assets/image-20241015193834961.png)

**Key Features:**

- **Deep Residual Blocks**: Allowing the training of deeper networks without degradation in performance.
- **Batch Normalization**: Enhancing training stability and speed.
- **Global Average Pooling**: Reducing model parameters and preventing overfitting.

#### AlexNet

AlexNet is a pioneering convolutional neural network known for its success in the ImageNet competition, demonstrating the potential of deep learning in image classification.

![AlexNet Architecture](./assets/WhatsApp%20%E5%9B%BE%E5%83%8F2024-10-15%E4%BA%8E19.29.43_6d3c2c53.jpg)

**Key Features:**

- **Deep Convolutional Layers**: Extracting hierarchical features from images.
- **ReLU Activation**: Introducing non-linearity and accelerating training.
- **Dropout Layers**: Mitigating overfitting by randomly dropping neurons during training.

#### Vision Transformer (ViT)

ViT applies the Transformer architecture directly to image recognition tasks, leveraging self-attention mechanisms to capture global dependencies within images.

![Vision Transformer Architecture](./assets/image-20241015194113619.png)

**Key Features:**

- **Patch Embedding**: Dividing images into patches and embedding them into a sequence.
- **Self-Attention Mechanism**: Enabling the model to focus on relevant parts of the image.
- **Transformer Blocks**: Facilitating the capture of complex feature relationships across the entire image.

### Unsupervised Learning

Our unsupervised models focus on learning intrinsic data representations and detecting anomalies without explicit labels.

#### Deep Convolutional Autoencoder (DCAE)

The DCAE employs a symmetric encoder-decoder architecture tailored for concrete crack image processing, focusing on dimensionality reduction and feature learning.

![Deep Convolutional Autoencoder (DCAE)](./assets/WhatsApp%20%E5%9B%BE%E5%83%8F2024-10-22%E4%BA%8E12.05.39_17b2e6e8.jpg)

**Architecture Details:**

- **Encoder:**

  - Conv2D (3→2048, kernel=3x3)
  - Conv2D (2048→1024, kernel=3x3)
  - Conv2D (1024→512, kernel=3x3)
  - Fully Connected layer (512*2*2→128)
- **Latent Space:**

  - 128-dimensional representation
- **Decoder:**

  - Fully Connected layer (128→512*2*2)
  - ConvTranspose2D (512→512, kernel=3x3)
  - ConvTranspose2D (512→1024, kernel=3x3)
  - ConvTranspose2D (1024→2048, kernel=3x3)
  - ConvTranspose2D (2048→3, kernel=3x3)

**Key Features:**

- Maintains spatial information through convolutional operations.
- Enables efficient dimensionality reduction.
- Facilitates anomaly detection via reconstruction error.

#### Deep Convolutional Variational Autoencoder (DCVAE)

DCVAE extends the traditional autoencoder by introducing probabilistic encoding, enhancing the model's ability to generalize and detect anomalies.

![Deep Convolutional Variational Autoencoder (DCVAE)](./assets/WhatsApp%20%E5%9B%BE%E5%83%8F2024-10-22%E4%BA%8E12.05.39_17b2e6e8-1729581873230-50.jpg)

**Architecture Details:**

- **Encoder:**

  - Conv2D (3→2048, kernel=3x3)
  - Conv2D (2048→1024, kernel=3x3)
  - Conv2D (1024→512, kernel=3x3)
  - Two parallel FC layers for μ and σ
- **Latent Space:**

  - Probabilistic sampling using the reparameterization trick: Z ~ N(μ, σ²I)
  - FC layer (512*2*2→128) for both mean and variance
- **Decoder:**

  - ConvTranspose2D (512→512, kernel=3x3)
  - ConvTranspose2D (512→1024, kernel=3x3)
  - ConvTranspose2D (1024→2048, kernel=3x3)
  - Final reconstruction layer

**Key Features:**

- Stochastic latent representation.
- KL divergence regularization.
- Enhanced generalization through variational inference.

#### Anomaly Detection using Vision Transformers

Our ViT-based anomaly detection system leverages the self-attention mechanism of transformers to identify irregular patterns in images without explicit labels.

![Anomaly Detection using Vision Transformers](./assets/WhatsApp%20%E5%9B%BE%E5%83%8F2024-10-21%E4%BA%8E12.21.59_d38439ad.jpg)

**Architecture Components:**

1. **Image Tokenization and Embedding:**

   - Divides input images into fixed-size patches (e.g., 16x16 pixels).
   - Projects each patch into a higher-dimensional space using linear embedding.
   - Adds a special classification token and positional embeddings to retain spatial information.
2. **Transformer Encoder:**

   - Comprises multiple transformer blocks with layer normalization, multi-head self-attention, and MLP layers.
   - Utilizes residual connections for stable gradient flow.
3. **Classification Head:**

   - Processes the output corresponding to the classification token.
   - Produces a binary output indicating normal or anomaly.

**Key Features:**

- **Patch-Based Processing**: Maintains spatial relationships and enables parallel processing.
- **Self-Attention Mechanism**: Captures global dependencies and focuses on relevant image regions.
- **MLP Block Design**: Incorporates non-linearity and ensures robust feature learning.
- **Anomaly Detection Strategy**: Identifies deviations from learned normal patterns without explicit labels.

**Usage Example:**

```python
# Model initialization
model = ViTAnomalyDetector(
    img_size=224,          # Input image size
    patch_size=16,         # Size of each patch
    embed_dim=768,         # Embedding dimension
    num_heads=12,          # Number of attention heads
    num_layers=12,         # Number of transformer blocks
    mlp_ratio=4,           # MLP hidden dimension ratio
    num_classes=2          # Binary classification
)

# Anomaly detection
anomaly_scores = model.get_anomaly_score(images)
predictions = anomaly_scores > threshold
```

## Experimentation and Results

We conducted extensive experiments to evaluate the performance of our implemented models within the Deep Learning Visual Analysis framework.

### Performance Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

### Experimental Findings

1. **ResNet50** achieved high accuracy due to its deep architecture and residual connections, effectively learning complex features.
2. **AlexNet** demonstrated reasonable performance but was outperformed by deeper networks like ResNet50 and ViT.
3. **Vision Transformer (ViT)** showcased the potential of transformer architectures in image classification, achieving competitive results with superior global feature capture.
4. **Deep Convolutional Autoencoder (DCAE)** and **Deep Convolutional Variational Autoencoder (DCVAE)** provided valuable insights through feature extraction and anomaly detection, aiding in understanding underlying data representations.
5. **Anomaly Detection using ViT** proved effective in identifying irregular patterns without explicit labels, highlighting its utility in scenarios with limited annotated data.
6. **Intelligent Segmentation** strategy enhanced overall detection accuracy by enabling detailed local analysis and efficient processing.

### Observations

- **Data Augmentation**: Implementing techniques such as rotation, scaling, and flipping improved model generalization.
- **Hyperparameter Tuning**: Optimizing learning rates, batch sizes, and model-specific parameters was crucial for achieving optimal performance.
- **Computational Resources**: Transformer-based models required significant computational power, necessitating efficient training strategies and resource management.
- **Unsupervised Learning**: Incorporating unsupervised methods provided additional layers of analysis, enabling the detection of anomalies beyond supervised capabilities.

## LLM Integration

To elevate the analytical capabilities of our system, we integrated Large Language Models (LLMs) to interpret and interact with the visual data classifications intelligently.

### Features:

- **Automated Reporting**: LLMs generate comprehensive reports based on classification results, summarizing key findings and insights.
- **Interactive Queries**: Users can interact with the system using natural language queries, receiving detailed explanations and analyses.
- **Contextual Understanding**: LLMs enhance the system's ability to understand contextual relationships within the data, providing more meaningful interpretations.

### Implementation:

- **Agent System**: Developed an agent that serves as an intermediary between the visual classification module and the LLM, facilitating seamless communication and data exchange.
- **Task Automation**: Enabled the LLM to perform tasks such as counting detected anomalies, marking their locations, and providing trend analyses.
- **User Interface**: Designed a natural language interface allowing users to interact with the system effortlessly, enhancing usability and accessibility.

### Benefits:

- **Enhanced Interpretability**: Transforms raw classification data into understandable and actionable insights.
- **Improved User Experience**: Facilitates intuitive interactions, making the system accessible to users without technical expertise.
- **Scalable Intelligence**: Allows the system to adapt and respond to a wide range of queries and analytical needs dynamically.

## Usage

### Training

To train a model, modify the `config/config.yaml` file to set the desired model and parameters.

**Example: To train with ResNet50:**

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

**Example:**

```bash
python main.py model.name=efficientnet model.pretrained=False
```

## Limitations

While our framework provides a structured approach to integrating deep learning models with LLMs for visual analysis, it is not without limitations:

1. **Edge Effects**: Splitting images into patches can result in loss of contextual information at the boundaries. Although overlapping segmentation can mitigate this, it adds computational overhead.
2. **Global Context Loss**: The patch-based approach may fail to capture the overall structure and global dependencies within an image, potentially affecting detection accuracy in complex scenarios.
3. **Resource Intensity**: Processing large numbers of patches, especially with transformer-based models, requires substantial computational resources, which may not be feasible for all applications.
4. **Data Annotation Complexity**: Annotating data at the patch level increases the labeling effort, particularly for unsupervised tasks where anomaly labels are not readily available.
5. **Integration Complexity**: Combining deep learning models with LLMs introduces additional layers of complexity in system design and requires careful coordination to ensure seamless interaction and data flow.

Future work will focus on addressing these limitations by exploring more advanced segmentation techniques, optimizing computational efficiency, and enhancing the integration mechanisms between deep learning models and LLMs.

## Acknowledgements

We extend our gratitude to the providers of the [Concrete Crack Images for Classification](https://data.mendeley.com/datasets/5y9wdsg2zt/2) dataset.

**Citations:**

- Özgenel, Ç.F., & Gönenç Sorguç, A. (2018). *Performance Comparison of Pretrained Convolutional Neural Networks on Crack Detection in Buildings*. ISARC 2018, Berlin.
- Zhang, L., Yang, F., Zhang, Y. D., & Zhu, Y. J. (2016). *Road Crack Detection Using Deep Convolutional Neural Network*. IEEE International Conference on Image Processing (ICIP). [doi:10.1109/ICIP.2016.7533052](http://doi.org/10.1109/ICIP.2016.7533052)

## License&&author

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

*Note*: This project represents an attempt to develop a general framework integrating traditional deep learning techniques with Large Language Models (LLMs) for enhanced visual analysis. The implementations and findings reflect thoughtful application and critical thinking in applying established models to practical, real-world problems. We have meticulously compared different architectures, implemented both supervised and unsupervised learning methods, and provided comprehensive documentation to ensure understanding and reproducibility.

Feel free to explore the codebase, run experiments, and contribute to the project!

**Project GitHub Link**: [https://github.com/Qingbolan/Vision-LLM-Integration](https://github.com/Qingbolan/Vision-LLM-Integration)

<table>
  <tr>
    <td align="center"><a href="https://github.com/Qingbolan"><img src="https://github.com/Qingbolan.png" width="100px;" alt=""/><br /><sub><b>HU SILAN</b></sub></a><br /><a href="https://github.com/Qingbolan/deep-learning-visual-analysis" title="Code">💻</a></td>
    <td align="center"><a href="#"><img src="https://via.placeholder.com/100" width="100px;" alt=""/><br /><sub><b>Tan Kah Xuan</b></sub></a><br /><a href="https://github.com/Qingbolan/deep-learning-visual-analysis" title="Code">💻</a></td>
  </tr>
</table>
