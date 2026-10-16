
# ResNet-18 Image Classifier

An image classification web application built with PyTorch,
Hugging Face Transformers, and Gradio.

## Features

- Upload images through a web interface
- Predict the top 5 ImageNet classes
- Display prediction confidence scores
- Automatically use CUDA when available
- Use a pretrained ResNet-18 model

## Model

microsoft/resnet-18

The model is pretrained on ImageNet with 1,000 classes.

## Installation

```bash
python -m venv env
```

Activate the environment on Windows:

```bash
env\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the application

```bash
python app.py
```

Open the local URL shown in your terminal.

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Gradio
