
import os

import gradio as gr
import torch
from transformers import AutoImageProcessor, ResNetForImageClassification

# Model configuration
MODEL_NAME = "microsoft/resnet-18"

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load pretrained model and image processor
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = ResNetForImageClassification.from_pretrained(MODEL_NAME)

model.to(device)
model.eval()


def predict(image):
    """Predict the top 5 ImageNet classes for an uploaded image."""
    if image is None:
        return {}

    image = image.convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probabilities = torch.softmax(logits, dim=-1)[0]
    top5 = torch.topk(probabilities, k=5)

    predictions = {
        model.config.id2label[index.item()]: float(probability.item())
        for probability, index in zip(top5.values, top5.indices)
    }

    return predictions


# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Label(
        num_top_classes=5,
        label="Top 5 Predictions"
    ),
    title="ResNet-18 Image Classifier",
    description=(
        "Upload an image to see the top 5 predicted ImageNet classes "
        "with their confidence scores."
    ),
    examples=None
)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
