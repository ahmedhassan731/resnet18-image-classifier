
import torch
import gradio as gr
from transformers import AutoImageProcessor, ResNetForImageClassification

MODEL_NAME = "microsoft/resnet-18"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = ResNetForImageClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()


def predict(image):
    if image is None:
        return {}

    inputs = processor(
        images=image.convert("RGB"),
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probabilities = torch.softmax(logits, dim=-1)[0]
    top5 = torch.topk(probabilities, k=5)

    return {
        model.config.id2label[index.item()]: float(probability)
        for probability, index in zip(top5.values, top5.indices)
    }


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Label(num_top_classes=5, label="Predictions"),
    title="ResNet-18 Image Classifier",
    description=(
        "Upload an image to get the top 5 ImageNet class predictions."
    )
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
