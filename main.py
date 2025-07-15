import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import onnx
import onnxruntime as ort
import os
import torch.serialization
torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])

# Paths
PT_MODEL_PATH = "yolo11n.pt"
ONNX_MODEL_PATH = "yoldin.onnx"
IMAGE_PATH = "image.png"
OUTPUT_PT = "output_pytorch.png"
OUTPUT_ONNX = "output_onnx.png"

# 1. Load and preprocess image
def load_image(image_path, img_size=640):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ])
    img_tensor = transform(image)
    return image, img_tensor.unsqueeze(0)  # Add batch dimension

# 2. Inference with PyTorch model
def run_pytorch_inference(model, img_tensor, orig_image):
    model.eval()
    with torch.no_grad():
        preds = model(img_tensor)[0]  # YOLO models usually return a tuple/list
    # Post-processing: assuming YOLOv5/YOLOv8 style output [N, 6]: x1, y1, x2, y2, conf, class
    preds = preds.cpu().numpy()
    boxes = []
    for pred in preds:
        x1, y1, x2, y2, conf, cls = pred
        if conf > 0.3:  # Confidence threshold
            boxes.append((x1, y1, x2, y2, conf, int(cls)))
    draw_boxes(orig_image, boxes, OUTPUT_PT, "PyTorch")
    print("PyTorch Inference Results:")
    for box in boxes:
        print(f"Box: {box[:4]}, Confidence: {box[4]:.2f}, Class: {box[5]}")
    return boxes

# 3. Convert to ONNX
def convert_to_onnx(model, img_tensor, onnx_path):
    torch.onnx.export(
        model,
        img_tensor,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes={'images': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model exported to {onnx_path}")

# 4. Validate ONNX
def validate_onnx(onnx_path):
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("ONNX model is valid.")

# 5. Inference with ONNX
def run_onnx_inference(onnx_path, img_tensor, orig_image):
    ort_session = ort.InferenceSession(onnx_path)
    img_numpy = img_tensor.numpy()
    outputs = ort_session.run(None, {'images': img_numpy})
    preds = outputs[0][0]  # [batch, N, 6] or [N, 6]
    boxes = []
    for pred in preds:
        x1, y1, x2, y2, conf, cls = pred
        if conf > 0.3:
            boxes.append((x1, y1, x2, y2, conf, int(cls)))
    draw_boxes(orig_image, boxes, OUTPUT_ONNX, "ONNX")
    print("ONNX Inference Results:")
    for box in boxes:
        print(f"Box: {box[:4]}, Confidence: {box[4]:.2f}, Class: {box[5]}")
    return boxes

# 6. Draw boxes
def draw_boxes(image, boxes, output_path, title):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), f"{cls}:{conf:.2f}", fill="yellow", font=font)
    img.save(output_path)
    print(f"{title} result saved to {output_path}")

def main():
    # Load image
    orig_image, img_tensor = load_image(IMAGE_PATH)
    print("Image loaded and preprocessed.")

    # Load PyTorch model
    model = torch.load(PT_MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
    print("PyTorch model loaded.")

    # PyTorch inference
    run_pytorch_inference(model, img_tensor, orig_image)

    # Convert to ONNX
    convert_to_onnx(model, img_tensor, ONNX_MODEL_PATH)
    validate_onnx(ONNX_MODEL_PATH)

    # ONNX inference
    run_onnx_inference(ONNX_MODEL_PATH, img_tensor, orig_image)

if __name__ == "__main__":
    main() 