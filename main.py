import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import onnx
import onnxruntime as ort
import os
from ultralytics import YOLO

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

# 2. Inference with Ultralytics YOLO model
def run_pytorch_inference(model, img_tensor, orig_image):
    results = model(img_tensor)
    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            if conf > 0.3:
                boxes.append((x1, y1, x2, y2, conf, cls))
    draw_boxes(orig_image, boxes, OUTPUT_PT, "PyTorch")
    print("PyTorch Inference Results:")
    for box in boxes:
        print(f"Box: {box[:4]}, Confidence: {box[4]:.2f}, Class: {box[5]}")
    return boxes

# 3. Convert to ONNX
def convert_to_onnx(model, img_tensor, onnx_path):
    torch.onnx.export(
        model.model,
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
    print("ONNX raw output shape:", outputs[0].shape)
    boxes = postprocess_yolov8_onnx(outputs[0])
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

def postprocess_yolov8_onnx(output, conf_thres=0.3):
    # output: [1, 84, 8400] -> [8400, 84]
    output = output[0]  # remove batch dim, now [84, 8400]
    output = output.transpose(1, 0)  # [8400, 84]
    boxes = []
    for row in output:
        # YOLOv8 ONNX: [cx, cy, w, h, obj_conf, class_conf_0, ..., class_conf_n]
        cx, cy, w, h = row[:4]
        obj_conf = row[4]
        class_confs = row[5:]
        class_id = np.argmax(class_confs)
        class_conf = class_confs[class_id]
        conf = obj_conf * class_conf
        if conf > conf_thres:
            # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            # Ensure valid coordinates
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2, conf, class_id))
    return boxes

def main():
    # Load image
    orig_image, img_tensor = load_image(IMAGE_PATH)
    print("Image loaded and preprocessed.")

    # Load Ultralytics YOLO model
    model = YOLO(PT_MODEL_PATH)
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