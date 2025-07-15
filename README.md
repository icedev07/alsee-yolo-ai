# AL.SEE Assessment Solution

## Steps

1. **Create and activate a virtual environment:**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Place `yoldin.pt` and `image.png` in the project directory.**

4. **Run the workflow:**
   ```
   python main.py
   ```

5. **Outputs:**
   - `output_pytorch.png`: Detections from PyTorch model
   - `output_onnx.png`: Detections from ONNX model

## Notes

- The code assumes the YOLO model outputs `[N, 6]` arrays: `[x1, y1, x2, y2, conf, class]`.
- Adjust confidence threshold or output parsing if your model differs.
- All steps are explained in the code and can be narrated during your video recording.

---

**Good luck with your assessment! If you need any code adjustments (e.g., for a different YOLO output format), let me know!** 