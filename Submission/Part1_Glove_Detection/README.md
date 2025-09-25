# YOLOv5 Glove Detection

A custom YOLOv5 implementation for detecting gloves and no-glove instances in images using a trained model.

## Overview

This project uses YOLOv5 for binary classification to detect:
- **Glove**: When protective gloves are present
- **No Glove**: When hands are visible without gloves

## Project Structure

```
yolo-tank/
├── yolov5/                          # YOLOv5 repository
│   ├── detect.py                    # YOLOv5 detection script
│   ├── runs/train/glove/weights/    # Trained model weights
│   │   └── best.pt                  # Best trained model
│   ├── test1/images/                # Test images directory
│   └── results/                     # Detection results output
└── Part1_Glove_Detection/
    └── detection_script.py          # Custom detection script
```


2. **Saves Results**:
   - Annotated images: `results/detection_run/`
   - Detection coordinates: `results/detection_run/labels/`
   - JSON log: `results/detection_run/detection_log.json`

### Output Format

**JSON Log Structure**:
```json
{
  "image1.jpg": {
    "filename": "image1.jpg",
    "detections": [
      {
        "label": "glove",
        "confidence": 0.85,
        "bbox": [0.1, 0.2, 0.5, 0.7]
      },
      {
        "label": "no_glove",
        "confidence": 0.92,
        "bbox": [0.3, 0.1, 0.8, 0.6]
      }
    ]
  }
}
```

**Bounding Box Format**: `[x1, y1, x2, y2]` (normalized coordinates 0-1)

## Model Information

- **Architecture**: YOLOv5l (Large model for better accuracy)
- **Input Size**: 640x640 pixels
- **Classes**: 2 (glove, no_glove)
- **Training Resolution**: 640x640 (matches inference resolution)

## Configuration

### Detection Parameters

You can modify these parameters in `detection_script.py`:

```python
'--conf', '0.25',    # Confidence threshold (0.1-0.9)
'--img', '640',      # Input image size
```

### Class Names

Update class mappings in the script:
```python
class_names = {0: "glove", 1: "no_glove"}
```

## File Paths

**Key Directories**:
- Model weights: `/home/r1/Desktop/Work/yolo-tank/yolov5/runs/train/glove/weights/best.pt`
- Test images: `/home/r1/Desktop/Work/yolo-tank/yolov5/test1/images`
- Results: `/home/r1/Desktop/Work/yolo-tank/yolov5/results/detection_run/`


## Model Training


```bash
python train.py --img 640 --batch 32 --epochs 100 --data custom.yaml --weights yolov5l.pt
```

