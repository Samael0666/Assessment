import subprocess
import json
import os
from pathlib import Path
import glob

subprocess.run([
    'python', '/home/r1/Desktop/Work/yolo-tank/Part1_Glove_Detection/yolov5/detect.py',
    '--weights', '/home/r1/Desktop/Work/yolo-tank/Part1_Glove_Detection/best.pt',
    '--img', '640',
    '--conf', '0.25',
    '--source', '/home/r1/Desktop/Work/yolo-tank/Part1_Glove_Detection/yolov5/test1/images',
    '--project', '/home/r1/Desktop/Work/yolo-tank/Part1_Glove_Detection/yolov5/results',
    '--name', 'detection_run',  
    '--save-txt',  
    '--save-conf' 
])

print("Detection completed!")


results_dir = Path('/home/r1/Desktop/Work/yolo-tank/Part1_Glove_Detection/yolov5/results/detection_run/labels')
json_log = {}

print(f"Looking for results in: {results_dir}")
print(f"Directory exists: {results_dir.exists()}")


if results_dir.exists():
    txt_files = list(results_dir.glob('*.txt'))
    print(f"Found {len(txt_files)} txt files")
    
    for txt_file in txt_files:
        print(f"Processing: {txt_file}")
        filename = txt_file.stem  
        image_filename = filename + '.jpg'  
        
        detections = []
        
        if txt_file.stat().st_size > 0:  
            with open(txt_file, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    print(f"Line parts: {parts}")
                    if len(parts) >= 6:  
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        confidence = float(parts[5])
                        
                        
                        x1 = x_center - width/2
                        y1 = y_center - height/2
                        x2 = x_center + width/2
                        y2 = y_center + height/2
                        
                        class_names = {0: "glove", 1: "no_glove"}  
                        label = class_names.get(class_id, f"class_{class_id}")
                        
                        
                        detections.append({
                            "label": label,
                            "confidence": confidence,
                            "bbox": [x1, y1, x2, y2]
                        })
        else:
            print(f"Empty file: {txt_file}")
        
        
        json_log[image_filename] = {
            "filename": image_filename,
            "detections": detections
        }
else:
    print("Results directory does not exist!")
    
    base_results = Path('/home/r1/Desktop/Work/yolo-tank/Part1_Glove_Detection/yolov5/results')
    if base_results.exists():
        print("Available directories:")
        for d in base_results.iterdir():
            if d.is_dir():
                print(f"  {d.name}")
                labels_dir = d / 'labels'
                if labels_dir.exists():
                    print(f"    -> has labels directory with {len(list(labels_dir.glob('*.txt')))} txt files")


json_output_path = '/home/r1/Desktop/Work/yolo-tank/Part1_Glove_Detection/yolov5/results/detection_run/detection_log.json'
with open(json_output_path, 'w') as f:
    json.dump(json_log, f, indent=2)

print(f"JSON log saved to: {json_output_path}")
print(f"Total images processed: {len(json_log)}")