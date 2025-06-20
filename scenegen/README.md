# 🚀 Quick Start Guide - Multimodal BEV Visualization System

## **System Overview**
Your multimodal BEV (Bird's Eye View) system fuses LiDAR point clouds with YOLO camera detections to create comprehensive 4-panel animated visualizations.

### **Components:**
- 📊 **MultimodalDataLoader**: Loads dangerous events data
- 🤖 **OptimizedYOLODetector**: GPU-accelerated batch object detection  
- 📡 **SafeLiDARProcessor**: Robust point cloud processing with sensor fusion
- 🎬 **BEVVisualizerMain**: Orchestrates everything into animated GIFs

---

## **🔧 Before You Start**

### **1. Run System Validation**
```bash
python validate_multimodal_setup.py
```
This checks all dependencies, data paths, and integration points.

### **2. Apply Optimizations**
```bash
python quick_fixes_and_optimizations.py
```
This optimizes system settings and creates output directories.

---

## **📊 Your Dataset**
- **28 clips** with **32 dangerous events**
- **Season distribution**: Spring 2020 (7), Autumn 2020 (10), Summer 2020 (8), Winter 2021 (3)
- **Event types**: dangerous_event_1 (20), dangerous_event_2 (9), dangerous_event_3 (3)
- **LiDAR files per event**: 75-346 files (avg: 190)

### **Recommended Test Events** (medium size, 100-200 LiDAR files):
1. `04akO6mLeIFQRjbq9XwT71QNx0IJ0sTy/dangerous_event_1` (177 files)
2. `05lBLQJs4ilyORCox6j9ndWAKZc31rs9/dangerous_event_1` (138 files)
3. `0Pl6cyE1PWIIFx3ldUI1C0sP8H5qpRp6/dangerous_event_2` (179 files)

---

## **🎯 Quick Test Commands**

### **List All Available Events**
```bash
python bev_visualizer.py --list
```

### **Generate Random Visualization (Quick Test)**
```bash
python bev_visualizer.py --random --max_frames 15
```

### **Generate Specific Event**
```bash
python bev_visualizer.py --event_id 04akO6mLeIFQRjbq9XwT71QNx0IJ0sTy --max_frames 20
```

### **High-Quality Visualization**
```bash
python bev_visualizer.py --random --max_frames 25 --batch_size 8
```

---

## **🎞️ Output Structure**

Your GIFs will be saved to:
```
bev_visualizations/
├── gifs/                    # Animated GIF outputs
├── analysis/               # JSON analysis files
├── frames/                 # Individual frame exports
└── temp/                   # Temporary processing files
```

### **4-Panel GIF Layout:**
1. **Top-Left**: Camera view with YOLO bounding boxes
2. **Top-Right**: LiDAR BEV with point classifications  
3. **Bottom-Left**: Fusion confidence overlay
4. **Bottom-Right**: Semantic scene understanding

---

## **⚡ Performance Optimization**

### **Memory Usage:**
- **Low memory** (<4GB available): `--max_frames 10 --batch_size 2`
- **Medium memory** (4-8GB): `--max_frames 20 --batch_size 4`
- **High memory** (>8GB): `--max_frames 30 --batch_size 8`

### **Processing Speed:**
- **YOLO available**: ~2-5 FPS processing
- **LiDAR only**: ~5-10 FPS processing
- **Batch processing**: 2-4x faster than single frame

---

## **🔍 Troubleshooting**

### **Common Issues:**

**1. "YOLO not available"**
```bash
pip install ultralytics torch torchvision
```

**2. "LiDAR files not found"**
- Check if session paths in JSON match actual file locations
- Verify `/mnt/db/av_dataset/` is mounted correctly

**3. "Memory Error"**
- Reduce `--max_frames` to 10-15
- Use smaller `--batch_size` (2-4)
- Close other applications

**4. "Video cannot be opened"**
- Check if video files exist in `/home/jainy007/PEM/triage_brain/llm_input_dataset/`
- Some events may not have video - system will use LiDAR only

### **Debug Mode:**
Add verbose output to any command:
```bash
python bev_visualizer.py --random --max_frames 15 -v
```

---

## **🎨 Customization Options**

### **Modify Parameters in Code:**

**bev_visualizer.py:**
```python
self.bev_range = 80          # BEV visualization range (meters)
self.bev_resolution = 0.2    # Grid resolution
```

**yolo_detector.py:**
```python
confidence_threshold = 0.5   # YOLO detection confidence
model_size = 'n'            # 'n'ano, 's'mall, 'm'edium, 'l'arge
```

**lidar_processor.py:**
```python
ego_range = 100.0           # LiDAR processing range
```

---

## **📈 Expected Output**

### **Successful Run Example:**
```
🚀 BEV Visualizer initialized
✅ YOLO detection enabled
🎬 Generating Multi-Modal BEV GIF for 04akO6mLeIFQRjbq9XwT71QNx0IJ0sTy/dangerous_event_1
📊 Loading multimodal event data...
🎥 Extracting video frames...
📊 Processing 25 frames from 177 total
🎯 YOLO processed 25 frames in 3.45s (0.138s/frame)
🔄 Processing frame 1/25
🎞️ Creating 4-panel animated GIF...
💾 Saving GIF to bev_visualizations/gifs/04akO6mLeIFQRjbq9XwT71QNx0IJ0sTy_dangerous_event_1_multimodal_bev.gif...
✅ Multi-modal BEV GIF saved: bev_visualizations/gifs/04akO6mLeIFQRjbq9XwT71QNx0IJ0sTy_dangerous_event_1_multimodal_bev.gif
📁 GIF location: bev_visualizations/gifs/04akO6mLeIFQRjbq9XwT71QNx0IJ0sTy_dangerous_event_1_multimodal_bev.gif
📏 File size: 8.3 MB
```

### **Analysis Output:**
Each run creates a JSON analysis file with:
- Detection statistics
- Performance metrics  
- Event metadata
- Processing summary

---

## **🚀 Ready to Start!**

**Recommended first command:**
```bash
python bev_visualizer.py --random --max_frames 15
```

This will:
1. Select a random dangerous event
2. Process 15 frames for quick results
3. Generate a 4-panel animated GIF
4. Save analysis data
5. Display performance statistics

**Expected processing time**: 1-3 minutes per event
**Expected GIF size**: 5-15 MB depending on complexity

---

## **💡 Tips for Best Results**

1. **Start small**: Use `--max_frames 10-15` for initial testing
2. **Monitor memory**: Watch for memory warnings in output
3. **Choose good events**: Medium-sized events (100-200 LiDAR files) work best
4. **YOLO optional**: System works with LiDAR-only if YOLO unavailable
5. **Batch processing**: Higher batch sizes are faster but use more memory

**Happy visualizing!** 🎉