#!/usr/bin/env python3
"""
validate_multimodal_setup.py - Validation Script for Multimodal BEV System

Comprehensive validation of all components before running the main pipeline:
- Data path validation
- Module import checks  
- YOLO availability testing
- LiDAR data accessibility
- Integration point verification
- Performance baseline establishment

Author: PEM | June 2025
"""

import os
import sys
import json
import traceback
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from termcolor import colored, cprint
import time

def check_imports():
    """Check all required imports and dependencies"""
    cprint("\n🔍 CHECKING IMPORTS AND DEPENDENCIES", "blue", attrs=["bold"])
    
    import_results = {}
    
    # Core dependencies
    core_deps = {
        "numpy": "numpy",
        "pandas": "pandas", 
        "opencv": "cv2",
        "matplotlib": "matplotlib.pyplot",
        "termcolor": "termcolor"
    }
    
    for name, module in core_deps.items():
        try:
            __import__(module)
            import_results[name] = "✅ Available"
            cprint(f"  {name}: ✅", "green")
        except ImportError as e:
            import_results[name] = f"❌ Missing: {e}"
            cprint(f"  {name}: ❌ {e}", "red")
    
    # YOLO dependencies (optional)
    cprint("\n🤖 Checking YOLO dependencies:", "yellow")
    try:
        import torch
        cprint(f"  PyTorch: ✅ {torch.__version__}", "green")
        cprint(f"  CUDA available: {'✅' if torch.cuda.is_available() else '❌'}", 
               "green" if torch.cuda.is_available() else "yellow")
        if torch.cuda.is_available():
            cprint(f"  GPU count: {torch.cuda.device_count()}", "green")
        import_results["torch"] = "✅ Available"
    except ImportError:
        cprint(f"  PyTorch: ❌ Not available", "red")
        import_results["torch"] = "❌ Missing"
    
    try:
        from ultralytics import YOLO
        cprint(f"  Ultralytics YOLO: ✅", "green")
        import_results["yolo"] = "✅ Available"
    except ImportError:
        cprint(f"  Ultralytics YOLO: ❌ Not available", "yellow")
        import_results["yolo"] = "❌ Missing"
    
    return import_results

def validate_data_paths():
    """Validate all data paths and directory structure"""
    cprint("\n📂 VALIDATING DATA PATHS", "blue", attrs=["bold"])
    
    path_results = {}
    
    # Check usable clips file
    usable_clips_file = "analysis_output/usable_clips_for_multimodal.json"
    if os.path.exists(usable_clips_file):
        cprint(f"  ✅ Usable clips file found: {usable_clips_file}", "green")
        path_results["usable_clips"] = "✅ Found"
        
        # Load and validate structure
        try:
            with open(usable_clips_file, 'r') as f:
                clips = json.load(f)
            
            total_events = sum(clip["total_events"] for clip in clips)
            cprint(f"      📊 {len(clips)} clips, {total_events} events", "white")
            path_results["clips_data"] = f"✅ {len(clips)} clips, {total_events} events"
            
        except Exception as e:
            cprint(f"  ❌ Error reading clips file: {e}", "red")
            path_results["clips_data"] = f"❌ Error: {e}"
            return path_results
            
    else:
        cprint(f"  ❌ Usable clips file not found: {usable_clips_file}", "red")
        path_results["usable_clips"] = "❌ Not found"
        return path_results
    
    # Check dangerous clips directory
    dangerous_clips_dir = "/home/jainy007/PEM/triage_brain/llm_input_dataset"
    if os.path.exists(dangerous_clips_dir):
        cprint(f"  ✅ Dangerous clips directory found", "green")
        path_results["dangerous_clips_dir"] = "✅ Found"
    else:
        cprint(f"  ❌ Dangerous clips directory not found: {dangerous_clips_dir}", "red")
        path_results["dangerous_clips_dir"] = "❌ Not found"
    
    # Check a few sample session paths
    cprint(f"\n  🔍 Checking sample session paths:", "yellow")
    session_results = []
    
    for i, clip in enumerate(clips[:3]):  # Check first 3 clips
        session_path = clip["session_path"]
        if os.path.exists(session_path):
            cprint(f"    ✅ Session {i+1}: {os.path.basename(session_path)}", "green")
            session_results.append("✅")
            
            # Check for required subdirectories
            lidar_dir = os.path.join(session_path, "sensors", "lidar")
            ego_file = os.path.join(session_path, "city_SE3_egovehicle.feather")
            
            if os.path.exists(lidar_dir):
                lidar_files = [f for f in os.listdir(lidar_dir) if f.endswith('.feather')]
                cprint(f"        LiDAR: ✅ {len(lidar_files)} files", "green")
            else:
                cprint(f"        LiDAR: ❌ Directory not found", "red")
            
            if os.path.exists(ego_file):
                cprint(f"        Ego trajectory: ✅", "green")
            else:
                cprint(f"        Ego trajectory: ❌ Not found", "red")
                
        else:
            cprint(f"    ❌ Session {i+1}: Path not found", "red")
            session_results.append("❌")
    
    path_results["sample_sessions"] = f"{session_results.count('✅')}/{len(session_results)} accessible"
    
    return path_results

def test_module_imports():
    """Test importing our custom modules"""
    cprint("\n🔧 TESTING MODULE IMPORTS", "blue", attrs=["bold"])
    
    module_results = {}
    
    # Add current directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    modules_to_test = [
        "multimodal_data_loader",
        "yolo_detector", 
        "lidar_processor"
    ]
    
    for module_name in modules_to_test:
        try:
            module = __import__(module_name)
            cprint(f"  ✅ {module_name}: Imported successfully", "green")
            module_results[module_name] = "✅ Success"
            
            # Test key classes
            if module_name == "multimodal_data_loader":
                loader = module.MultimodalDataLoader()
                cprint(f"      📊 Found {len(loader.usable_clips)} usable clips", "white")
                
            elif module_name == "yolo_detector":
                detector = module.OptimizedYOLODetector()
                cprint(f"      🤖 YOLO available: {module.YOLO_AVAILABLE}", "white")
                
            elif module_name == "lidar_processor":
                processor = module.SafeLiDARProcessor()
                cprint(f"      📡 LiDAR processor initialized", "white")
                
        except ImportError as e:
            cprint(f"  ❌ {module_name}: Import failed - {e}", "red")
            module_results[module_name] = f"❌ Import error: {e}"
        except Exception as e:
            cprint(f"  ⚠️  {module_name}: Import succeeded but initialization failed - {e}", "yellow")
            module_results[module_name] = f"⚠️ Init error: {e}"
    
    return module_results

def test_data_loading():
    """Test loading a sample event"""
    cprint("\n📊 TESTING DATA LOADING", "blue", attrs=["bold"])
    
    try:
        # Import our data loader
        from multimodal_data_loader import MultimodalDataLoader
        
        loader = MultimodalDataLoader()
        
        # Get a small test event
        small_events = []
        for clip in loader.usable_clips:
            for event in clip["usable_events"]:
                if 100 <= event["lidar_files_count"] <= 150:  # Medium-sized for testing
                    small_events.append({
                        "hash_id": clip["hash_id"],
                        "event_name": event["event_name"],
                        "session_path": clip["session_path"],
                        "lidar_count": event["lidar_files_count"]
                    })
        
        if not small_events:
            cprint("  ⚠️  No suitable test events found", "yellow")
            return {"data_loading": "⚠️ No test events"}
        
        test_event = small_events[0]
        cprint(f"  🎯 Testing with: {test_event['hash_id']}/{test_event['event_name']}", "yellow")
        cprint(f"      LiDAR files: {test_event['lidar_count']}", "white")
        
        # Test loading
        start_time = time.time()
        event = loader.load_dangerous_event(
            test_event["hash_id"],
            test_event["event_name"], 
            test_event["session_path"]
        )
        load_time = time.time() - start_time
        
        cprint(f"  ✅ Data loading successful in {load_time:.2f}s", "green")
        cprint(f"      Motion data points: {len(event.motion_data)}", "white")
        cprint(f"      LiDAR files: {len(event.lidar_files)}", "white")
        cprint(f"      Ego trajectory points: {len(event.ego_trajectory)}", "white")
        
        # Test video path
        if os.path.exists(event.video_path):
            cprint(f"      Video file: ✅ Found", "green")
            
            # Test video reading
            cap = cv2.VideoCapture(event.video_path)
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cprint(f"      Video frames: {frame_count}", "white")
                cap.release()
            else:
                cprint(f"      Video: ❌ Cannot open", "red")
        else:
            cprint(f"      Video file: ❌ Not found", "red")
        
        # Test LiDAR loading
        if event.lidar_files:
            test_lidar = event.lidar_files[0]
            try:
                lidar_df = pd.read_feather(test_lidar)
                cprint(f"      Sample LiDAR: ✅ {len(lidar_df)} points", "green")
            except Exception as e:
                cprint(f"      Sample LiDAR: ❌ {e}", "red")
        
        return {
            "data_loading": "✅ Success",
            "load_time": f"{load_time:.2f}s",
            "test_event": f"{test_event['hash_id']}/{test_event['event_name']}"
        }
        
    except Exception as e:
        cprint(f"  ❌ Data loading failed: {e}", "red")
        cprint(f"      Traceback: {traceback.format_exc()}", "white")
        return {"data_loading": f"❌ Error: {e}"}

def test_yolo_processing():
    """Test YOLO processing if available"""
    cprint("\n🤖 TESTING YOLO PROCESSING", "blue", attrs=["bold"])
    
    try:
        from yolo_detector import OptimizedYOLODetector, YOLO_AVAILABLE
        
        if not YOLO_AVAILABLE:
            cprint("  ⚠️  YOLO not available - skipping test", "yellow")
            return {"yolo": "⚠️ Not available"}
        
        detector = OptimizedYOLODetector(model_size='n', batch_size=2)
        
        # Create test frames
        test_frames = []
        for i in range(4):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            test_frames.append(frame)
        
        cprint(f"  🔬 Testing with {len(test_frames)} random frames...", "yellow")
        
        # Test batch detection
        start_time = time.time()
        detections = detector.detect_objects_batch(test_frames)
        test_time = time.time() - start_time
        
        total_detections = sum(len(d) for d in detections)
        cprint(f"  ✅ YOLO processing successful", "green")
        cprint(f"      Processing time: {test_time:.2f}s", "white")
        cprint(f"      FPS: {len(test_frames)/test_time:.1f}", "white")
        cprint(f"      Total detections: {total_detections}", "white")
        
        return {
            "yolo": "✅ Success",
            "fps": f"{len(test_frames)/test_time:.1f}",
            "detections": total_detections
        }
        
    except Exception as e:
        cprint(f"  ❌ YOLO testing failed: {e}", "red")
        return {"yolo": f"❌ Error: {e}"}

def test_lidar_processing():
    """Test LiDAR processing capabilities"""
    cprint("\n📡 TESTING LIDAR PROCESSING", "blue", attrs=["bold"])
    
    try:
        from lidar_processor import SafeLiDARProcessor
        
        processor = SafeLiDARProcessor()
        
        # Create test point cloud
        n_points = 10000
        test_points = np.random.randn(n_points, 4) * 10
        test_points[:, 3] = np.random.randint(0, 255, n_points)  # Intensity
        
        # Add some problematic points to test safety
        test_points[0] = [np.inf, 0, 0, 100]
        test_points[1] = [0, np.nan, 0, 50]
        
        cprint(f"  🔬 Testing with {n_points} points (including invalid ones)...", "yellow")
        
        # Test validation
        start_time = time.time()
        validated = processor._validate_points(test_points)
        validation_time = time.time() - start_time
        
        cprint(f"  ✅ Point validation: {len(validated)}/{n_points} valid", "green")
        cprint(f"      Validation time: {validation_time:.3f}s", "white")
        
        # Test classification
        start_time = time.time()
        classified = processor.classify_points_by_features(validated)
        classification_time = time.time() - start_time
        
        cprint(f"  ✅ Point classification successful", "green")
        cprint(f"      Classification time: {classification_time:.3f}s", "white")
        
        for category, points in classified.items():
            cprint(f"      {category}: {len(points)} points", "white")
        
        return {
            "lidar": "✅ Success",
            "validation_time": f"{validation_time:.3f}s",
            "classification_time": f"{classification_time:.3f}s",
            "valid_points": f"{len(validated)}/{n_points}"
        }
        
    except Exception as e:
        cprint(f"  ❌ LiDAR testing failed: {e}", "red")
        return {"lidar": f"❌ Error: {e}"}

def main():
    """Run comprehensive validation"""
    cprint("\n" + "="*80, "cyan")
    cprint("🚀 MULTIMODAL BEV SYSTEM VALIDATION", "cyan", attrs=["bold"])
    cprint("="*80, "cyan")
    
    start_time = time.time()
    results = {}
    
    # Run all validation tests
    results.update(check_imports())
    results.update(validate_data_paths())
    results.update(test_module_imports())
    results.update(test_data_loading())
    results.update(test_yolo_processing())
    results.update(test_lidar_processing())
    
    total_time = time.time() - start_time
    
    # Summary
    cprint("\n" + "="*80, "cyan")
    cprint("📋 VALIDATION SUMMARY", "cyan", attrs=["bold"])
    cprint("="*80, "cyan")
    
    success_count = 0
    warning_count = 0
    error_count = 0
    
    for component, status in results.items():
        if isinstance(status, str):
            if status.startswith("✅"):
                cprint(f"  ✅ {component}: {status[2:]}", "green")
                success_count += 1
            elif status.startswith("⚠️"):
                cprint(f"  ⚠️  {component}: {status[2:]}", "yellow")
                warning_count += 1
            elif status.startswith("❌"):
                cprint(f"  ❌ {component}: {status[2:]}", "red")
                error_count += 1
    
    cprint(f"\n🎯 Results: {success_count} ✅ | {warning_count} ⚠️ | {error_count} ❌", "white")
    cprint(f"⏱️  Total validation time: {total_time:.2f}s", "white")
    
    # Recommendations
    cprint(f"\n💡 RECOMMENDATIONS:", "blue", attrs=["bold"])
    
    if error_count == 0:
        cprint(f"  🎉 System ready! You can proceed with BEV visualization.", "green")
        cprint(f"     Recommended test command:", "white")
        cprint(f"     python bev_visualizer.py --random --max_frames 15", "white")
    else:
        cprint(f"  ⚠️  Fix {error_count} errors before proceeding:", "yellow")
        for component, status in results.items():
            if isinstance(status, str) and status.startswith("❌"):
                cprint(f"     - {component}: {status[2:]}", "white")
    
    if "yolo" in results and results["yolo"].startswith("⚠️"):
        cprint(f"  📦 To enable YOLO: pip install ultralytics torch torchvision", "blue")
    
    return results

if __name__ == "__main__":
    validation_results = main()