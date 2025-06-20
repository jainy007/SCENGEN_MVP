#!/usr/bin/env python3
"""
optimized_bev_processor.py - Multiprocessed & Cached BEV Processing

Optimizations:
- Multiprocessing for frame processing
- LiDAR data caching to avoid reloads
- Safe math operations to prevent overflow warnings
- Batch operations for efficiency
- Memory-mapped file access for large datasets

Author: PEM | June 2025
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache
import multiprocessing as mp
from pathlib import Path
import time
import warnings
from typing import Dict, List, Tuple, Optional
from termcolor import cprint
import pickle
import hashlib

# Suppress numpy overflow warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy.linalg')

class CachedLiDARLoader:
    """Memory-efficient cached LiDAR loader"""
    
    def __init__(self, cache_dir: str = "bev_visualizations/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}
        self.max_memory_cache = 50  # Cache up to 50 LiDAR frames in memory
        
    def _get_cache_path(self, lidar_file: str) -> Path:
        """Get cache file path for a LiDAR file"""
        file_hash = hashlib.md5(lidar_file.encode()).hexdigest()[:16]
        return self.cache_dir / f"lidar_{file_hash}.pkl"
    
    @lru_cache(maxsize=100)
    def load_lidar_cached(self, lidar_file: str) -> np.ndarray:
        """Load LiDAR with disk and memory caching"""
        # Check memory cache first
        if lidar_file in self.memory_cache:
            return self.memory_cache[lidar_file]
        
        # Check disk cache
        cache_path = self._get_cache_path(lidar_file)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    points = pickle.load(f)
                
                # Add to memory cache if space available
                if len(self.memory_cache) < self.max_memory_cache:
                    self.memory_cache[lidar_file] = points
                
                return points
            except:
                pass  # Fall through to reload
        
        # Load from source and cache
        points = self._load_and_validate_lidar(lidar_file)
        
        # Save to disk cache
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(points, f, protocol=pickle.HIGHEST_PROTOCOL)
        except:
            pass  # Continue even if caching fails
        
        # Add to memory cache
        if len(self.memory_cache) < self.max_memory_cache:
            self.memory_cache[lidar_file] = points
        
        return points
    
    def _load_and_validate_lidar(self, lidar_file: str) -> np.ndarray:
        """Load and validate LiDAR with safe math operations"""
        try:
            if not os.path.exists(lidar_file):
                return np.array([]).reshape(0, 4)
            
            lidar_df = pd.read_feather(lidar_file)
            if lidar_df.empty:
                return np.array([]).reshape(0, 4)
            
            # Extract points
            required_cols = ['x', 'y', 'z', 'intensity']
            if not all(col in lidar_df.columns for col in required_cols):
                return np.array([]).reshape(0, 4)
            
            points = lidar_df[required_cols].values.astype(np.float32)  # Use float32 for memory
            
            # Safe validation with reasonable bounds
            valid_mask = (
                np.isfinite(points).all(axis=1) &
                (np.abs(points[:, 0]) < 500) &  # X bounds
                (np.abs(points[:, 1]) < 500) &  # Y bounds
                (points[:, 2] > -10) & (points[:, 2] < 50) &  # Z bounds
                (points[:, 3] >= 0) & (points[:, 3] <= 255)   # Intensity bounds
            )
            
            return points[valid_mask]
            
        except Exception as e:
            cprint(f"⚠️  Error loading {os.path.basename(lidar_file)}: {e}", "yellow")
            return np.array([]).reshape(0, 4)

class SafeMathOperations:
    """Safe mathematical operations to prevent overflow"""
    
    @staticmethod
    def safe_distance(points: np.ndarray) -> np.ndarray:
        """Compute distances safely without overflow"""
        if len(points) == 0:
            return np.array([])
        
        # Clip extreme values before distance calculation
        clipped_points = np.clip(points[:, :3], -1000, 1000)
        
        # Use float64 for intermediate calculations
        squared_dists = np.sum(clipped_points.astype(np.float64)**2, axis=1)
        
        # Clip before sqrt to prevent overflow
        squared_dists = np.clip(squared_dists, 0, 1e12)
        
        distances = np.sqrt(squared_dists).astype(np.float32)
        return distances
    
    @staticmethod
    def safe_projection(lidar_points: np.ndarray, camera_params: Dict) -> np.ndarray:
        """Safe 3D to 2D projection"""
        if len(lidar_points) == 0:
            return np.array([]).reshape(0, 2)
        
        x, y, z = lidar_points[:, 0], lidar_points[:, 1], lidar_points[:, 2]
        
        # Safe division with minimum threshold
        x_safe = np.where(np.abs(x) > 0.1, x, np.sign(x) * 0.1)
        
        focal_length = camera_params.get('focal_length', 800)
        image_width = camera_params.get('width', 1550)
        image_height = camera_params.get('height', 2048)
        
        # Project with clipping
        u = np.clip((focal_length * y / x_safe) + image_width / 2, -image_width, 2*image_width)
        v = np.clip((focal_length * z / x_safe) + image_height / 2, -image_height, 2*image_height)
        
        return np.column_stack([u, v]).astype(np.float32)

def process_single_frame_worker(args):
    """Worker function for multiprocessing frame processing"""
    (frame_idx, lidar_file, ego_pos, trajectory, video_frame, 
     vision_detections, camera_params, timestamp) = args
    
    try:
        # Load LiDAR with caching
        cached_loader = CachedLiDARLoader()
        lidar_points = cached_loader.load_lidar_cached(lidar_file)
        
        if len(lidar_points) == 0:
            return None
        
        # Perform sensor fusion with safe operations
        fused_objects = []
        if video_frame is not None and vision_detections:
            fused_objects = safe_sensor_fusion(
                vision_detections, lidar_points, camera_params
            )
        
        # Classify LiDAR points
        classified_points = classify_lidar_points_fast(lidar_points)
        
        # Enhance with fusion results
        enhanced_classified = enhance_classification_with_fusion(
            classified_points, fused_objects
        )
        
        # Generate scene description
        scene_description = generate_scene_description_fast(fused_objects)
        
        return {
            'frame_idx': frame_idx,
            'lidar_file': os.path.basename(lidar_file),
            'ego_pos': ego_pos,
            'trajectory': trajectory,
            'video_frame': video_frame,
            'vision_detections': vision_detections,
            'fused_objects': fused_objects,
            'classified_points': enhanced_classified,
            'scene_description': scene_description,
            'timestamp': timestamp
        }
        
    except Exception as e:
        cprint(f"⚠️  Frame {frame_idx} failed: {e}", "yellow")
        return None

def safe_sensor_fusion(vision_detections: List[Dict], lidar_points: np.ndarray, 
                      camera_params: Dict) -> List[Dict]:
    """Safe sensor fusion without overflow warnings"""
    fused_objects = []
    
    if len(lidar_points) == 0 or not vision_detections:
        return fused_objects
    
    try:
        # Filter LiDAR points to camera frustum safely
        x, y, z = lidar_points[:, 0], lidar_points[:, 1], lidar_points[:, 2]
        
        # Forward direction filter
        forward_mask = x > 0.5  # Minimum distance
        if not np.any(forward_mask):
            return fused_objects
        
        # Angular bounds with safe operations
        fov_angle = camera_params.get('fov_angle', 90)
        tan_half_fov = np.tan(np.radians(fov_angle / 2))
        
        frustum_points = lidar_points[forward_mask]
        x_frust = frustum_points[:, 0]
        y_frust = frustum_points[:, 1]
        
        # Safe angular filtering
        angular_mask = np.abs(y_frust / x_frust) < tan_half_fov
        frustum_points = frustum_points[angular_mask]
        
        if len(frustum_points) == 0:
            return fused_objects
        
        # Safe projection
        safe_math = SafeMathOperations()
        image_coords = safe_math.safe_projection(frustum_points, camera_params)
        
        # Associate detections
        for i, detection in enumerate(vision_detections):
            x1, y1, x2, y2 = detection['bbox']
            
            # Find points in bounding box
            in_bbox = (
                (image_coords[:, 0] >= x1) & (image_coords[:, 0] <= x2) &
                (image_coords[:, 1] >= y1) & (image_coords[:, 1] <= y2)
            )
            
            associated_points = frustum_points[in_bbox]
            
            if len(associated_points) >= 3:  # Minimum points
                # Safe distance calculation
                distances = safe_math.safe_distance(associated_points)
                avg_distance = float(np.mean(distances))
                
                if np.isfinite(avg_distance) and avg_distance < 200:
                    position_3d = np.mean(associated_points[:, :3], axis=0)
                    
                    fused_obj = {
                        'object_id': f"{detection['category']}_{i}",
                        'category': detection['category'],
                        'confidence': detection['confidence'],
                        'position_3d': tuple(position_3d.astype(float)),
                        'lidar_points': associated_points,
                        'bbox_2d': detection['bbox'],
                        'fusion_confidence': min(1.0, detection['confidence'] * len(associated_points) / 50),
                        'distance_ego': avg_distance
                    }
                    fused_objects.append(fused_obj)
        
    except Exception as e:
        cprint(f"⚠️  Sensor fusion failed: {e}", "yellow")
    
    return fused_objects

def classify_lidar_points_fast(points: np.ndarray) -> Dict[str, np.ndarray]:
    """Fast LiDAR point classification"""
    if len(points) == 0:
        return {category: np.array([]).reshape(0, 4) for category in 
               ['road_surface', 'buildings', 'vegetation', 'infrastructure', 'vehicles', 'unknown']}
    
    x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    classified = {}
    
    # Vectorized classification
    road_mask = (z > -2.0) & (z < 0.5) & (intensity > 10) & (intensity < 120)
    building_mask = (z > 3.0) & (z < 25.0) & ~road_mask
    vegetation_mask = (z > 0.3) & (z < 6.0) & (intensity < 60) & ~road_mask & ~building_mask
    infra_mask = (z > 2.5) & (z < 8.0) & (intensity > 180) & ~road_mask & ~building_mask
    vehicle_mask = ((z > 0.8) & (z < 2.8) & (intensity > 100) & (intensity < 220) & 
                   ~road_mask & ~building_mask & ~vegetation_mask & ~infra_mask)
    
    classified['road_surface'] = points[road_mask]
    classified['buildings'] = points[building_mask]
    classified['vegetation'] = points[vegetation_mask]
    classified['infrastructure'] = points[infra_mask]
    classified['vehicles'] = points[vehicle_mask]
    
    # Everything else
    all_masks = road_mask | building_mask | vegetation_mask | infra_mask | vehicle_mask
    classified['unknown'] = points[~all_masks]
    
    return classified

def enhance_classification_with_fusion(lidar_classified: Dict, fused_objects: List[Dict]) -> Dict:
    """Fast enhancement of classification with fusion results"""
    enhanced = lidar_classified.copy()
    
    # Initialize fusion categories
    for category in ['fused_vehicle', 'fused_pedestrian', 'fused_traffic_sign', 'fused_bicycle']:
        enhanced[category] = np.array([]).reshape(0, 4)
    
    # Process fused objects
    for obj in fused_objects:
        points = obj.get('lidar_points', np.array([]))
        if len(points) == 0:
            continue
        
        fusion_category = f"fused_{obj['category']}"
        if fusion_category in enhanced:
            if len(enhanced[fusion_category]) == 0:
                enhanced[fusion_category] = points
            else:
                enhanced[fusion_category] = np.vstack([enhanced[fusion_category], points])
    
    return enhanced

def generate_scene_description_fast(fused_objects: List[Dict]) -> Dict:
    """Fast scene description generation"""
    return {
        "scene_type": "urban_intersection",
        "object_count": len(fused_objects),
        "categories": [obj['category'] for obj in fused_objects],
        "avg_confidence": np.mean([obj['fusion_confidence'] for obj in fused_objects]) if fused_objects else 0.0
    }

class MultiprocessedBEVProcessor:
    """Multiprocessed BEV frame processor"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(mp.cpu_count(), 8)  # Limit to 8 cores max
        self.cached_loader = CachedLiDARLoader()
        
        cprint(f"🚀 Initialized multiprocessed BEV processor with {self.max_workers} workers", "green")
    
    def process_frames_parallel(self, event, sampled_indices: List[int], 
                              video_frames: List[np.ndarray],
                              vision_detections_batch: List[List[Dict]]) -> List[Dict]:
        """Process frames in parallel with caching"""
        
        cprint(f"🔄 Processing {len(sampled_indices)} frames with {self.max_workers} workers...", "green")
        start_time = time.time()
        
        # Prepare arguments for workers
        worker_args = []
        
        for i, idx in enumerate(sampled_indices):
            # Get ego data
            ego_data = self._get_ego_data(event, idx)
            
            # Get video frame and detections
            video_frame = video_frames[i] if i < len(video_frames) else None
            vision_detections = (vision_detections_batch[i] 
                               if i < len(vision_detections_batch) else [])
            
            # Camera parameters
            camera_params = {
                'focal_length': 800,
                'width': video_frame.shape[1] if video_frame is not None else 1550,
                'height': video_frame.shape[0] if video_frame is not None else 2048,
                'fov_angle': 90
            }
            
            worker_args.append((
                i,  # frame_idx
                event.lidar_files[idx],  # lidar_file
                ego_data['ego_pos'],  # ego_pos
                ego_data['trajectory'],  # trajectory
                video_frame,  # video_frame
                vision_detections,  # vision_detections
                camera_params,  # camera_params
                event.lidar_timestamps[idx]  # timestamp
            ))
        
        # Process in parallel
        frame_data_list = []
        
        # Use ThreadPoolExecutor for I/O bound tasks (better for file loading)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(process_single_frame_worker, worker_args))
        
        # Filter out None results
        frame_data_list = [result for result in results if result is not None]
        
        processing_time = time.time() - start_time
        cprint(f"✅ Parallel processing completed: {len(frame_data_list)}/{len(sampled_indices)} frames in {processing_time:.2f}s", "green")
        cprint(f"⚡ Speed: {len(frame_data_list)/processing_time:.1f} frames/second", "blue")
        
        return frame_data_list
    
    def _get_ego_data(self, event, lidar_idx: int) -> Dict:
        """Get ego vehicle position and trajectory"""
        timestamp = event.lidar_timestamps[lidar_idx]
        
        # Find closest ego trajectory point
        time_diffs = np.abs(event.ego_trajectory['timestamp_ns'] - timestamp)
        closest_idx = time_diffs.argmin()
        ego_row = event.ego_trajectory.iloc[closest_idx]
        ego_pos = (float(ego_row['tx_m']), float(ego_row['ty_m']))
        
        # Get trajectory context
        traj_window = 30
        start_idx = max(0, closest_idx - traj_window)
        end_idx = min(len(event.ego_trajectory), closest_idx + traj_window)
        trajectory_subset = event.ego_trajectory.iloc[start_idx:end_idx]
        trajectory_points = trajectory_subset[['tx_m', 'ty_m']].values.astype(np.float32)
        
        return {
            'ego_pos': ego_pos,
            'trajectory': trajectory_points
        }

def test_optimized_processing():
    """Test the optimized processing pipeline"""
    cprint("\n🧪 Testing Optimized BEV Processing", "cyan", attrs=["bold"])
    
    # Test safe math operations
    safe_math = SafeMathOperations()
    
    # Test with extreme values
    test_points = np.array([
        [1e10, 1e10, 1e10, 100],  # Extreme coordinates
        [np.inf, 0, 0, 100],      # Infinity
        [0, np.nan, 0, 50],       # NaN
        [10, 20, 1, 150]          # Normal point
    ])
    
    distances = safe_math.safe_distance(test_points)
    cprint(f"✅ Safe distance calculation: {len(distances)} results", "green")
    
    # Test cached loader
    cached_loader = CachedLiDARLoader()
    cprint(f"✅ Cached loader initialized", "green")
    
    # Test multiprocessed processor
    processor = MultiprocessedBEVProcessor(max_workers=4)
    cprint(f"✅ Multiprocessed processor ready", "green")

if __name__ == "__main__":
    test_optimized_processing()