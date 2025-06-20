#!/usr/bin/env python3
"""
lidar_processor.py - LiDAR Processing and Sensor Fusion Module

Handles LiDAR point cloud processing, sensor fusion, and scene understanding:
- Safe point cloud operations (fixes numpy overflow warnings)
- Efficient spatial transformations
- Robust sensor fusion with vision data
- Enhanced object classification

Author: PEM | June 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from termcolor import colored, cprint
import time

@dataclass
class DetectedObject:
    """Detected object with fused sensor information"""
    object_id: str
    category: str  # pedestrian, vehicle, traffic_sign, etc.
    confidence: float
    
    # 3D position (LiDAR)
    position_3d: Tuple[float, float, float]  # x, y, z in ego coordinates
    lidar_points: np.ndarray
    
    # 2D detection (Camera)
    bbox_2d: Tuple[int, int, int, int]  # x1, y1, x2, y2
    pixel_mask: Optional[np.ndarray]
    
    # Fusion metrics
    fusion_confidence: float
    distance_ego: float

class SafeLiDARProcessor:
    """LiDAR processor with safe mathematical operations"""
    
    def __init__(self, ego_range: float = 100.0):
        self.ego_range = ego_range
        self.processing_times = []
    
    def load_lidar_frame(self, lidar_file: str) -> np.ndarray:
        """Load and validate LiDAR frame with error handling"""
        try:
            lidar_df = pd.read_feather(lidar_file)
            points = lidar_df[['x', 'y', 'z', 'intensity']].values
            
            # Validate and clean points
            points = self._validate_points(points)
            return points
            
        except Exception as e:
            cprint(f"❌ Error loading LiDAR file: {e}", "red")
            return np.array([]).reshape(0, 4)
    
    def _validate_points(self, points: np.ndarray) -> np.ndarray:
        """Validate and clean point cloud data to prevent overflow"""
        if len(points) == 0:
            return points
        
        # Check for invalid values
        valid_mask = (
            np.isfinite(points).all(axis=1) &  # No inf/nan
            (np.abs(points[:, :3]) < 1000).all(axis=1) &  # Reasonable coordinates
            (points[:, 3] >= 0) & (points[:, 3] <= 255)  # Valid intensity
        )
        
        valid_points = points[valid_mask]
        
        if len(valid_points) < len(points):
            removed = len(points) - len(valid_points)
            cprint(f"⚠️  Removed {removed} invalid points", "yellow")
        
        return valid_points
    
    def get_points_in_camera_frustum(self, lidar_points: np.ndarray, camera_params: Dict) -> np.ndarray:
        """Filter LiDAR points within camera FOV with safe operations"""
        if len(lidar_points) == 0:
            return lidar_points
        
        x, y, z = lidar_points[:, 0], lidar_points[:, 1], lidar_points[:, 2]
        
        # Safe frustum filtering - avoid extreme values
        fov_angle = camera_params.get('fov_angle', 90)  # degrees
        fov_rad = np.radians(fov_angle / 2)
        
        # Forward direction with safe angular bounds
        forward_mask = x > 0.1  # Avoid division by zero
        
        # Angular bounds (tan(fov/2))
        tan_fov = np.tan(fov_rad)
        angular_mask = np.abs(y[forward_mask] / x[forward_mask]) < tan_fov
        
        # Height bounds
        height_mask = (z > -3) & (z < 15)  # Reasonable height range
        
        # Combine masks safely
        in_fov = np.zeros(len(lidar_points), dtype=bool)
        in_fov[forward_mask] = angular_mask
        final_mask = in_fov & height_mask
        
        return lidar_points[final_mask]
    
    def project_to_image_safe(self, lidar_points: np.ndarray, camera_params: Dict) -> np.ndarray:
        """Safe 3D to 2D projection avoiding overflow warnings"""
        if len(lidar_points) == 0:
            return np.array([]).reshape(0, 2)
        
        x, y, z = lidar_points[:, 0], lidar_points[:, 1], lidar_points[:, 2]
        
        # Safe projection parameters
        focal_length = camera_params.get('focal_length', 800)
        image_width = camera_params.get('width', 1550)
        image_height = camera_params.get('height', 2048)
        
        # Avoid division by very small numbers
        x_safe = np.where(x > 0.1, x, 0.1)
        
        # Safe projection with clipping
        u = np.clip((focal_length * y / x_safe) + image_width / 2, 0, image_width)
        v = np.clip((focal_length * z / x_safe) + image_height / 2, 0, image_height)
        
        return np.column_stack([u, v])
    
    def classify_points_by_features(self, points: np.ndarray) -> Dict[str, np.ndarray]:
        """Enhanced point classification with safe operations"""
        if len(points) == 0:
            return {category: np.array([]).reshape(0, 4) for category in 
                   ['road_surface', 'buildings', 'vegetation', 'infrastructure', 'vehicles', 'unknown']}
        
        x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
        classified = {}
        
        # Road surface: low height, moderate intensity
        road_mask = (z > -2.0) & (z < 0.5) & (intensity > 10) & (intensity < 120)
        classified['road_surface'] = points[road_mask]
        
        # Buildings: high objects with varied intensity
        building_mask = (z > 3.0) & (z < 25.0)
        classified['buildings'] = points[building_mask]
        
        # Vegetation: medium height, low intensity, scattered
        vegetation_mask = (z > 0.3) & (z < 6.0) & (intensity < 60)
        classified['vegetation'] = points[vegetation_mask]
        
        # Infrastructure (signs, lights): high intensity, specific height
        infra_mask = (z > 2.5) & (z < 8.0) & (intensity > 180)
        classified['infrastructure'] = points[infra_mask]
        
        # Potential vehicles: moderate height and intensity
        vehicle_mask = (
            (z > 0.8) & (z < 2.8) & 
            (intensity > 100) & (intensity < 220) & 
            ~infra_mask  # Exclude infrastructure
        )
        classified['vehicles'] = points[vehicle_mask]
        
        # Everything else
        all_masks = road_mask | building_mask | vegetation_mask | infra_mask | vehicle_mask
        classified['unknown'] = points[~all_masks]
        
        return classified

class EnhancedSensorFusion:
    """Enhanced sensor fusion with safe mathematical operations"""
    
    def __init__(self, yolo_detector=None):
        self.lidar_processor = SafeLiDARProcessor()
        self.yolo_detector = yolo_detector
        self.fusion_times = []
    
    def associate_detections_safe(self, vision_detections: List[Dict], 
                                lidar_points: np.ndarray, 
                                camera_params: Dict) -> List[DetectedObject]:
        """Safe detection association avoiding overflow warnings"""
        start_time = time.time()
        fused_objects = []
        
        if len(lidar_points) == 0 or not vision_detections:
            return fused_objects
        
        try:
            # Get points in camera frustum
            frustum_points = self.lidar_processor.get_points_in_camera_frustum(
                lidar_points, camera_params
            )
            
            if len(frustum_points) == 0:
                return fused_objects
            
            # Project to image coordinates
            image_coords = self.lidar_processor.project_to_image_safe(
                frustum_points, camera_params
            )
            
            # Associate each vision detection with LiDAR points
            for i, detection in enumerate(vision_detections):
                x1, y1, x2, y2 = detection['bbox']
                
                # Find points within bounding box
                in_bbox = (
                    (image_coords[:, 0] >= x1) & (image_coords[:, 0] <= x2) &
                    (image_coords[:, 1] >= y1) & (image_coords[:, 1] <= y2)
                )
                
                associated_points = frustum_points[in_bbox]
                
                if len(associated_points) >= 5:  # Minimum points for valid object
                    fused_obj = self._create_fused_object(
                        detection, associated_points, i
                    )
                    if fused_obj:
                        fused_objects.append(fused_obj)
            
            # Track performance
            fusion_time = time.time() - start_time
            self.fusion_times.append(fusion_time)
            
        except Exception as e:
            cprint(f"⚠️  Fusion association failed: {e}", "yellow")
        
        return fused_objects
    
    def _create_fused_object(self, detection: Dict, points: np.ndarray, idx: int) -> Optional[DetectedObject]:
        """Create fused object with safe distance calculation"""
        try:
            # Safe 3D position calculation
            if len(points) == 0:
                return None
            
            # Compute centroid safely
            position_3d = np.mean(points[:, :3], axis=0)
            
            # Safe distance calculation
            distance = float(np.sqrt(np.sum(position_3d**2)))
            
            # Validate distance is reasonable
            if not np.isfinite(distance) or distance > 200:
                return None
            
            # Calculate fusion confidence
            bbox_area = (detection['bbox'][2] - detection['bbox'][0]) * \
                       (detection['bbox'][3] - detection['bbox'][1])
            point_density = len(points) / max(bbox_area, 1)  # Avoid division by zero
            
            fusion_confidence = min(1.0, 
                detection['confidence'] * min(1.0, point_density / 0.001)
            )
            
            fused_object = DetectedObject(
                object_id=f"{detection['category']}_{idx}",
                category=detection['category'],
                confidence=detection['confidence'],
                position_3d=tuple(position_3d),
                lidar_points=points,
                bbox_2d=detection['bbox'],
                pixel_mask=None,
                fusion_confidence=fusion_confidence,
                distance_ego=distance
            )
            
            return fused_object
            
        except Exception as e:
            cprint(f"⚠️  Failed to create fused object: {e}", "yellow")
            return None
    
    def generate_scene_description(self, fused_objects: List[DetectedObject], 
                                 event_metadata: Dict = None) -> Dict:
        """Generate semantic scene description"""
        if event_metadata is None:
            event_metadata = {}
        
        scene_description = {
            "scene_type": "urban_intersection",
            "detected_objects": [],
            "spatial_relationships": [],
            "risk_assessment": {
                "primary_risks": [],
                "interaction_zones": []
            },
            "infrastructure": {
                "traffic_control": [],
                "road_geometry": "intersection"
            }
        }
        
        # Process detected objects
        for obj in fused_objects:
            obj_desc = {
                "type": obj.category,
                "position": {
                    "distance_m": round(obj.distance_ego, 1),
                    "bearing": self._calculate_bearing(obj.position_3d),
                    "coordinates_3d": [round(x, 1) for x in obj.position_3d]
                },
                "confidence": round(obj.fusion_confidence, 2),
                "size": self._estimate_size(len(obj.lidar_points))
            }
            
            scene_description["detected_objects"].append(obj_desc)
            
            # Infrastructure classification
            if obj.category in ["traffic_sign", "traffic_light"]:
                scene_description["infrastructure"]["traffic_control"].append({
                    "type": obj.category,
                    "position": obj_desc["position"]
                })
            
            # Risk assessment
            if obj.category == "pedestrian" and obj.distance_ego < 20:
                scene_description["risk_assessment"]["primary_risks"].append({
                    "type": "pedestrian_proximity",
                    "distance": obj.distance_ego,
                    "severity": "high" if obj.distance_ego < 10 else "medium"
                })
        
        return scene_description
    
    def _calculate_bearing(self, position_3d: Tuple[float, float, float]) -> str:
        """Calculate bearing relative to ego vehicle"""
        x, y, z = position_3d
        angle = np.arctan2(y, x) * 180 / np.pi
        
        bearings = [
            (-22.5, 22.5, "front"),
            (22.5, 67.5, "front_right"),
            (67.5, 112.5, "right"),
            (112.5, 157.5, "rear_right"),
            (-157.5, -112.5, "rear_left"),
            (-112.5, -67.5, "left"),
            (-67.5, -22.5, "front_left")
        ]
        
        for min_angle, max_angle, bearing in bearings:
            if min_angle <= angle < max_angle:
                return bearing
        
        return "rear"  # angles >= 157.5 or < -157.5
    
    def _estimate_size(self, point_count: int) -> str:
        """Estimate object size based on LiDAR point count"""
        if point_count < 50:
            return "small"
        elif point_count < 200:
            return "medium"
        else:
            return "large"
    
    def get_performance_stats(self) -> Dict:
        """Get fusion performance statistics"""
        if not self.fusion_times:
            return {"status": "no_fusions_yet"}
        
        return {
            "total_fusions": len(self.fusion_times),
            "avg_fusion_time": np.mean(self.fusion_times),
            "total_fusion_time": sum(self.fusion_times)
        }

def test_lidar_processor():
    """Test LiDAR processor with safe operations"""
    cprint("\n🧪 Testing LiDAR Processor", "cyan", attrs=["bold"])
    
    # Create test point cloud
    n_points = 10000
    test_points = np.random.randn(n_points, 4) * 10
    test_points[:, 3] = np.random.randint(0, 255, n_points)  # Intensity
    
    # Add some extreme values to test safety
    test_points[0] = [np.inf, 0, 0, 100]  # Test infinity handling
    test_points[1] = [0, np.nan, 0, 50]   # Test NaN handling
    test_points[2] = [1000, 1000, 1000, 300]  # Test extreme values
    
    processor = SafeLiDARProcessor()
    
    # Test validation
    cprint("🔧 Testing point validation...", "blue")
    validated_points = processor._validate_points(test_points)
    cprint(f"✅ Validated {len(validated_points)}/{len(test_points)} points", "green")
    
    # Test classification
    cprint("🏷️  Testing point classification...", "blue")
    classified = processor.classify_points_by_features(validated_points)
    
    for category, points in classified.items():
        cprint(f"  {category}: {len(points)} points", "white")
    
    cprint("✅ LiDAR processor test complete", "green")

if __name__ == "__main__":
    test_lidar_processor()