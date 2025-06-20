#!/usr/bin/env python3
"""
multimodal_data_loader.py - Comprehensive Multimodal Data Loader

Loads and processes dangerous event data from multiple modalities:
- Video segments (MP4)
- LiDAR point clouds (feather files)  
- Motion data (velocity, acceleration, jerk)
- Event metadata (annotations, timing)
- Ego vehicle trajectory

Prepares data for LLM scenario understanding and generation.

Author: PEM | June 2025
"""

import os
import json
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from termcolor import colored, cprint
import time

# Add workspace root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class DangerousEvent:
    """Container for a single dangerous event's multimodal data"""
    # Identifiers
    hash_id: str
    event_name: str
    session_path: str
    
    # Metadata
    event_metadata: Dict
    motion_data: List[Dict]
    video_path: str
    keyframes_dir: str
    
    # LiDAR data
    lidar_files: List[str]
    lidar_timestamps: List[int]
    ego_trajectory: pd.DataFrame
    
    # Processed data
    video_frames: Optional[np.ndarray] = None
    representative_frames: Optional[List[np.ndarray]] = None
    lidar_summary: Optional[Dict] = None

class MultimodalDataLoader:
    """Loads and processes multimodal dangerous event data"""
    
    def __init__(self, usable_clips_file: str = "analysis_output/usable_clips_for_multimodal.json"):
        self.usable_clips_file = usable_clips_file
        self.dangerous_clips_dir = "/home/jainy007/PEM/triage_brain/llm_input_dataset"
        self.events = []
        self.load_usable_clips()
    
    def load_usable_clips(self):
        """Load the list of usable clips from analysis"""
        cprint(f"📂 Loading usable clips from {self.usable_clips_file}", "blue")
        
        with open(self.usable_clips_file, 'r') as f:
            self.usable_clips = json.load(f)
        
        total_events = sum(clip["total_events"] for clip in self.usable_clips)
        cprint(f"✅ Found {len(self.usable_clips)} clips with {total_events} usable events", "green")
    
    def load_event_metadata(self, hash_id: str, event_name: str) -> Tuple[Dict, List[Dict]]:
        """Load event metadata and motion data"""
        event_dir = os.path.join(self.dangerous_clips_dir, hash_id, event_name)
        
        # Load event metadata
        metadata_file = os.path.join(event_dir, "event_metadata.json")
        with open(metadata_file, 'r') as f:
            event_metadata = json.load(f)
        
        # Load motion data
        motion_file = os.path.join(event_dir, "motion_data.json")
        with open(motion_file, 'r') as f:
            motion_data = json.load(f)
        
        return event_metadata, motion_data
    
    def load_lidar_data(self, session_path: str, event_metadata: Dict) -> Tuple[List[str], List[int], pd.DataFrame]:
        """Load LiDAR files and ego trajectory for the event timeframe"""
        # Get event timing
        start_time_s = event_metadata["event_info"]["start_time_seconds"]
        duration_s = event_metadata["event_info"]["duration_seconds"]
        end_time_s = start_time_s + duration_s
        
        # Load ego trajectory
        ego_pose_file = os.path.join(session_path, "city_SE3_egovehicle.feather")
        ego_df = pd.read_feather(ego_pose_file)
        
        # Calculate event timestamp range
        session_start_ns = ego_df['timestamp_ns'].iloc[0]
        event_start_ns = session_start_ns + start_time_s * 1e9
        event_end_ns = session_start_ns + end_time_s * 1e9
        
        # Filter ego trajectory for event timeframe
        event_ego_mask = (ego_df['timestamp_ns'] >= event_start_ns) & (ego_df['timestamp_ns'] <= event_end_ns)
        event_ego_trajectory = ego_df[event_ego_mask].copy()
        
        # Find matching LiDAR files
        lidar_dir = os.path.join(session_path, "sensors", "lidar")
        all_lidar_files = [f for f in os.listdir(lidar_dir) if f.endswith('.feather')]
        
        # Filter LiDAR files for event timeframe
        event_lidar_files = []
        event_lidar_timestamps = []
        
        for lidar_file in all_lidar_files:
            timestamp = int(lidar_file.replace('.feather', ''))
            if event_start_ns <= timestamp <= event_end_ns:
                event_lidar_files.append(os.path.join(lidar_dir, lidar_file))
                event_lidar_timestamps.append(timestamp)
        
        # Sort by timestamp
        sorted_pairs = sorted(zip(event_lidar_timestamps, event_lidar_files))
        event_lidar_timestamps = [pair[0] for pair in sorted_pairs]
        event_lidar_files = [pair[1] for pair in sorted_pairs]
        
        return event_lidar_files, event_lidar_timestamps, event_ego_trajectory
    
    def load_video_data(self, hash_id: str, event_name: str) -> Tuple[str, str]:
        """Get video and keyframes paths"""
        event_dir = os.path.join(self.dangerous_clips_dir, hash_id, event_name)
        
        video_path = os.path.join(event_dir, "video_segment.mp4")
        keyframes_dir = os.path.join(event_dir, "keyframes")
        
        return video_path, keyframes_dir
    
    def extract_representative_frames(self, video_path: str, num_frames: int = 5) -> List[np.ndarray]:
        """Extract representative frames from video"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Select evenly spaced frames
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        cap.release()
        return frames
    
    def analyze_lidar_scene(self, lidar_files: List[str]) -> Dict:
        """Analyze LiDAR point clouds to extract scene information"""
        if not lidar_files:
            return {"error": "No LiDAR files available"}
        
        # Sample a few LiDAR files for analysis
        sample_files = lidar_files[::max(1, len(lidar_files) // 3)][:3]
        
        scene_summary = {
            "total_lidar_frames": len(lidar_files),
            "point_statistics": [],
            "spatial_bounds": None,
            "object_density_zones": []
        }
        
        all_points = []
        
        for lidar_file in sample_files:
            try:
                lidar_df = pd.read_feather(lidar_file)
                points = lidar_df[['x', 'y', 'z']].values
                all_points.append(points)
                
                scene_summary["point_statistics"].append({
                    "file": os.path.basename(lidar_file),
                    "num_points": len(points),
                    "intensity_range": [lidar_df['intensity'].min(), lidar_df['intensity'].max()],
                    "spatial_extent": {
                        "x_range": [points[:, 0].min(), points[:, 0].max()],
                        "y_range": [points[:, 1].min(), points[:, 1].max()],
                        "z_range": [points[:, 2].min(), points[:, 2].max()]
                    }
                })
            except Exception as e:
                scene_summary["point_statistics"].append({
                    "file": os.path.basename(lidar_file),
                    "error": str(e)
                })
        
        # Compute overall spatial bounds
        if all_points:
            combined_points = np.vstack(all_points)
            scene_summary["spatial_bounds"] = {
                "x_range": [combined_points[:, 0].min(), combined_points[:, 0].max()],
                "y_range": [combined_points[:, 1].min(), combined_points[:, 1].max()],
                "z_range": [combined_points[:, 2].min(), combined_points[:, 2].max()],
                "scene_extent_meters": np.sqrt(
                    (combined_points[:, 0].max() - combined_points[:, 0].min())**2 + 
                    (combined_points[:, 1].max() - combined_points[:, 1].min())**2
                )
            }
        
        return scene_summary
    
    def load_dangerous_event(self, hash_id: str, event_name: str, session_path: str) -> DangerousEvent:
        """Load complete multimodal data for a dangerous event"""
        cprint(f"📊 Loading {hash_id}/{event_name}...", "cyan")
        
        start_time = time.time()
        
        # Load metadata and motion data
        event_metadata, motion_data = self.load_event_metadata(hash_id, event_name)
        
        # Load video paths
        video_path, keyframes_dir = self.load_video_data(hash_id, event_name)
        
        # Load LiDAR data
        lidar_files, lidar_timestamps, ego_trajectory = self.load_lidar_data(session_path, event_metadata)
        
        # Create DangerousEvent object
        event = DangerousEvent(
            hash_id=hash_id,
            event_name=event_name,
            session_path=session_path,
            event_metadata=event_metadata,
            motion_data=motion_data,
            video_path=video_path,
            keyframes_dir=keyframes_dir,
            lidar_files=lidar_files,
            lidar_timestamps=lidar_timestamps,
            ego_trajectory=ego_trajectory
        )
        
        load_time = time.time() - start_time
        cprint(f"  ✅ Loaded in {load_time:.2f}s: {len(lidar_files)} LiDAR files, {len(motion_data)} motion points", "green")
        
        return event
    
    def process_event_for_llm(self, event: DangerousEvent, include_frames: bool = True, include_lidar: bool = True) -> Dict:
        """Process event data into LLM-friendly format"""
        cprint(f"🤖 Processing {event.hash_id}/{event.event_name} for LLM...", "yellow")
        
        processed_data = {
            "event_id": f"{event.hash_id}/{event.event_name}",
            "temporal_info": {
                "start_time_seconds": event.event_metadata["event_info"]["start_time_seconds"],
                "duration_seconds": event.event_metadata["event_info"]["duration_seconds"],
                "total_frames": event.event_metadata["event_info"]["end_frame"] - event.event_metadata["event_info"]["start_frame"]
            },
            "risk_assessment": {
                "confidence": event.event_metadata["detection_info"]["confidence"],
                "risk_score": event.event_metadata["detection_info"]["risk_score"],
                "annotation": event.event_metadata["annotation_info"]["comment"]
            },
            "motion_signature": event.event_metadata["motion_signature"],
            "sensor_data_summary": {
                "motion_data_points": len(event.motion_data),
                "lidar_frames": len(event.lidar_files),
                "ego_trajectory_points": len(event.ego_trajectory)
            }
        }
        
        # Add representative frames
        if include_frames and os.path.exists(event.video_path):
            try:
                frames = self.extract_representative_frames(event.video_path)
                processed_data["representative_frames"] = {
                    "count": len(frames),
                    "frame_shape": frames[0].shape if frames else None,
                    "frames_extracted": True
                }
                event.representative_frames = frames
            except Exception as e:
                processed_data["representative_frames"] = {"error": str(e)}
        
        # Add LiDAR scene analysis
        if include_lidar:
            lidar_summary = self.analyze_lidar_scene(event.lidar_files)
            processed_data["lidar_scene_analysis"] = lidar_summary
            event.lidar_summary = lidar_summary
        
        return processed_data
    
    def load_all_events(self, max_events: Optional[int] = None) -> List[DangerousEvent]:
        """Load all usable dangerous events"""
        cprint(f"\n🚀 LOADING ALL MULTIMODAL DANGEROUS EVENTS", "blue", attrs=["bold"])
        
        events = []
        total_loaded = 0
        
        for clip in self.usable_clips:
            if max_events and total_loaded >= max_events:
                break
                
            hash_id = clip["hash_id"]
            session_path = clip["session_path"]
            
            for usable_event in clip["usable_events"]:
                if max_events and total_loaded >= max_events:
                    break
                    
                event_name = usable_event["event_name"]
                
                try:
                    event = self.load_dangerous_event(hash_id, event_name, session_path)
                    events.append(event)
                    total_loaded += 1
                    
                except Exception as e:
                    cprint(f"❌ Failed to load {hash_id}/{event_name}: {e}", "red")
        
        cprint(f"\n✅ Successfully loaded {len(events)} dangerous events", "green", attrs=["bold"])
        return events
    
    def generate_llm_dataset(self, events: List[DangerousEvent], output_file: str = "multimodal_llm_dataset.json") -> Dict:
        """Generate LLM-ready dataset from dangerous events"""
        cprint(f"\n🧠 GENERATING LLM DATASET", "blue", attrs=["bold"])
        
        llm_dataset = {
            "dataset_info": {
                "total_events": len(events),
                "generation_timestamp": time.time(),
                "data_sources": ["video", "lidar", "motion", "metadata"]
            },
            "events": []
        }
        
        for event in events:
            processed_event = self.process_event_for_llm(event)
            llm_dataset["events"].append(processed_event)
        
        # Save dataset
        with open(output_file, 'w') as f:
            json.dump(llm_dataset, f, indent=2, default=str)
        
        cprint(f"💾 LLM dataset saved to {output_file}", "green")
        return llm_dataset

# Example usage and testing
if __name__ == "__main__":
    # Initialize loader
    loader = MultimodalDataLoader()
    
    # Test loading a single event
    cprint("\n🧪 TESTING: Loading single event", "yellow", attrs=["bold"])
    test_clip = loader.usable_clips[0]
    test_event = loader.load_dangerous_event(
        test_clip["hash_id"], 
        test_clip["usable_events"][0]["event_name"],
        test_clip["session_path"]
    )
    
    # Process for LLM
    processed = loader.process_event_for_llm(test_event)
    cprint(f"📋 Sample processed event: {json.dumps(processed, indent=2, default=str)[:500]}...", "white")
    
    # Load first 5 events for testing
    cprint(f"\n🔬 TESTING: Loading first 5 events", "yellow", attrs=["bold"])
    events = loader.load_all_events(max_events=5)
    
    # Generate LLM dataset
    dataset = loader.generate_llm_dataset(events, "test_multimodal_dataset.json")
    
    cprint(f"\n🎉 Test complete! Generated dataset with {len(events)} events", "green", attrs=["bold"])
    cprint(f"📁 Check 'test_multimodal_dataset.json' for results", "blue")