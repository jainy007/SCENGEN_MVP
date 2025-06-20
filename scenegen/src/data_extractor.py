#!/usr/bin/env python3
"""
data_extractor.py - Triage Brain Data Extraction Module

Clean extraction and preprocessing of dangerous event data from triage_brain output.
Focuses on semantic features needed for DSL generation while filtering out noise.

Key Functions:
- load_event_metadata(): Load event metadata with risk scores and annotations  
- load_motion_data(): Load detailed motion profiles (velocity, acceleration, jerk)
- extract_semantic_features(): Compress to DSL-relevant features
- validate_event_data(): Ensure data quality before DSL generation

Author: PEM | June 2025
"""

import os
import json
import sys
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from termcolor import cprint
import traceback

class TriageBrainDataExtractor:
    """Extract and preprocess dangerous event data from triage_brain output"""
    
    def __init__(self, triage_data_dir: str = "/home/jainy007/PEM/triage_brain/llm_input_dataset"):
        self.triage_data_dir = Path(triage_data_dir)
        self.usable_clips_file = "analysis_output/usable_clips_for_multimodal.json"
        self.usable_events = self._load_usable_events()
        
        cprint(f"📂 Initialized data extractor with {len(self.usable_events)} usable events", "blue")
    
    def _load_usable_events(self) -> Dict[str, Dict]:
        """Load the list of events that have LiDAR coverage"""
        try:
            with open(self.usable_clips_file, 'r') as f:
                clips_data = json.load(f)
            
            # Create lookup dictionary for fast access
            usable_events = {}
            
            for clip in clips_data:
                hash_id = clip["hash_id"]
                for event in clip["usable_events"]:
                    event_name = event["event_name"]
                    event_id = f"{hash_id}/{event_name}"
                    
                    usable_events[event_id] = {
                        "hash_id": hash_id,
                        "event_name": event_name,
                        "session_path": clip["session_path"],
                        "lidar_files_count": event["lidar_files_count"],
                        "perfect_alignment": event["perfect_alignment"]
                    }
            
            return usable_events
            
        except Exception as e:
            cprint(f"❌ Failed to load usable events: {e}", "red")
            return {}
    
    def get_event_path(self, hash_id: str, event_name: str) -> Path:
        """Get the full path to an event directory"""
        return self.triage_data_dir / hash_id / event_name
    
    def validate_event_exists(self, event_id: str) -> bool:
        """Check if event exists and has required files"""
        if event_id not in self.usable_events:
            cprint(f"❌ Event not in usable list: {event_id}", "red")
            return False
        
        event_info = self.usable_events[event_id]
        event_path = self.get_event_path(event_info["hash_id"], event_info["event_name"])
        
        required_files = ["event_metadata.json", "motion_data.json"]
        missing_files = []
        
        for required_file in required_files:
            file_path = event_path / required_file
            if not file_path.exists():
                missing_files.append(required_file)
        
        if missing_files:
            cprint(f"❌ Missing files for {event_id}: {missing_files}", "red")
            return False
        
        return True
    
    def load_event_metadata(self, hash_id: str, event_name: str) -> Optional[Dict]:
        """Load event metadata with error handling"""
        event_path = self.get_event_path(hash_id, event_name)
        metadata_file = event_path / "event_metadata.json"
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            cprint(f"✅ Loaded metadata for {hash_id}/{event_name}", "green")
            return metadata
            
        except FileNotFoundError:
            cprint(f"❌ Metadata file not found: {metadata_file}", "red")
            return None
        except json.JSONDecodeError as e:
            cprint(f"❌ Invalid JSON in metadata file: {e}", "red")
            return None
        except Exception as e:
            cprint(f"❌ Error loading metadata: {e}", "red")
            return None
    
    def load_motion_data(self, hash_id: str, event_name: str) -> Optional[Dict]:
        """Load motion data with error handling"""
        event_path = self.get_event_path(hash_id, event_name)
        motion_file = event_path / "motion_data.json"
        
        try:
            with open(motion_file, 'r') as f:
                motion_data = json.load(f)
            
            cprint(f"✅ Loaded motion data for {hash_id}/{event_name}", "green")
            return motion_data
            
        except FileNotFoundError:
            cprint(f"❌ Motion file not found: {motion_file}", "red")
            return None
        except json.JSONDecodeError as e:
            cprint(f"❌ Invalid JSON in motion file: {e}", "red")
            return None
        except Exception as e:
            cprint(f"❌ Error loading motion data: {e}", "red")
            return None
    
    def extract_semantic_features(self, metadata: Dict, motion_data: Dict, event_id: str) -> Dict:
        """Extract semantic features relevant for DSL generation"""
        
        # Core event information
        event_info = metadata.get("event_info", {})
        detection_info = metadata.get("detection_info", {})
        annotation_info = metadata.get("annotation_info", {})
        motion_signature = metadata.get("motion_signature", {})
        
        # Motion profiles from metadata
        velocity_profile = motion_signature.get("velocity_profile", {})
        acceleration_profile = motion_signature.get("acceleration_profile", {})
        jerk_profile = motion_signature.get("jerk_profile", {})
        motion_characteristics = motion_signature.get("motion_characteristics", {})
        
        # Extract key semantic features
        semantic_features = {
            # Event identification
            "event_id": event_id,
            "scenario_type": annotation_info.get("comment", "unknown_scenario"),
            
            # Risk assessment
            "risk_assessment": {
                "risk_score": detection_info.get("risk_score", 0.0),
                "confidence": detection_info.get("confidence", 0.0),
                "detection_method": detection_info.get("detection_method", "unknown")
            },
            
            # Temporal context
            "temporal_context": {
                "duration_seconds": event_info.get("duration_seconds", 0.0),
                "start_time_seconds": event_info.get("start_time_seconds", 0.0),
                "end_time_seconds": event_info.get("end_time_seconds", 0.0)
            },
            
            # Motion behavior (critical for CARLA)
            "motion_behavior": {
                "velocity_stats": {
                    "mean_ms": velocity_profile.get("mean", 0.0),
                    "max_ms": velocity_profile.get("max", 0.0),
                    "min_ms": velocity_profile.get("min", 0.0),
                    "range_ms": velocity_profile.get("range", 0.0)
                },
                "acceleration_stats": {
                    "mean_ms2": acceleration_profile.get("mean", 0.0),
                    "emergency_braking_events": acceleration_profile.get("emergency_braking_events", 0)
                },
                "jerk_stats": {
                    "rms": jerk_profile.get("rms", 0.0),
                    "max_magnitude": jerk_profile.get("max_magnitude", 0.0)
                },
                "motion_characteristics": {
                    "has_emergency_braking": motion_characteristics.get("has_emergency_braking", False),
                    "has_high_jerk": motion_characteristics.get("has_high_jerk", False),
                    "smooth_motion": motion_characteristics.get("smooth_motion", True),
                    "velocity_variance_high": motion_characteristics.get("velocity_variance_high", False)
                }
            },
            
            # Detailed motion timeline (from motion_data.json)
            "motion_timeline": self._extract_motion_timeline(motion_data)
        }
        
        return semantic_features
    
    def _extract_motion_timeline(self, motion_data: list) -> Dict:
        """Extract motion timeline for understanding event progression"""
        # motion_data is already a list of motion points
        motion_points = motion_data if isinstance(motion_data, list) else []
        
        if not motion_points:
            return {"error": "no_motion_data"}
        
        # Calculate timeline statistics
        timestamps = [point.get("timestamp_s", 0) for point in motion_points]
        velocities = [point.get("velocity_ms", 0) for point in motion_points]
        accelerations = [point.get("acceleration_ms2", 0) for point in motion_points]
        
        # Find critical moments
        max_velocity_idx = velocities.index(max(velocities)) if velocities else 0
        min_acceleration_idx = accelerations.index(min(accelerations)) if accelerations else 0
        
        timeline_summary = {
            "total_points": len(motion_points),
            "duration_calculated": max(timestamps) - min(timestamps) if timestamps else 0,
            "critical_moments": {
                "max_velocity": {
                    "time_s": timestamps[max_velocity_idx] if timestamps else 0,
                    "velocity_ms": max(velocities) if velocities else 0
                },
                "max_deceleration": {
                    "time_s": timestamps[min_acceleration_idx] if timestamps else 0,
                    "acceleration_ms2": min(accelerations) if accelerations else 0
                }
            },
            "motion_pattern": self._classify_motion_pattern(velocities, accelerations)
        }
        
        return timeline_summary
    
    def _classify_motion_pattern(self, velocities: list, accelerations: list) -> str:
        """Classify the overall motion pattern for semantic understanding"""
        if not velocities or not accelerations:
            return "unknown"
        
        avg_velocity = sum(velocities) / len(velocities)
        min_acceleration = min(accelerations)
        velocity_variance = max(velocities) - min(velocities)
        
        # Simple pattern classification
        if min_acceleration < -10:  # Strong braking
            if avg_velocity > 8:  # High speed
                return "high_speed_emergency_braking"
            else:
                return "moderate_speed_emergency_braking"
        elif velocity_variance > 5:  # High speed variation
            return "variable_speed_maneuvering"
        elif avg_velocity > 10:
            return "high_speed_cruising"
        elif avg_velocity < 3:
            return "low_speed_navigation"
        else:
            return "moderate_speed_driving"
    
    def load_complete_event_data(self, event_id: str) -> Optional[Dict]:
        """Load complete event data (metadata + motion + semantic features)"""
        cprint(f"\n📊 Loading complete data for: {event_id}", "cyan", attrs=["bold"])
        
        # Validate event exists
        if not self.validate_event_exists(event_id):
            return None
        
        # Get event components
        event_info = self.usable_events[event_id]
        hash_id = event_info["hash_id"]
        event_name = event_info["event_name"]
        
        # Load raw data
        metadata = self.load_event_metadata(hash_id, event_name)
        motion_data = self.load_motion_data(hash_id, event_name)
        
        if not metadata or not motion_data:
            cprint(f"❌ Failed to load required data for {event_id}", "red")
            return None
        
        # Extract semantic features
        semantic_features = self.extract_semantic_features(metadata, motion_data, event_id)
        
        # Combine everything
        complete_data = {
            "event_info": event_info,
            "raw_metadata": metadata,
            "raw_motion_data": motion_data,
            "semantic_features": semantic_features
        }
        
        cprint(f"✅ Complete data loaded for {event_id}", "green")
        return complete_data
    
    def get_all_usable_event_ids(self) -> list:
        """Get list of all usable event IDs"""
        return list(self.usable_events.keys())
    
    def print_event_summary(self, event_id: str):
        """Print a summary of event data for debugging"""
        complete_data = self.load_complete_event_data(event_id)
        
        if not complete_data:
            cprint(f"❌ Cannot summarize {event_id} - data loading failed", "red")
            return
        
        semantic = complete_data["semantic_features"]
        
        cprint(f"\n📋 EVENT SUMMARY: {event_id}", "blue", attrs=["bold"])
        cprint(f"  Scenario: {semantic['scenario_type']}", "white")
        cprint(f"  Risk Score: {semantic['risk_assessment']['risk_score']:.2f}", "white")
        cprint(f"  Duration: {semantic['temporal_context']['duration_seconds']:.1f}s", "white")
        cprint(f"  Motion Pattern: {semantic['motion_timeline']['motion_pattern']}", "white")
        cprint(f"  Emergency Braking: {semantic['motion_behavior']['motion_characteristics']['has_emergency_braking']}", "white")
        cprint(f"  High Jerk: {semantic['motion_behavior']['motion_characteristics']['has_high_jerk']}", "white")


def test_data_extractor():
    """Test the data extractor with a sample event"""
    extractor = TriageBrainDataExtractor()
    
    # Test with a known event
    test_event_id = "07YOTznatmYypvQYpzviEcU3yGPsyaGg/dangerous_event_3"
    
    cprint(f"\n🧪 Testing data extractor with: {test_event_id}", "yellow", attrs=["bold"])
    
    # Print summary
    extractor.print_event_summary(test_event_id)
    
    # Show all available events
    all_events = extractor.get_all_usable_event_ids()
    cprint(f"\n📋 Available events: {len(all_events)}", "blue")
    for i, event_id in enumerate(all_events[:5], 1):
        cprint(f"  {i}. {event_id}", "white")
    
    if len(all_events) > 5:
        cprint(f"  ... and {len(all_events) - 5} more", "white")


if __name__ == "__main__":
    test_data_extractor()