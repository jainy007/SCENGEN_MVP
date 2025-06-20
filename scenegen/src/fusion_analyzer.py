#!/usr/bin/env python3
"""
fusion_analyzer.py - BEV Fusion Data Analysis Module

Extracts spatial context and object relationships from BEV analysis files.
Provides environmental awareness to complement motion signatures for DSL generation.

Key Functions:
- load_bev_analysis(): Load BEV analysis JSON files from previous processing
- extract_spatial_context(): Get object positions, types, and relationships
- analyze_scene_dynamics(): Understand object interactions and movements
- compress_fusion_insights(): Summarize for token-efficient DSL generation

Author: PEM | June 2025
"""

import os
import json
import glob
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from termcolor import cprint
import traceback

class BEVFusionAnalyzer:
    """Analyze BEV fusion data to extract spatial context for DSL generation"""
    
    def __init__(self, bev_analysis_dir: str = "bev_visualizations/analysis"):
        self.bev_analysis_dir = Path(bev_analysis_dir)
        self.analysis_cache = {}
        
        # Load all available analysis files
        self.available_analyses = self._discover_analysis_files()
        cprint(f"📊 Found {len(self.available_analyses)} BEV analysis files", "blue")
    
    def _discover_analysis_files(self) -> Dict[str, str]:
        """Discover all available BEV analysis files"""
        analysis_files = {}
        
        if not self.bev_analysis_dir.exists():
            cprint(f"⚠️  BEV analysis directory not found: {self.bev_analysis_dir}", "yellow")
            return analysis_files
        
        pattern = str(self.bev_analysis_dir / "*_analysis.json")
        found_files = glob.glob(pattern)
        
        for file_path in found_files:
            filename = os.path.basename(file_path)
            # Extract event_id from filename: "hash_event_optimized_bev_analysis.json"
            # Convert back to "hash/event" format
            if "_optimized_bev_analysis.json" in filename:
                event_part = filename.replace("_optimized_bev_analysis.json", "")
                # Find the event name part (dangerous_event_X)
                parts = event_part.split("_")
                if len(parts) >= 3 and "dangerous" in event_part:
                    # Reconstruct: everything before last 3 parts is hash, last 3 are event
                    hash_parts = parts[:-3]
                    event_parts = parts[-3:]
                    
                    hash_id = "_".join(hash_parts)
                    event_name = "_".join(event_parts)
                    event_id = f"{hash_id}/{event_name}"
                    
                    analysis_files[event_id] = file_path
        
        return analysis_files
    
    def load_bev_analysis(self, event_id: str) -> Optional[Dict]:
        """Load BEV analysis for specific event"""
        if event_id in self.analysis_cache:
            return self.analysis_cache[event_id]
        
        if event_id not in self.available_analyses:
            cprint(f"⚠️  No BEV analysis found for: {event_id}", "yellow")
            return None
        
        file_path = self.available_analyses[event_id]
        
        try:
            with open(file_path, 'r') as f:
                analysis = json.load(f)
            
            self.analysis_cache[event_id] = analysis
            cprint(f"✅ Loaded BEV analysis for {event_id}", "green")
            return analysis
            
        except Exception as e:
            cprint(f"❌ Error loading BEV analysis for {event_id}: {e}", "red")
            return None
    
    def extract_spatial_context(self, event_id: str) -> Dict:
        """Extract spatial context from BEV analysis"""
        analysis = self.load_bev_analysis(event_id)
        
        if not analysis:
            return self._create_empty_spatial_context()
        
        # Extract processing summary
        processing_summary = analysis.get("processing_summary", {})
        performance_metrics = analysis.get("performance_metrics", {})
        
        # Basic spatial context
        spatial_context = {
            "event_id": event_id,
            "has_bev_data": True,
            "processing_quality": {
                "frames_processed": processing_summary.get("total_frames_processed", 0),
                "yolo_enabled": processing_summary.get("yolo_enabled", False),
                "multiprocessing_enabled": processing_summary.get("multiprocessing_enabled", False),
                "processing_fps": performance_metrics.get("frames_per_second", 0)
            },
            
            # Placeholder for detailed spatial analysis
            # Note: Current BEV analysis files don't contain detailed object data
            # This structure is prepared for when we enhance the BEV output
            "environmental_context": {
                "scene_type": "urban_residential",  # Inferred from scenarios
                "infrastructure_present": True,
                "traffic_density": "moderate",
                "visibility_conditions": "normal"
            },
            
            "spatial_awareness": {
                "ego_interactions": [],
                "nearby_objects": [],
                "critical_distances": {},
                "spatial_complexity": "moderate"
            }
        }
        
        return spatial_context
    
    def _create_empty_spatial_context(self) -> Dict:
        """Create empty spatial context when no BEV data available"""
        return {
            "has_bev_data": False,
            "processing_quality": {},
            "environmental_context": {
                "scene_type": "unknown",
                "infrastructure_present": False,
                "traffic_density": "unknown",
                "visibility_conditions": "unknown"
            },
            "spatial_awareness": {
                "ego_interactions": [],
                "nearby_objects": [],
                "critical_distances": {},
                "spatial_complexity": "unknown"
            }
        }
    
    def infer_scene_context_from_motion(self, motion_behavior: Dict, scenario_type: str) -> Dict:
        """Infer spatial context from motion data when BEV data is limited"""
        
        # Analyze motion characteristics to infer environment
        motion_chars = motion_behavior.get("motion_characteristics", {})
        velocity_stats = motion_behavior.get("velocity_stats", {})
        
        avg_speed = velocity_stats.get("mean_ms", 0)
        has_emergency_braking = motion_chars.get("has_emergency_braking", False)
        has_high_jerk = motion_chars.get("has_high_jerk", False)
        
        # Infer scene type from scenario and motion
        scene_context = {
            "inferred_scene_type": self._infer_scene_type(scenario_type, avg_speed),
            "inferred_complexity": self._infer_spatial_complexity(
                has_emergency_braking, has_high_jerk, scenario_type
            ),
            "inferred_interactions": self._infer_object_interactions(
                scenario_type, motion_chars
            )
        }
        
        return scene_context
    
    def _infer_scene_type(self, scenario_type: str, avg_speed: float) -> str:
        """Infer scene type from scenario and speed"""
        scenario_lower = scenario_type.lower()
        
        if "intersection" in scenario_lower:
            return "urban_intersection"
        elif "stop sign" in scenario_lower or "sign" in scenario_lower:
            return "residential_intersection"
        elif "narrow" in scenario_lower or "near miss" in scenario_lower:
            if avg_speed < 5:  # Low speed
                return "residential_street"
            else:
                return "urban_street"
        elif "highway" in scenario_lower or avg_speed > 15:
            return "highway"
        elif "parking" in scenario_lower:
            return "parking_area"
        elif "pedestrian" in scenario_lower or "bicycle" in scenario_lower:
            return "mixed_traffic_zone"
        else:
            return "urban_street"  # Default
    
    def _infer_spatial_complexity(self, has_emergency_braking: bool, 
                                 has_high_jerk: bool, scenario_type: str) -> str:
        """Infer spatial complexity from motion characteristics"""
        complexity_score = 0
        
        if has_emergency_braking:
            complexity_score += 2
        if has_high_jerk:
            complexity_score += 2
        
        scenario_lower = scenario_type.lower()
        if any(keyword in scenario_lower for keyword in 
               ["narrow", "near miss", "overshoot", "bicycle", "pedestrian"]):
            complexity_score += 1
        
        if complexity_score >= 4:
            return "high"
        elif complexity_score >= 2:
            return "moderate"
        else:
            return "low"
    
    def _infer_object_interactions(self, scenario_type: str, motion_chars: Dict) -> List[str]:
        """Infer likely object interactions from scenario"""
        interactions = []
        scenario_lower = scenario_type.lower()
        
        if "near miss" in scenario_lower:
            interactions.append("vehicle_avoidance")
        if "bicycle" in scenario_lower or "cyclist" in scenario_lower:
            interactions.append("bicycle_interaction")
        if "pedestrian" in scenario_lower or "man" in scenario_lower:
            interactions.append("pedestrian_interaction")
        if "stop sign" in scenario_lower:
            interactions.append("traffic_control_compliance")
        if "overshoot" in scenario_lower:
            interactions.append("traffic_violation_recovery")
        if motion_chars.get("has_emergency_braking", False):
            interactions.append("emergency_response")
        
        return interactions
    
    def analyze_scene_dynamics(self, event_id: str, motion_behavior: Dict, 
                             scenario_type: str) -> Dict:
        """Comprehensive scene dynamics analysis"""
        
        # Load BEV spatial context
        spatial_context = self.extract_spatial_context(event_id)
        
        # Infer additional context from motion
        inferred_context = self.infer_scene_context_from_motion(motion_behavior, scenario_type)
        
        # Combine for comprehensive scene understanding
        scene_dynamics = {
            "scene_classification": {
                "primary_type": inferred_context["inferred_scene_type"],
                "complexity_level": inferred_context["inferred_complexity"],
                "interaction_types": inferred_context["inferred_interactions"]
            },
            
            "environmental_factors": {
                "infrastructure_type": self._get_infrastructure_type(scenario_type),
                "traffic_control": self._get_traffic_control(scenario_type),
                "space_constraints": self._get_space_constraints(scenario_type, motion_behavior)
            },
            
            "dynamic_elements": {
                "ego_behavior_pattern": motion_behavior.get("motion_timeline", {}).get("motion_pattern", "unknown"),
                "interaction_sequence": self._analyze_interaction_sequence(scenario_type, motion_behavior),
                "critical_moments": self._identify_critical_moments(motion_behavior)
            },
            
            "spatial_relationships": {
                "proximity_factors": self._analyze_proximity_factors(scenario_type),
                "movement_constraints": self._analyze_movement_constraints(motion_behavior),
                "safety_margins": self._analyze_safety_margins(scenario_type, motion_behavior)
            }
        }
        
        return scene_dynamics
    
    def _get_infrastructure_type(self, scenario_type: str) -> str:
        """Determine infrastructure type from scenario"""
        scenario_lower = scenario_type.lower()
        
        if "stop sign" in scenario_lower:
            return "stop_sign_intersection"
        elif "intersection" in scenario_lower:
            return "traffic_light_intersection"
        elif "narrow" in scenario_lower:
            return "narrow_street_with_parking"
        else:
            return "standard_road"
    
    def _get_traffic_control(self, scenario_type: str) -> List[str]:
        """Get traffic control elements present"""
        controls = []
        scenario_lower = scenario_type.lower()
        
        if "stop sign" in scenario_lower:
            controls.append("stop_sign")
        if "intersection" in scenario_lower:
            controls.append("intersection_rules")
        if "overshoot" in scenario_lower:
            controls.append("violation_recovery")
        
        return controls
    
    def _get_space_constraints(self, scenario_type: str, motion_behavior: Dict) -> Dict:
        """Analyze space constraints from scenario and motion"""
        constraints = {"level": "normal", "factors": []}
        
        scenario_lower = scenario_type.lower()
        has_high_jerk = motion_behavior.get("motion_characteristics", {}).get("has_high_jerk", False)
        
        if "narrow" in scenario_lower:
            constraints["level"] = "high"
            constraints["factors"].append("narrow_passage")
        
        if has_high_jerk:
            constraints["factors"].append("precision_maneuvering")
        
        if "parking" in scenario_lower:
            constraints["factors"].append("parked_vehicles")
        
        return constraints
    
    def _analyze_interaction_sequence(self, scenario_type: str, motion_behavior: Dict) -> List[str]:
        """Analyze the sequence of interactions"""
        sequence = []
        
        # Start with approach
        sequence.append("approach_situation")
        
        # Add scenario-specific interactions
        scenario_lower = scenario_type.lower()
        
        if "stop" in scenario_lower:
            sequence.append("decelerate_to_stop")
        
        if motion_behavior.get("motion_characteristics", {}).get("has_emergency_braking", False):
            sequence.append("emergency_braking")
        
        if "near miss" in scenario_lower:
            sequence.append("collision_avoidance")
        
        if "bicycle" in scenario_lower or "pedestrian" in scenario_lower:
            sequence.append("yield_to_vulnerable_user")
        
        sequence.append("resume_normal_operation")
        
        return sequence
    
    def _identify_critical_moments(self, motion_behavior: Dict) -> List[Dict]:
        """Identify critical moments from motion timeline"""
        critical_moments = []
        
        timeline = motion_behavior.get("motion_timeline", {})
        critical_data = timeline.get("critical_moments", {})
        
        if critical_data.get("max_velocity", {}).get("velocity_ms", 0) > 0:
            critical_moments.append({
                "type": "peak_velocity",
                "time_s": critical_data["max_velocity"].get("time_s", 0),
                "value": critical_data["max_velocity"].get("velocity_ms", 0)
            })
        
        if critical_data.get("max_deceleration", {}).get("acceleration_ms2", 0) < -5:
            critical_moments.append({
                "type": "emergency_braking",
                "time_s": critical_data["max_deceleration"].get("time_s", 0),
                "value": critical_data["max_deceleration"].get("acceleration_ms2", 0)
            })
        
        return critical_moments
    
    def _analyze_proximity_factors(self, scenario_type: str) -> List[str]:
        """Analyze proximity factors from scenario"""
        factors = []
        scenario_lower = scenario_type.lower()
        
        if "near miss" in scenario_lower:
            factors.append("close_vehicle_proximity")
        if "narrow" in scenario_lower:
            factors.append("infrastructure_proximity")
        if "bicycle" in scenario_lower or "pedestrian" in scenario_lower:
            factors.append("vulnerable_user_proximity")
        
        return factors
    
    def _analyze_movement_constraints(self, motion_behavior: Dict) -> List[str]:
        """Analyze movement constraints from motion data"""
        constraints = []
        
        motion_chars = motion_behavior.get("motion_characteristics", {})
        
        if motion_chars.get("has_high_jerk", False):
            constraints.append("precision_steering_required")
        
        if motion_chars.get("has_emergency_braking", False):
            constraints.append("sudden_deceleration_required")
        
        if not motion_chars.get("smooth_motion", True):
            constraints.append("erratic_movement_pattern")
        
        return constraints
    
    def _analyze_safety_margins(self, scenario_type: str, motion_behavior: Dict) -> Dict:
        """Analyze safety margins from scenario and motion"""
        margins = {"spatial": "unknown", "temporal": "unknown", "factors": []}
        
        scenario_lower = scenario_type.lower()
        
        if "near miss" in scenario_lower:
            margins["spatial"] = "minimal"
            margins["factors"].append("close_call")
        
        if motion_behavior.get("motion_characteristics", {}).get("has_emergency_braking", False):
            margins["temporal"] = "minimal"
            margins["factors"].append("last_second_response")
        
        return margins
    
    def compress_fusion_insights(self, scene_dynamics: Dict) -> Dict:
        """Compress scene dynamics for token-efficient DSL generation"""
        
        compressed = {
            "scene_type": scene_dynamics["scene_classification"]["primary_type"],
            "complexity": scene_dynamics["scene_classification"]["complexity_level"],
            "key_interactions": scene_dynamics["scene_classification"]["interaction_types"][:3],  # Top 3
            "infrastructure": scene_dynamics["environmental_factors"]["infrastructure_type"],
            "critical_sequence": scene_dynamics["dynamic_elements"]["interaction_sequence"][:4],  # Key steps
            "safety_concerns": scene_dynamics["spatial_relationships"]["proximity_factors"][:2]  # Top 2
        }
        
        return compressed


def test_fusion_analyzer():
    """Test the fusion analyzer with sample events"""
    analyzer = BEVFusionAnalyzer()
    
    # Test events
    test_events = [
        "04akO6mLeIFQRjbq9XwT71QNx0IJ0sTy/dangerous_event_1",
        "07YOTznatmYypvQYpzviEcU3yGPsyaGg/dangerous_event_3"
    ]
    
    for event_id in test_events:
        cprint(f"\n🧪 Testing fusion analysis for: {event_id}", "yellow", attrs=["bold"])
        
        # Mock motion behavior data (normally would come from data_extractor)
        mock_motion_behavior = {
            "motion_characteristics": {
                "has_emergency_braking": True,
                "has_high_jerk": True,
                "smooth_motion": False
            },
            "velocity_stats": {"mean_ms": 6.0},
            "motion_timeline": {
                "motion_pattern": "low_speed_navigation",
                "critical_moments": {
                    "max_velocity": {"time_s": 10, "velocity_ms": 8},
                    "max_deceleration": {"time_s": 15, "acceleration_ms2": -12}
                }
            }
        }
        
        scenario_type = "narrow near miss" if "event_1" in event_id else "stop sign overshoot"
        
        # Analyze scene dynamics
        scene_dynamics = analyzer.analyze_scene_dynamics(event_id, mock_motion_behavior, scenario_type)
        
        # Compress for DSL
        compressed = analyzer.compress_fusion_insights(scene_dynamics)
        
        # Print results
        cprint(f"  Scene Type: {compressed['scene_type']}", "white")
        cprint(f"  Complexity: {compressed['complexity']}", "white")
        cprint(f"  Key Interactions: {compressed['key_interactions']}", "white")
        cprint(f"  Infrastructure: {compressed['infrastructure']}", "white")
        cprint(f"  Critical Sequence: {compressed['critical_sequence']}", "white")
        cprint(f"  Safety Concerns: {compressed['safety_concerns']}", "white")


if __name__ == "__main__":
    test_fusion_analyzer()