#!/usr/bin/env python3
"""
dsl_generator.py - Mistral DSL Generation Module

Generates semantic DSL scenarios using Mistral-7B for CARLA scenario creation.
Combines motion signatures and spatial awareness into comprehensive scenario descriptions.

Key Functions:
- create_dsl_prompt(): Build optimized prompts from semantic + spatial data
- generate_scenario_dsl(): Use Mistral to create CARLA-ready DSL
- validate_dsl_structure(): Ensure DSL completeness for OpenAI processing
- process_single_event(): Complete pipeline for one event

Author: PEM | June 2025
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import torch
import gc
from typing import Dict, Any, Optional, Tuple
from termcolor import cprint
import traceback
from pathlib import Path

# Import our modules
from data_extractor import TriageBrainDataExtractor
from fusion_analyzer import BEVFusionAnalyzer
from utils.model_loader import load_mistral_model, unload_model

# Import LLM handling from warmup
sys.path.append("scenegen")
from llm_warmup import TokenManager, JSONTemplateValidator, LLMTester

class DSLGenerator:
    """Generate semantic DSL scenarios using Mistral-7B"""
    
    def __init__(self):
        self.data_extractor = TriageBrainDataExtractor()
        self.fusion_analyzer = BEVFusionAnalyzer()
        self.llm_tester = LLMTester()
        self.output_dir = Path("dsl_scenarios")
        self.output_dir.mkdir(exist_ok=True)
        
        # CARLA scenario DSL template
        self.dsl_template = {
            "scenario_id": "string",
            "scenario_type": "string", 
            "environment": {
                "location_type": "string",
                "infrastructure": "string", 
                "space_constraints": "string",
                "weather": "string",
                "time_of_day": "string"
            },
            "ego_vehicle": {
                "initial_speed_ms": "float",
                "target_speed_ms": "float", 
                "behavior_sequence": "array",
                "motion_pattern": "string"
            },
            "scenario_actors": "array",
            "critical_events": {
                "trigger_sequence": "array",
                "interaction_types": "array",
                "safety_outcome": "string",
                "risk_level": "string"
            },
            "success_criteria": {
                "collision_avoidance": "boolean",
                "minimum_safe_distance_m": "float",
                "reaction_time_s": "float",
                "completion_time_s": "float"
            },
            "carla_specifics": {
                "recommended_map": "string",
                "spawn_points": "array",
                "weather_preset": "string"
            }
        }
    
    def create_dsl_prompt(self, semantic_features: Dict, fusion_insights: Dict) -> str:
        """Create optimized prompt for Mistral DSL generation"""
        
        # Extract key data for prompt
        scenario_type = semantic_features.get("scenario_type", "unknown")
        risk_score = semantic_features.get("risk_assessment", {}).get("risk_score", 0)
        duration = semantic_features.get("temporal_context", {}).get("duration_seconds", 0)
        
        motion_behavior = semantic_features.get("motion_behavior", {})
        motion_pattern = semantic_features.get("motion_timeline", {}).get("motion_pattern", "unknown")
        
        # Key motion stats - FIX: Ensure proper units
        avg_speed = motion_behavior.get("velocity_stats", {}).get("mean_ms", 0)
        max_speed = motion_behavior.get("velocity_stats", {}).get("max_ms", 0)
        has_emergency_braking = motion_behavior.get("motion_characteristics", {}).get("has_emergency_braking", False)
        has_high_jerk = motion_behavior.get("motion_characteristics", {}).get("has_high_jerk", False)
        
        # Spatial context
        scene_type = fusion_insights.get("scene_type", "urban_street")
        infrastructure = fusion_insights.get("infrastructure", "standard_road")
        complexity = fusion_insights.get("complexity", "moderate")
        key_interactions = fusion_insights.get("key_interactions", [])
        critical_sequence = fusion_insights.get("critical_sequence", [])
        safety_concerns = fusion_insights.get("safety_concerns", [])
        
        prompt = f"""You are an expert in autonomous vehicle scenario generation for CARLA simulator. Convert the following dangerous driving scenario analysis into a complete, executable CARLA scenario DSL.

SCENARIO ANALYSIS:
- Type: {scenario_type}
- Risk Score: {risk_score}/2.0 (high risk)
- Duration: {duration:.1f} seconds
- Scene: {scene_type} with {infrastructure}
- Complexity: {complexity}

MOTION PROFILE:
- Average Speed: {avg_speed:.1f} m/s (meters per second)
- Maximum Speed: {max_speed:.1f} m/s (meters per second)
- Motion Pattern: {motion_pattern}
- Emergency Braking: {has_emergency_braking}
- High Jerk Events: {has_high_jerk}

SPATIAL CONTEXT:
- Key Interactions: {key_interactions}
- Critical Sequence: {critical_sequence}
- Safety Concerns: {safety_concerns}

Generate a complete CARLA scenario in this JSON format:

{json.dumps(self.dsl_template, indent=2)}

CRITICAL REQUIREMENTS:
1. UNITS: All speeds in m/s (range 0-30), distances in meters (0-200), times in seconds (0-60)
2. REALISTIC VALUES: initial_speed_ms and target_speed_ms must be reasonable (e.g., 5.0-15.0 m/s)
3. NO EXTRA TEXT: Respond with ONLY valid JSON. No explanations, examples, or additional text
4. END IMMEDIATELY: Stop after the closing brace }}

SCENARIO SPECIFICATIONS:
- scenario_type: Use descriptive name like "narrow_passage_near_miss" or "stop_sign_overshoot_with_cyclist"
- behavior_sequence: List ego actions ["approach_intersection", "detect_hazard", "emergency_brake", "evade", "recover"]
- scenario_actors: Include other vehicles, pedestrians, cyclists based on interactions
- carla_specifics: Use real CARLA map names like "Town01", "Town02", "Town03", "Town04", "Town05"
- All coordinates in meters, speeds in m/s (NOT km/h or other units)

JSON ONLY - NO ADDITIONAL TEXT:"""

        return prompt
    
    def generate_scenario_dsl(self, semantic_features: Dict, fusion_insights: Dict, event_id: str) -> Tuple[bool, Optional[Dict]]:
        """Generate DSL from semantic and spatial data using Mistral"""
        
        cprint(f"\n🎯 Generating DSL for: {event_id}", "cyan", attrs=["bold"])
        cprint(f"  Scenario: {semantic_features.get('scenario_type', 'unknown')}", "white")
        cprint(f"  Scene: {fusion_insights.get('scene_type', 'unknown')}", "white")
        
        try:
            # Load Mistral
            cprint("🤖 Loading Mistral-7B...", "blue")
            self.llm_tester.load_model("mistral")
            
            # Create prompt
            prompt = self.create_dsl_prompt(semantic_features, fusion_insights)
            
            # Generate DSL with longer response for complete scenario
            cprint("🎬 Generating CARLA scenario DSL...", "blue")
            success, dsl_result = self.llm_tester.test_json_template(
                prompt, 
                self.dsl_template, 
                max_new_tokens=1536  # Longer for complete scenarios
            )
            
            # Unload model to save memory
            self.llm_tester.unload_model()
            
            if success and dsl_result:
                cprint(f"✅ DSL generation successful!", "green")
                
                # Add metadata for tracking
                dsl_result["_metadata"] = {
                    "source_event_id": event_id,
                    "generation_timestamp": pd.Timestamp.now().isoformat() if 'pd' in globals() else "unknown",
                    "risk_score": semantic_features.get("risk_assessment", {}).get("risk_score", 0),
                    "complexity": fusion_insights.get("complexity", "unknown")
                }
                
                return True, dsl_result
            else:
                cprint(f"❌ DSL validation failed", "red")
                return False, None
            
        except Exception as e:
            cprint(f"❌ DSL generation failed: {e}", "red")
            self.llm_tester.unload_model()
            return False, None
    
    def save_dsl_scenario(self, dsl_result: Dict, event_id: str) -> str:
        """Save DSL scenario to file"""
        safe_event_id = event_id.replace("/", "_")
        filename = f"{safe_event_id}_carla_scenario.json"
        file_path = self.output_dir / filename
        
        try:
            with open(file_path, 'w') as f:
                json.dump(dsl_result, f, indent=2)
            
            cprint(f"💾 DSL saved: {filename}", "green")
            return str(file_path)
            
        except Exception as e:
            cprint(f"❌ Error saving DSL: {e}", "red")
            return ""
    
    def process_single_event(self, event_id: str) -> Tuple[bool, Optional[str]]:
        """Complete pipeline: extract data → analyze → generate DSL → save"""
        
        cprint(f"\n🚀 PROCESSING EVENT: {event_id}", "blue", attrs=["bold"])
        
        try:
            # Step 1: Extract semantic features
            cprint("📊 Step 1: Extracting semantic features...", "blue")
            complete_data = self.data_extractor.load_complete_event_data(event_id)
            
            if not complete_data:
                cprint(f"❌ Failed to load event data", "red")
                return False, None
            
            semantic_features = complete_data["semantic_features"]
            
            # Step 2: Analyze spatial context
            cprint("🔍 Step 2: Analyzing spatial context...", "blue")
            motion_behavior = semantic_features["motion_behavior"]
            scenario_type = semantic_features["scenario_type"]
            
            scene_dynamics = self.fusion_analyzer.analyze_scene_dynamics(
                event_id, motion_behavior, scenario_type
            )
            
            fusion_insights = self.fusion_analyzer.compress_fusion_insights(scene_dynamics)
            
            # Step 3: Generate DSL
            cprint("🎯 Step 3: Generating CARLA DSL...", "blue")
            success, dsl_result = self.generate_scenario_dsl(
                semantic_features, fusion_insights, event_id
            )
            
            if not success or not dsl_result:
                cprint(f"❌ DSL generation failed", "red")
                return False, None
            
            # Step 4: Save DSL
            cprint("💾 Step 4: Saving DSL scenario...", "blue")
            saved_path = self.save_dsl_scenario(dsl_result, event_id)
            
            if saved_path:
                cprint(f"🎉 SUCCESS: Complete DSL pipeline for {event_id}", "green", attrs=["bold"])
                return True, saved_path
            else:
                return False, None
                
        except Exception as e:
            cprint(f"💥 Pipeline failed: {e}", "red")
            cprint(f"Debug: {traceback.format_exc()}", "white")
            return False, None
    
    def print_dsl_summary(self, dsl_result: Dict):
        """Print a summary of generated DSL"""
        if not dsl_result:
            return
        
        # Safe extraction with proper handling of different data structures
        environment = dsl_result.get('environment', {})
        ego_vehicle = dsl_result.get('ego_vehicle', {})
        scenario_actors = dsl_result.get('scenario_actors', [])
        critical_events = dsl_result.get('critical_events', {})
        carla_specifics = dsl_result.get('carla_specifics', {})
        
        # Handle critical_events - could be dict or list
        if isinstance(critical_events, list):
            risk_level = critical_events[0].get('risk_level', 'unknown') if critical_events else 'unknown'
        else:
            risk_level = critical_events.get('risk_level', 'unknown')
        
        cprint(f"\n📋 DSL SUMMARY:", "cyan", attrs=["bold"])
        cprint(f"  Scenario: {dsl_result.get('scenario_type', 'unknown')}", "white")
        cprint(f"  Environment: {environment.get('location_type', 'unknown')}", "white")
        cprint(f"  Infrastructure: {environment.get('infrastructure', 'unknown')}", "white")
        cprint(f"  Ego Behavior: {ego_vehicle.get('behavior_sequence', [])}", "white")
        cprint(f"  Actors: {len(scenario_actors)}", "white")
        cprint(f"  Risk Level: {risk_level}", "white")
        cprint(f"  CARLA Map: {carla_specifics.get('recommended_map', 'unknown')}", "white")


def test_dsl_generator():
    """Test DSL generation with the same 2 events used in previous modules"""
    
    generator = DSLGenerator()
    
    # Test events (same as previous modules)
    test_events = [
        "04akO6mLeIFQRjbq9XwT71QNx0IJ0sTy/dangerous_event_1",
        "07YOTznatmYypvQYpzviEcU3yGPsyaGg/dangerous_event_3"
    ]
    
    cprint(f"\n🧪 TESTING DSL GENERATOR", "yellow", attrs=["bold"])
    cprint(f"Processing {len(test_events)} test events...", "white")
    
    successful_dsls = []
    failed_events = []
    
    for i, event_id in enumerate(test_events, 1):
        cprint(f"\n{'='*60}", "blue")
        cprint(f"TEST {i}/{len(test_events)}: {event_id}", "blue", attrs=["bold"])
        cprint(f"{'='*60}", "blue")
        
        try:
            success, saved_path = generator.process_single_event(event_id)
            
            if success:
                successful_dsls.append({
                    'event_id': event_id,
                    'dsl_path': saved_path
                })
                
                # Load and show summary
                if saved_path:
                    with open(saved_path, 'r') as f:
                        dsl_result = json.load(f)
                    generator.print_dsl_summary(dsl_result)
                    
            else:
                failed_events.append(event_id)
                
        except Exception as e:
            cprint(f"💥 Test crashed: {e}", "red")
            failed_events.append(event_id)
    
    # Final summary
    cprint(f"\n{'='*60}", "cyan")
    cprint(f"🏁 DSL GENERATION TEST COMPLETE", "cyan", attrs=["bold"])
    cprint(f"{'='*60}", "cyan")
    
    cprint(f"✅ Successful: {len(successful_dsls)}/{len(test_events)}", "green")
    cprint(f"❌ Failed: {len(failed_events)}", "red")
    
    if successful_dsls:
        cprint(f"\n📁 Generated DSL files:", "green")
        for dsl in successful_dsls:
            cprint(f"  • {dsl['dsl_path']}", "white")
    
    if failed_events:
        cprint(f"\n❌ Failed events:", "red")
        for event_id in failed_events:
            cprint(f"  • {event_id}", "white")
    
    cprint(f"\n💡 DSL files ready for OpenAI → CARLA code generation!", "blue", attrs=["bold"])


if __name__ == "__main__":
    test_dsl_generator()