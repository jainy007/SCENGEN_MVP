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
import time
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
        
        # DSL caching system
        self.dsl_cache = {}
        self._load_existing_dsls()
        
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
    
    def _load_existing_dsls(self):
        """Load existing DSL files into cache"""
        if not self.output_dir.exists():
            return
        
        dsl_files = list(self.output_dir.glob("*_carla_scenario.json"))
        
        for dsl_file in dsl_files:
            try:
                with open(dsl_file, 'r') as f:
                    dsl_data = json.load(f)
                
                # Extract event_id from metadata or filename
                event_id = None
                if "_metadata" in dsl_data and "source_event_id" in dsl_data["_metadata"]:
                    event_id = dsl_data["_metadata"]["source_event_id"]
                else:
                    # Fallback: extract from filename
                    filename = dsl_file.stem
                    if "_carla_scenario" in filename:
                        event_part = filename.replace("_carla_scenario", "")
                        # Convert back to hash/event format
                        if "_dangerous_event_" in event_part:
                            parts = event_part.split("_dangerous_event_")
                            if len(parts) == 2:
                                hash_id = parts[0]
                                event_name = f"dangerous_event_{parts[1]}"
                                event_id = f"{hash_id}/{event_name}"
                
                if event_id:
                    self.dsl_cache[event_id] = {
                        "file_path": str(dsl_file),
                        "dsl_data": dsl_data,
                        "cached_at": dsl_file.stat().st_mtime
                    }
                    cprint(f"📦 Cached DSL: {event_id}", "blue")
                    
            except Exception as e:
                cprint(f"⚠️  Error loading DSL cache from {dsl_file}: {e}", "yellow")
        
        if self.dsl_cache:
            cprint(f"✅ Loaded {len(self.dsl_cache)} existing DSLs into cache", "green")
        else:
            cprint(f"📂 No existing DSL files found - starting fresh", "blue")
    
    def is_dsl_cached(self, event_id: str) -> bool:
        """Check if DSL already exists for this event"""
        return event_id in self.dsl_cache
    
    def get_cached_dsl(self, event_id: str) -> Optional[Dict]:
        """Get cached DSL data for event"""
        if event_id in self.dsl_cache:
            cache_entry = self.dsl_cache[event_id]
            return cache_entry["dsl_data"]
        return None
    
    def get_cached_dsl_path(self, event_id: str) -> Optional[str]:
        """Get cached DSL file path for event"""
        if event_id in self.dsl_cache:
            cache_entry = self.dsl_cache[event_id]
            return cache_entry["file_path"]
        return None
    
    def invalidate_cache(self, event_id: str):
        """Remove event from cache (force regeneration)"""
        if event_id in self.dsl_cache:
            del self.dsl_cache[event_id]
            cprint(f"🗑️  Cache invalidated for: {event_id}", "yellow")
    
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
        """Save DSL scenario to file and update cache"""
        safe_event_id = event_id.replace("/", "_")
        filename = f"{safe_event_id}_carla_scenario.json"
        file_path = self.output_dir / filename
        
        try:
            with open(file_path, 'w') as f:
                json.dump(dsl_result, f, indent=2)
            
            # Update cache
            self.dsl_cache[event_id] = {
                "file_path": str(file_path),
                "dsl_data": dsl_result,
                "cached_at": file_path.stat().st_mtime
            }
            
            cprint(f"💾 DSL saved: {filename}", "green")
            return str(file_path)
            
        except Exception as e:
            cprint(f"❌ Error saving DSL: {e}", "red")
            return ""
    
    def process_single_event(self, event_id: str, force_regenerate: bool = False) -> Tuple[bool, Optional[str]]:
        """Complete pipeline: extract data → analyze → generate DSL → save
        
        Args:
            event_id: Event to process
            force_regenerate: If True, ignore cache and regenerate DSL
        """
        
        cprint(f"\n🚀 PROCESSING EVENT: {event_id}", "blue", attrs=["bold"])
        
        # Check cache first (unless force regenerate)
        if not force_regenerate and self.is_dsl_cached(event_id):
            cached_path = self.get_cached_dsl_path(event_id)
            cprint(f"📦 Using cached DSL: {cached_path}", "green")
            cprint(f"💡 Use force_regenerate=True to rebuild", "blue")
            return True, cached_path
        
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
    
    def process_all_events(self, force_regenerate: bool = False, max_events: Optional[int] = None) -> Dict[str, Any]:
        """Process all usable events to generate DSL scenarios
        
        Args:
            force_regenerate: If True, ignore cache and regenerate all DSLs
            max_events: Limit number of events to process (for testing)
        
        Returns:
            Dictionary with processing statistics
        """
        
        cprint(f"\n🚀 PROCESSING ALL EVENTS", "cyan", attrs=["bold"])
        
        # Get all usable events
        all_event_ids = self.data_extractor.get_all_usable_event_ids()
        
        if max_events:
            all_event_ids = all_event_ids[:max_events]
            cprint(f"📊 Limited to first {max_events} events for testing", "yellow")
        
        total_events = len(all_event_ids)
        cprint(f"📊 Processing {total_events} dangerous events", "white")
        
        # Show cache status
        cached_count = sum(1 for event_id in all_event_ids if self.is_dsl_cached(event_id))
        if not force_regenerate:
            cprint(f"📦 Cache status: {cached_count}/{total_events} already cached", "blue")
            if cached_count > 0:
                cprint(f"💡 Use --force to regenerate cached DSLs", "blue")
        else:
            cprint(f"🔄 Force regeneration: Will rebuild all DSLs", "yellow")
        
        # Processing tracking
        successful_dsls = []
        failed_events = []
        cached_events = []
        start_time = time.time()
        
        for i, event_id in enumerate(all_event_ids, 1):
            cprint(f"\n{'='*80}", "blue")
            cprint(f"PROCESSING {i}/{total_events}: {event_id}", "blue", attrs=["bold"])
            cprint(f"{'='*80}", "blue")
            
            try:
                # Check if cached and not forcing regeneration
                was_cached = False
                if not force_regenerate and self.is_dsl_cached(event_id):
                    was_cached = True
                
                # Process event
                success, saved_path = self.process_single_event(event_id, force_regenerate)
                
                if success and saved_path:
                    event_result = {
                        'event_id': event_id,
                        'dsl_path': saved_path,
                        'was_cached': was_cached,
                        'processing_order': i
                    }
                    
                    if was_cached:
                        cached_events.append(event_result)
                    else:
                        successful_dsls.append(event_result)
                    
                    # Show brief summary for successful events
                    dsl_data = self.get_cached_dsl(event_id)
                    if dsl_data:
                        scenario_type = dsl_data.get('scenario_type', 'unknown')
                        
                        # Safe risk level extraction
                        risk_level = 'unknown'
                        critical_events = dsl_data.get('critical_events', {})
                        
                        try:
                            if isinstance(critical_events, list) and critical_events:
                                risk_level = critical_events[0].get('risk_level', 'unknown')
                            elif isinstance(critical_events, dict):
                                risk_level = critical_events.get('risk_level', 'unknown')
                        except (AttributeError, IndexError, KeyError):
                            risk_level = 'unknown'
                        
                        status = "📦 CACHED" if was_cached else "✅ GENERATED"
                        cprint(f"    {status}: {scenario_type} (risk: {risk_level})", "green")
                else:
                    failed_events.append({
                        'event_id': event_id,
                        'processing_order': i,
                        'error': 'Processing failed'
                    })
                    cprint(f"    ❌ FAILED: Processing unsuccessful", "red")
                
                # Progress update
                remaining = total_events - i
                if remaining > 0 and i % 5 == 0:  # Every 5 events
                    elapsed = time.time() - start_time
                    avg_time = elapsed / i
                    eta_minutes = (avg_time * remaining) / 60
                    cprint(f"    ⏰ Progress: {i}/{total_events} | ETA: {eta_minutes:.1f} minutes", "blue")
                
            except Exception as e:
                failed_events.append({
                    'event_id': event_id,
                    'processing_order': i,
                    'error': str(e)
                })
                cprint(f"    💥 CRASHED: {e}", "red")
        
        # Final statistics
        total_time = time.time() - start_time
        newly_generated = len(successful_dsls)
        total_success = newly_generated + len(cached_events)
        
        cprint(f"\n{'='*80}", "cyan")
        cprint(f"🏁 BATCH PROCESSING COMPLETE", "cyan", attrs=["bold"])
        cprint(f"{'='*80}", "cyan")
        
        cprint(f"⏰ Total processing time: {total_time/60:.1f} minutes", "white")
        cprint(f"✅ Total successful: {total_success}/{total_events}", "green")
        cprint(f"🔄 Newly generated: {newly_generated}", "green")
        cprint(f"📦 Used cached: {len(cached_events)}", "blue")
        cprint(f"❌ Failed: {len(failed_events)}", "red")
        
        # Processing rate statistics
        if newly_generated > 0:
            avg_generation_time = total_time / newly_generated
            cprint(f"⚡ Average generation time: {avg_generation_time:.1f}s per DSL", "white")
            cprint(f"📈 Generation rate: {newly_generated/(total_time/60):.1f} DSLs per minute", "white")
        
        # Scenario type distribution
        scenario_types = {}
        all_successful = successful_dsls + cached_events
        
        for event_result in all_successful:
            dsl_data = self.get_cached_dsl(event_result['event_id'])
            if dsl_data:
                scenario_type = dsl_data.get('scenario_type', 'unknown')
                scenario_types[scenario_type] = scenario_types.get(scenario_type, 0) + 1
        
        if scenario_types:
            cprint(f"\n📊 Scenario type distribution:", "blue")
            for scenario_type, count in sorted(scenario_types.items(), key=lambda x: x[1], reverse=True):
                cprint(f"  {scenario_type}: {count}", "white")
        
        # Risk level distribution
        risk_levels = {'high': 0, 'medium': 0, 'low': 0, 'unknown': 0}
        
        for event_result in all_successful:
            dsl_data = self.get_cached_dsl(event_result['event_id'])
            if dsl_data:
                critical_events = dsl_data.get('critical_events', {})
                risk_level = 'unknown'
                
                try:
                    if isinstance(critical_events, list) and critical_events:
                        risk_level = critical_events[0].get('risk_level', 'unknown')
                    elif isinstance(critical_events, dict):
                        risk_level = critical_events.get('risk_level', 'unknown')
                except (AttributeError, IndexError, KeyError):
                    risk_level = 'unknown'
                
                risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1
        
        cprint(f"\n⚠️  Risk level distribution:", "blue")
        for risk_level, count in risk_levels.items():
            if count > 0:
                cprint(f"  {risk_level}: {count}", "white")
        
        # Show failed events
        if failed_events:
            cprint(f"\n❌ Failed events:", "red")
            for failure in failed_events:
                cprint(f"  {failure['processing_order']:2d}. {failure['event_id']}: {failure['error']}", "white")
        
        # File locations
        cprint(f"\n📁 All DSL files saved in: dsl_scenarios/", "blue")
        if total_success > 0:
            cprint(f"🎯 {total_success} CARLA scenarios ready for OpenAI code generation!", "green", attrs=["bold"])
        
        # Return statistics for programmatic use
        return {
            "total_events": total_events,
            "successful": total_success,
            "newly_generated": newly_generated,
            "cached": len(cached_events),
            "failed": len(failed_events),
            "processing_time_minutes": total_time / 60,
            "scenario_types": scenario_types,
            "risk_levels": risk_levels,
            "failed_events": failed_events,
            "successful_events": [e['event_id'] for e in all_successful]
        }
    
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
    
    cprint(f"\n🧪 TESTING DSL GENERATOR WITH CACHING", "yellow", attrs=["bold"])
    cprint(f"Processing {len(test_events)} test events...", "white")
    
    # Show cache status
    cprint(f"\n📦 Cache Status:", "blue")
    for event_id in test_events:
        if generator.is_dsl_cached(event_id):
            cached_path = generator.get_cached_dsl_path(event_id)
            cprint(f"  ✅ {event_id}: Cached at {cached_path}", "green")
        else:
            cprint(f"  📭 {event_id}: Not cached", "white")
    
    successful_dsls = []
    failed_events = []
    
    for i, event_id in enumerate(test_events, 1):
        cprint(f"\n{'='*60}", "blue")
        cprint(f"TEST {i}/{len(test_events)}: {event_id}", "blue", attrs=["bold"])
        cprint(f"{'='*60}", "blue")
        
        try:
            # Process event (will use cache if available)
            success, saved_path = generator.process_single_event(event_id)
            
            if success:
                successful_dsls.append({
                    'event_id': event_id,
                    'dsl_path': saved_path
                })
                
                # Load and show summary
                if saved_path:
                    # Get DSL data (from cache or newly generated)
                    dsl_result = generator.get_cached_dsl(event_id)
                    if dsl_result:
                        generator.print_dsl_summary(dsl_result)
                    
            else:
                failed_events.append(event_id)
                
        except Exception as e:
            cprint(f"💥 Test crashed: {e}", "red")
            failed_events.append(event_id)
    
    # Test force regeneration for first event
    if test_events:
        cprint(f"\n🔄 TESTING FORCE REGENERATION", "yellow", attrs=["bold"])
        test_event = test_events[0]
        cprint(f"Force regenerating: {test_event}", "white")
        
        try:
            success, saved_path = generator.process_single_event(test_event, force_regenerate=True)
            if success:
                cprint(f"✅ Force regeneration successful", "green")
            else:
                cprint(f"❌ Force regeneration failed", "red")
        except Exception as e:
            cprint(f"💥 Force regeneration crashed: {e}", "red")
    
    # Final summary
    cprint(f"\n{'='*60}", "cyan")
    cprint(f"🏁 DSL GENERATION TEST COMPLETE", "cyan", attrs=["bold"])
    cprint(f"{'='*60}", "cyan")
    
    cprint(f"✅ Successful: {len(successful_dsls)}/{len(test_events)}", "green")
    cprint(f"❌ Failed: {len(failed_events)}", "red")
    
    # Show final cache status
    cprint(f"\n📦 Final Cache Status:", "blue")
    for event_id in test_events:
        if generator.is_dsl_cached(event_id):
            cprint(f"  ✅ {event_id}: Cached", "green")
        else:
            cprint(f"  📭 {event_id}: Not cached", "white")
    
    if successful_dsls:
        cprint(f"\n📁 Generated DSL files:", "green")
        for dsl in successful_dsls:
            cprint(f"  • {dsl['dsl_path']}", "white")
    
    if failed_events:
        cprint(f"\n❌ Failed events:", "red")
        for event_id in failed_events:
            cprint(f"  • {event_id}", "white")
    
    cprint(f"\n💡 DSL files ready for OpenAI → CARLA code generation!", "blue", attrs=["bold"])
    cprint(f"🚀 Cache system ensures no redundant Mistral calls!", "green", attrs=["bold"])


def main():
    """Main entry point with command line interface"""
    import argparse
    import time
    
    parser = argparse.ArgumentParser(
        description="Generate CARLA DSL scenarios from dangerous driving events",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test with sample events
    python dsl_generator.py
    
    # Process specific event
    python dsl_generator.py --event_id 04akO6mLeIFQRjbq9XwT71QNx0IJ0sTy/dangerous_event_1
    
    # Process all 32 events
    python dsl_generator.py --all
    
    # Force regenerate all (ignore cache)
    python dsl_generator.py --all --force
    
    # Process limited number for testing
    python dsl_generator.py --all --max_events 5
        """
    )
    
    parser.add_argument("--event_id", type=str, 
                       help="Process specific event (format: hash_id/event_name)")
    parser.add_argument("--all", action="store_true",
                       help="Process all 32 dangerous events")
    parser.add_argument("--force", action="store_true",
                       help="Force regeneration (ignore cache)")
    parser.add_argument("--max_events", type=int,
                       help="Limit number of events to process (for testing)")
    
    args = parser.parse_args()
    
    try:
        # Initialize DSL generator
        cprint(f"\n🚀 INITIALIZING DSL GENERATOR", "blue", attrs=["bold"])
        generator = DSLGenerator()
        
        if args.event_id:
            # Process specific event
            cprint(f"\n🎯 Processing specific event: {args.event_id}", "cyan")
            
            success, dsl_path = generator.process_single_event(args.event_id, args.force)
            
            if success:
                cprint(f"\n🎉 SUCCESS: DSL generated for {args.event_id}", "green", attrs=["bold"])
                cprint(f"📁 File: {dsl_path}", "blue")
                
                # Show DSL summary
                dsl_data = generator.get_cached_dsl(args.event_id)
                if dsl_data:
                    generator.print_dsl_summary(dsl_data)
            else:
                cprint(f"\n❌ FAILED: Could not generate DSL for {args.event_id}", "red", attrs=["bold"])
                
        elif args.all:
            # Process all events
            cprint(f"\n🌟 Processing ALL dangerous events", "cyan")
            if args.force:
                cprint(f"🔄 Force mode: Will regenerate all cached DSLs", "yellow")
            if args.max_events:
                cprint(f"📊 Limited to {args.max_events} events", "yellow")
            
            stats = generator.process_all_events(
                force_regenerate=args.force,
                max_events=args.max_events
            )
            
            # Show final summary
            success_rate = (stats["successful"] / stats["total_events"]) * 100
            cprint(f"\n🎯 FINAL SUMMARY:", "green", attrs=["bold"])
            cprint(f"Success rate: {success_rate:.1f}% ({stats['successful']}/{stats['total_events']})", "white")
            cprint(f"Processing time: {stats['processing_time_minutes']:.1f} minutes", "white")
            
            if stats["successful"] > 0:
                cprint(f"\n🚀 {stats['successful']} CARLA scenarios ready for OpenAI!", "green", attrs=["bold"])
            
        else:
            # Default: run test with sample events
            cprint(f"\n🧪 Running test with sample events", "cyan")
            cprint(f"💡 Use --all to process all 32 events or --event_id for specific event", "blue")
            
            test_dsl_generator()
            
    except KeyboardInterrupt:
        cprint(f"\n⏹️  Processing interrupted by user", "yellow")
    except Exception as e:
        cprint(f"\n💥 Error: {e}", "red")
        import traceback
        cprint(f"Debug: {traceback.format_exc()}", "white")


if __name__ == "__main__":
    main()