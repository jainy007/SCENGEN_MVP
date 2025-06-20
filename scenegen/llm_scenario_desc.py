#!/usr/bin/env python3
"""
llm_scenario_generator.py - Systematic LLM Scenario Generation

Processes dangerous events through Mistral 7B to generate semantic scenario DSL:
- Supports single random event (--random) or sequential processing of all 32
- Clean model reloading between events to prevent hallucinations
- Structured prompting for consistent DSL output
- Progress tracking and error handling

Usage:
    python llm_scenario_generator.py --random          # Process one random event
    python llm_scenario_generator.py                   # Process all 32 events

Author: PEM | June 2025
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import time
from termcolor import colored, cprint
import traceback

# Add workspace root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multimodal_data_loader import MultimodalDataLoader
from utils.model_loader import load_mistral_model, unload_model

class LLMScenarioGenerator:
    """Generates semantic scenario DSL from multimodal dangerous event data"""
    
    def __init__(self, usable_clips_file: str = "analysis_output/usable_clips_for_multimodal.json"):
        self.usable_clips_file = usable_clips_file
        self.output_dir = "scenario_generation_output"
        self.model = None
        self.tokenizer = None
        self.setup_output_dir()
        
        # Load multimodal data
        self.data_loader = MultimodalDataLoader(usable_clips_file)
        
        # Get all events for processing
        self.all_events = self.collect_all_events()
        cprint(f"🎯 Found {len(self.all_events)} events ready for processing", "green")
    
    def setup_output_dir(self):
        """Create output directory structure"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "individual_dsls"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "processing_logs"), exist_ok=True)
    
    def collect_all_events(self) -> List[Dict]:
        """Collect all event identifiers for processing"""
        events = []
        
        for clip in self.data_loader.usable_clips:
            hash_id = clip["hash_id"]
            session_path = clip["session_path"]
            
            for usable_event in clip["usable_events"]:
                event_name = usable_event["event_name"]
                lidar_files_count = usable_event["lidar_files_count"]
                
                events.append({
                    "hash_id": hash_id,
                    "event_name": event_name,
                    "session_path": session_path,
                    "lidar_files_count": lidar_files_count,
                    "event_id": f"{hash_id}/{event_name}"
                })
        
        return events
    
    def load_model_fresh(self):
        """Load Mistral model with clean state"""
        cprint("🔄 Loading fresh Mistral 7B model...", "blue")
        
        try:
            # Ensure any existing model is unloaded
            if self.model is not None:
                self.unload_model_clean()
            
            # Load fresh model
            self.tokenizer, self.model = load_mistral_model()
            cprint("✅ Mistral 7B loaded successfully", "green")
            
        except Exception as e:
            cprint(f"❌ Failed to load model: {e}", "red")
            raise
    
    def unload_model_clean(self):
        """Clean model unloading"""
        if self.model is not None:
            cprint("🧹 Unloading model and cleaning memory...", "yellow")
            try:
                unload_model(self.model)
                self.model = None
                self.tokenizer = None
                cprint("✅ Model unloaded successfully", "green")
            except Exception as e:
                cprint(f"⚠️  Model unloading warning: {e}", "yellow")
    
    def create_scenario_prompt(self, event_data: Dict) -> str:
        """Create structured prompt for Mistral 7B with proper instruction format"""
        
        # Extract key information
        event_id = event_data["event_id"]
        temporal_info = event_data["temporal_info"]
        risk_assessment = event_data["risk_assessment"]
        motion_signature = event_data["motion_signature"]
        lidar_analysis = event_data["lidar_scene_analysis"]
        
        # Use Mistral instruction format
        prompt = f"""<s>[INST] You are an AI expert in autonomous vehicle scenario analysis. Analyze the following dangerous driving event data and generate a structured scenario description.

DANGEROUS EVENT DATA:
- Event ID: {event_id}
- Duration: {temporal_info['duration_seconds']:.1f} seconds
- Risk Score: {risk_assessment['risk_score']:.2f} (confidence: {risk_assessment['confidence']:.2f})
- Human Annotation: "{risk_assessment['annotation']}"

MOTION CHARACTERISTICS:
- Emergency Braking Events: {motion_signature['acceleration_profile']['emergency_braking_events']}
- High Jerk Motion: {motion_signature['motion_characteristics']['has_high_jerk']}
- Velocity Range: {motion_signature['velocity_profile']['min']:.1f} - {motion_signature['velocity_profile']['max']:.1f} m/s
- Average Speed: {motion_signature['velocity_profile']['mean']:.1f} m/s

SCENE ENVIRONMENT:
- LiDAR Frames: {lidar_analysis['total_lidar_frames']}
- Scene Extent: {lidar_analysis['spatial_bounds']['scene_extent_meters']:.0f} meters
- Point Cloud Density: ~{lidar_analysis['point_statistics'][0]['num_points'] if lidar_analysis['point_statistics'] else 0} points per frame

Generate a JSON response with this exact structure:

{{
  "scenario_type": "descriptive name for this dangerous scenario",
  "risk_level": 0.85,
  "actors": ["ego_vehicle", "other_actor"],
  "critical_events": ["specific dangerous events"],
  "environmental_factors": {{
    "road_type": "urban/highway/residential",
    "traffic_conditions": "light/moderate/heavy", 
    "weather": "clear/rain/fog",
    "visibility": "good/poor/limited"
  }},
  "motion_patterns": {{
    "ego_vehicle_behavior": "normal_driving/aggressive/defensive",
    "other_actor_behavior": "unpredictable/aggressive/normal",
    "interaction_type": "near_miss/collision_avoidance/cut_in"
  }},
  "scenario_parameters": {{
    "initial_speed_ego": {motion_signature['velocity_profile']['mean']:.1f},
    "event_duration": {temporal_info['duration_seconds']:.1f},
    "critical_distance": 25.0,
    "time_to_collision": 3.0
  }},
  "carla_generation_hints": {{
    "suggested_map": "Town03",
    "spawn_configuration": "intersection",
    "weather_preset": "ClearNoon",
    "traffic_density": "medium"
  }}
}}

Respond only with valid JSON: [/INST]"""

        return prompt
    
    def validate_llm_response(self, response: str) -> tuple[bool, Optional[Dict]]:
        """Validate and parse LLM response"""
        try:
            # Clean response
            response = response.strip()
            
            # Extract JSON from response
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_text = response[start:end].strip()
            elif response.startswith("{") and response.endswith("}"):
                json_text = response
            else:
                # Try to find JSON-like content
                start = response.find("{")
                end = response.rfind("}") + 1
                if start != -1 and end > start:
                    json_text = response[start:end]
                else:
                    return False, None
            
            # Parse JSON
            parsed_json = json.loads(json_text)
            
            # Validate required fields
            required_fields = ["scenario_type", "risk_level", "actors", "critical_events"]
            for field in required_fields:
                if field not in parsed_json:
                    cprint(f"⚠️  Missing required field: {field}", "yellow")
                    return False, parsed_json
            
            return True, parsed_json
            
        except json.JSONDecodeError as e:
            cprint(f"❌ JSON parsing error: {e}", "red")
            return False, None
        except Exception as e:
            cprint(f"❌ Response validation error: {e}", "red")
            return False, None
    
    def generate_scenario_dsl(self, event_identifier: Dict) -> Optional[Dict]:
        """Generate DSL for a single event"""
        event_id = event_identifier["event_id"]
        hash_id = event_identifier["hash_id"]
        event_name = event_identifier["event_name"]
        session_path = event_identifier["session_path"]
        
        cprint(f"\n🎬 Generating DSL for {event_id}", "cyan", attrs=["bold"])
        
        try:
            # Load multimodal data
            cprint("📊 Loading multimodal event data...", "blue")
            event = self.data_loader.load_dangerous_event(hash_id, event_name, session_path)
            processed_data = self.data_loader.process_event_for_llm(event)
            
            # Create prompt
            prompt = self.create_scenario_prompt(processed_data)
            cprint(f"📝 Generated prompt ({len(prompt)} chars)", "blue")
            
            # Generate response
            cprint("🤖 Generating LLM response...", "yellow")
            start_time = time.time()
            
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=1024,
                    min_new_tokens=100,  # Ensure minimum response length
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response[len(self.tokenizer.decode(inputs[0], skip_special_tokens=True)):]
            
            generation_time = time.time() - start_time
            cprint(f"⚡ Generated in {generation_time:.2f}s ({len(outputs[0]) - len(inputs[0])} tokens)", "green")
            
            # Debug: print raw response for troubleshooting
            cprint(f"🔍 Raw response preview: {response[:200]}...", "white")
            
            # Validate response
            cprint("🔍 Validating DSL response...", "blue")
            is_valid, parsed_dsl = self.validate_llm_response(response)
            
            if is_valid:
                cprint("✅ Valid DSL generated successfully", "green")
                
                # Add metadata
                parsed_dsl["_metadata"] = {
                    "event_id": event_id,
                    "generation_timestamp": time.time(),
                    "generation_time_seconds": generation_time,
                    "model_used": "Mistral-7B",
                    "source_data": {
                        "hash_id": hash_id,
                        "event_name": event_name,
                        "duration_seconds": processed_data["temporal_info"]["duration_seconds"],
                        "risk_score": processed_data["risk_assessment"]["risk_score"],
                        "annotation": processed_data["risk_assessment"]["annotation"]
                    }
                }
                
                return parsed_dsl
            else:
                cprint("❌ Invalid DSL response", "red")
                cprint(f"Raw response: {response[:200]}...", "white")
                return None
                
        except Exception as e:
            cprint(f"💥 Error generating DSL: {e}", "red")
            cprint(f"Traceback: {traceback.format_exc()}", "white")
            return None
    
    def save_dsl_result(self, event_id: str, dsl_data: Dict, success: bool):
        """Save DSL generation result"""
        timestamp = int(time.time())
        
        if success:
            # Save successful DSL
            filename = f"{event_id.replace('/', '_')}_{timestamp}.json"
            filepath = os.path.join(self.output_dir, "individual_dsls", filename)
            
            with open(filepath, 'w') as f:
                json.dump(dsl_data, f, indent=2, default=str)
            
            cprint(f"💾 DSL saved: {filename}", "green")
        
        # Log processing result
        log_entry = {
            "event_id": event_id,
            "timestamp": timestamp,
            "success": success,
            "output_file": filename if success else None,
            "error": None if success else "DSL generation failed"
        }
        
        log_file = os.path.join(self.output_dir, "processing_logs", f"processing_log_{timestamp}.json")
        with open(log_file, 'w') as f:
            json.dump(log_entry, f, indent=2)
    
    def process_single_event(self, event_identifier: Dict):
        """Process a single event with clean model loading"""
        event_id = event_identifier["event_id"]
        
        cprint(f"\n{'='*60}", "blue")
        cprint(f"🎯 PROCESSING EVENT: {event_id}", "blue", attrs=["bold"])
        cprint(f"{'='*60}", "blue")
        
        try:
            # Load fresh model
            self.load_model_fresh()
            
            # Generate DSL
            dsl_result = self.generate_scenario_dsl(event_identifier)
            
            # Save result
            if dsl_result:
                self.save_dsl_result(event_id, dsl_result, True)
                cprint(f"🎉 Successfully processed {event_id}", "green", attrs=["bold"])
            else:
                self.save_dsl_result(event_id, {}, False)
                cprint(f"❌ Failed to process {event_id}", "red", attrs=["bold"])
            
        except Exception as e:
            cprint(f"💥 Critical error processing {event_id}: {e}", "red", attrs=["bold"])
            self.save_dsl_result(event_id, {}, False)
        
        finally:
            # Always unload model for clean state
            self.unload_model_clean()
    
    def process_random_event(self):
        """Process a single random event"""
        random_event = random.choice(self.all_events)
        cprint(f"🎲 Randomly selected: {random_event['event_id']}", "magenta", attrs=["bold"])
        self.process_single_event(random_event)
    
    def process_all_events(self):
        """Process all 32 events sequentially"""
        cprint(f"\n🚀 PROCESSING ALL {len(self.all_events)} EVENTS", "blue", attrs=["bold", "underline"])
        
        successful = 0
        failed = 0
        
        for i, event in enumerate(self.all_events, 1):
            cprint(f"\n📊 Progress: {i}/{len(self.all_events)}", "cyan", attrs=["bold"])
            
            try:
                self.process_single_event(event)
                successful += 1
            except Exception as e:
                cprint(f"💥 Critical failure for event {i}: {e}", "red")
                failed += 1
            
            # Brief pause between events
            time.sleep(2)
        
        # Final summary
        cprint(f"\n{'='*60}", "blue")
        cprint(f"📊 FINAL SUMMARY", "blue", attrs=["bold"])
        cprint(f"{'='*60}", "blue")
        cprint(f"✅ Successful: {successful}/{len(self.all_events)}", "green")
        cprint(f"❌ Failed: {failed}/{len(self.all_events)}", "red")
        cprint(f"📁 Results saved in: {self.output_dir}", "blue")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Generate semantic scenario DSL from dangerous events")
    parser.add_argument("--random", action="store_true", help="Process one random event instead of all events")
    args = parser.parse_args()
    
    # Import torch here to avoid loading until needed
    global torch
    import torch
    
    try:
        # Initialize generator
        generator = LLMScenarioGenerator()
        
        if args.random:
            # Process single random event
            generator.process_random_event()
        else:
            # Process all events
            generator.process_all_events()
    
    except KeyboardInterrupt:
        cprint("\n🛑 Process interrupted by user", "yellow")
    except Exception as e:
        cprint(f"\n💥 Fatal error: {e}", "red", attrs=["bold"])
        cprint(f"Traceback: {traceback.format_exc()}", "white")
    finally:
        # Ensure cleanup
        if 'generator' in locals():
            generator.unload_model_clean()

if __name__ == "__main__":
    main()