#!/usr/bin/env python3
"""
llm_warmup.py - LLM Model Testing & JSON Template Validation

Features:
- Clean model loading/unloading with VRAM monitoring
- Token counting and 4K context window management
- Strict JSON template validation
- Colored logging for request/response segregation
- Debug statements for fast error catching
- Support for Mistral-7B and Nous-Hermes fallback

Author: PEM | June 2025
"""

import sys
import os
# Add workspace root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import torch
import gc
from typing import Dict, Any, Optional, Tuple
from termcolor import colored, cprint
import traceback
import time

from utils.model_loader import load_mistral_model, load_nous_hermes_7b_4bit, unload_model


# Simple logger since we don't have common.logger
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Token Management ===
class TokenManager:
    def __init__(self, max_context_tokens: int = 4096):
        self.max_context = max_context_tokens
        self.safety_buffer = 500  # Reserve for response
        self.max_input_tokens = self.max_context - self.safety_buffer
        
    def count_tokens(self, tokenizer, text: str) -> int:
        """Count tokens in text"""
        try:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            count = len(tokens)
            print(f"🔢 Token count: {count}")
            return count
        except Exception as e:
            cprint(f"❌ Token counting error: {e}", "red")
            return len(text.split()) * 1.3  # Rough fallback estimate
            
    def validate_input_length(self, tokenizer, prompt: str) -> bool:
        """Check if prompt fits within context window"""
        token_count = self.count_tokens(tokenizer, prompt)
        
        if token_count > self.max_input_tokens:
            cprint(f"⚠️  Prompt too long: {token_count}/{self.max_input_tokens} tokens", "yellow")
            return False
        else:
            cprint(f"✅ Prompt fits: {token_count}/{self.max_input_tokens} tokens", "green")
            return True

# === JSON Template Validator ===
class JSONTemplateValidator:
    @staticmethod
    def validate_response(response_text: str, expected_template: Dict) -> Tuple[bool, Optional[Dict]]:
        """Validate LLM response against JSON template"""
        try:
            # Try to extract JSON from response
            response_text = response_text.strip()
            
            # Handle common LLM JSON wrapping
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_text = response_text[start:end].strip()
            elif response_text.startswith("{") and response_text.endswith("}"):
                json_text = response_text
            else:
                # Try to find JSON-like content
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start != -1 and end > start:
                    json_text = response_text[start:end]
                else:
                    cprint(f"❌ No JSON found in response", "red")
                    return False, None
            
            print(f"🔍 Extracted JSON: {json_text[:200]}...")
            
            # Parse JSON
            parsed_json = json.loads(json_text)
            cprint(f"✅ Valid JSON parsed", "green")
            
            # Validate structure against template
            missing_keys = []
            for key in expected_template.keys():
                if key not in parsed_json:
                    missing_keys.append(key)
            
            if missing_keys:
                cprint(f"⚠️  Missing keys: {missing_keys}", "yellow")
                return False, parsed_json
            else:
                cprint(f"✅ Template validation passed", "green")
                return True, parsed_json
                
        except json.JSONDecodeError as e:
            cprint(f"❌ JSON parsing error: {e}", "red")
            print(f"🔍 Raw response: {response_text}")
            return False, None
        except Exception as e:
            cprint(f"❌ Validation error: {e}", "red")
            print(f"🔍 Debug traceback: {traceback.format_exc()}")
            return False, None

# === LLM Wrapper ===
class LLMTester:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.token_manager = TokenManager()
        self.validator = JSONTemplateValidator()
        self.current_model_name = None
        
    def load_model(self, model_type: str = "mistral"):
        """Load specified model with error handling"""
        cprint(f"\n🚀 Loading {model_type} model...", "blue", attrs=["bold"])
        
        try:
            if model_type == "mistral":
                self.tokenizer, self.model = load_mistral_model()
                self.current_model_name = "Mistral-7B"
            elif model_type == "nous_hermes":
                self.tokenizer, self.model = load_nous_hermes_7b_4bit()
                self.current_model_name = "Nous-Hermes-7B"
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
            cprint(f"✅ {self.current_model_name} loaded successfully", "green")
            print(f"🔧 Model device: {self.model.device}")
            print(f"🔧 Model dtype: {self.model.dtype}")
            
        except Exception as e:
            cprint(f"❌ Model loading failed: {e}", "red")
            print(f"🔍 Debug traceback: {traceback.format_exc()}")
            raise
            
    def unload_model(self):
        """Clean model unloading"""
        if self.model is not None:
            cprint(f"\n🧹 Unloading {self.current_model_name}...", "yellow")
            try:
                unload_model(self.model)
                self.model = None
                self.tokenizer = None
                self.current_model_name = None
                cprint(f"✅ Model unloaded successfully", "green")
            except Exception as e:
                cprint(f"❌ Unloading error: {e}", "red")
    
    def generate_response(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate response with error handling"""
        cprint(f"\n📝 Generating response...", "cyan")
        
        try:
            # Validate input length
            if not self.token_manager.validate_input_length(self.tokenizer, prompt):
                raise ValueError("Prompt exceeds context window")
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
            print(f"🔢 Input tensor shape: {inputs.shape}")
            
            # Generate
            with torch.no_grad():
                start_time = time.time()
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                generation_time = time.time() - start_time
                
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response[len(self.tokenizer.decode(inputs[0], skip_special_tokens=True)):]
            
            cprint(f"✅ Generation completed in {generation_time:.2f}s", "green")
            print(f"🔢 Generated tokens: {len(outputs[0]) - len(inputs[0])}")
            
            return response.strip()
            
        except Exception as e:
            cprint(f"❌ Generation error: {e}", "red")
            print(f"🔍 Debug traceback: {traceback.format_exc()}")
            raise
    
    def test_json_template(self, prompt: str, json_template: Dict, max_new_tokens: int = 512) -> Tuple[bool, Optional[Dict]]:
        """Test model response against JSON template"""
        cprint(f"\n" + "="*50, "blue")
        cprint(f"🧪 TESTING JSON TEMPLATE", "blue", attrs=["bold"])
        cprint(f"="*50, "blue")
        
        # Print request
        cprint(f"\n📤 REQUEST:", "magenta", attrs=["bold"])
        cprint(f"{prompt}", "white")
        
        # Generate response
        try:
            response = self.generate_response(prompt, max_new_tokens)
            
            # Print response
            cprint(f"\n📥 RESPONSE:", "cyan", attrs=["bold"])
            cprint(f"{response}", "white")
            
            # Validate JSON
            cprint(f"\n🔍 JSON VALIDATION:", "yellow", attrs=["bold"])
            is_valid, parsed_json = self.validator.validate_response(response, json_template)
            
            if is_valid:
                cprint(f"\n✅ TEST PASSED - Valid JSON template", "green", attrs=["bold"])
                print(f"📊 Parsed result: {json.dumps(parsed_json, indent=2)}")
            else:
                cprint(f"\n❌ TEST FAILED - Invalid JSON template", "red", attrs=["bold"])
                
            return is_valid, parsed_json
            
        except Exception as e:
            cprint(f"\n💥 TEST CRASHED: {e}", "red", attrs=["bold"])
            return False, None

# === Test Scenarios ===
def run_warmup_tests():
    """Run comprehensive warmup tests"""
    cprint(f"\n🎯 STARTING LLM WARMUP TESTS", "blue", attrs=["bold", "underline"])
    
    # Initialize tester
    tester = LLMTester()
    
    # Test JSON template for dangerous scenario extraction
    scenario_template = {
        "scenario_type": "string",
        "risk_level": "float",
        "actors": "array",
        "critical_events": "array",
        "environmental_factors": "object"
    }
    
    # Test prompt
    test_prompt = f"""
You are analyzing a dangerous driving scenario. Please extract the key information and respond ONLY in the following JSON format:

{json.dumps(scenario_template, indent=2)}

Scenario Description: A cyclist suddenly cuts into the vehicle's lane from the right sidewalk at an intersection. The ego vehicle was traveling at 25 mph when the cyclist appeared 3 seconds before potential collision. Emergency braking was applied.

Respond with valid JSON only:
"""

    try:
        # Test Mistral first
        cprint(f"\n🔧 Testing Mistral-7B model...", "blue")
        tester.load_model("mistral")
        mistral_success, mistral_result = tester.test_json_template(test_prompt, scenario_template)
        tester.unload_model()
        
        # Test Nous-Hermes as fallback
        cprint(f"\n🔧 Testing Nous-Hermes-7B model...", "blue")
        tester.load_model("nous_hermes")
        hermes_success, hermes_result = tester.test_json_template(test_prompt, scenario_template)
        tester.unload_model()
        
        # Summary
        cprint(f"\n📊 WARMUP SUMMARY:", "blue", attrs=["bold"])
        cprint(f"Mistral-7B: {'✅ PASS' if mistral_success else '❌ FAIL'}", "green" if mistral_success else "red")
        cprint(f"Nous-Hermes-7B: {'✅ PASS' if hermes_success else '❌ FAIL'}", "green" if hermes_success else "red")
        
        if mistral_success:
            cprint(f"\n🎯 Recommended: Use Mistral-7B for production", "green", attrs=["bold"])
        elif hermes_success:
            cprint(f"\n🎯 Recommended: Use Nous-Hermes-7B as fallback", "yellow", attrs=["bold"])
        else:
            cprint(f"\n⚠️  Neither model passed JSON validation", "red", attrs=["bold"])
            
    except Exception as e:
        cprint(f"\n💥 Warmup test crashed: {e}", "red", attrs=["bold"])
        print(f"🔍 Debug traceback: {traceback.format_exc()}")
    finally:
        tester.unload_model()

if __name__ == "__main__":
    run_warmup_tests()