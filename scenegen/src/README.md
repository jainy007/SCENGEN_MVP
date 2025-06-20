# SceneGen Source Modules

This directory contains the core modules for the SceneGen multimodal dangerous scenario processing pipeline. Each module is designed as a clean, functional block with focused responsibility.

## 📁 Module Overview

```
scenegen/src/
├── data_extractor.py      # Triage brain data extraction & semantic preprocessing
├── fusion_analyzer.py     # Spatial context analysis & scene understanding
├── dsl_generator.py       # Mistral-7B DSL generation with caching
└── llm_warmup.py         # LLM testing & JSON validation utilities
```

---

## 📊 data_extractor.py

**Purpose:** Extract and preprocess dangerous event data from triage_brain output for DSL generation.

### Key Functions:
- `load_event_metadata()` - Load event metadata with risk scores and annotations
- `load_motion_data()` - Load detailed motion profiles (velocity, acceleration, jerk)
- `extract_semantic_features()` - Compress to DSL-relevant features
- `load_complete_event_data()` - One-stop function for complete event data

### Usage:

#### Standalone Testing:
```bash
python src/data_extractor.py
```

#### Programmatic Usage:
```python
from data_extractor import TriageBrainDataExtractor

# Initialize extractor
extractor = TriageBrainDataExtractor()

# Load complete event data
event_id = "04akO6mLeIFQRjbq9XwT71QNx0IJ0sTy/dangerous_event_1"
complete_data = extractor.load_complete_event_data(event_id)

# Print event summary
extractor.print_event_summary(event_id)

# Get all available events
all_events = extractor.get_all_usable_event_ids()
```

### Output Structure:
```python
{
    "event_info": {...},              # Original event metadata
    "raw_metadata": {...},            # Raw triage_brain metadata
    "raw_motion_data": [...],          # Raw motion timeline
    "semantic_features": {             # Processed for DSL
        "event_id": "hash/event_name",
        "scenario_type": "narrow near miss",
        "risk_assessment": {...},
        "temporal_context": {...},
        "motion_behavior": {...},
        "motion_timeline": {...}
    }
}
```

### Dependencies:
- `analysis_output/usable_clips_for_multimodal.json` (event list)
- `/home/jainy007/PEM/triage_brain/llm_input_dataset/` (triage brain data)

---

## 🔍 fusion_analyzer.py

**Purpose:** Analyze spatial context and object relationships from BEV analysis and motion data.

### Key Functions:
- `load_bev_analysis()` - Load BEV analysis JSON files
- `extract_spatial_context()` - Get environmental awareness
- `analyze_scene_dynamics()` - Comprehensive spatial analysis
- `compress_fusion_insights()` - Token-efficient compression for DSL

### Usage:

#### Standalone Testing:
```bash
python src/fusion_analyzer.py
```

#### Programmatic Usage:
```python
from fusion_analyzer import BEVFusionAnalyzer

# Initialize analyzer
analyzer = BEVFusionAnalyzer()

# Analyze scene dynamics (requires motion behavior from data_extractor)
motion_behavior = {...}  # From data_extractor semantic features
scenario_type = "narrow near miss"
event_id = "04akO6mLeIFQRjbq9XwT71QNx0IJ0sTy/dangerous_event_1"

scene_dynamics = analyzer.analyze_scene_dynamics(event_id, motion_behavior, scenario_type)

# Compress for DSL generation
fusion_insights = analyzer.compress_fusion_insights(scene_dynamics)
```

### Output Structure:
```python
{
    "scene_type": "urban_street",
    "complexity": "high",
    "key_interactions": ["vehicle_avoidance", "emergency_response"],
    "infrastructure": "narrow_street_with_parking",
    "critical_sequence": ["approach_situation", "emergency_braking", ...],
    "safety_concerns": ["close_vehicle_proximity"]
}
```

### Dependencies:
- `bev_visualizations/analysis/` (BEV analysis files - optional)
- Motion behavior data from `data_extractor.py`

---

## 🤖 dsl_generator.py

**Purpose:** Generate semantic DSL scenarios using Mistral-7B for CARLA scenario creation.

### Key Functions:
- `process_single_event()` - Complete pipeline for one event
- `generate_scenario_dsl()` - Use Mistral to create CARLA-ready DSL
- `save_dsl_scenario()` - Save DSL with caching
- DSL caching system (automatic cache loading/saving)

### Usage:

#### Standalone Testing:
```bash
python src/dsl_generator.py
```

#### Programmatic Usage:
```python
from dsl_generator import DSLGenerator

# Initialize generator (loads existing cache)
generator = DSLGenerator()

# Process single event (uses cache if available)
event_id = "04akO6mLeIFQRjbq9XwT71QNx0IJ0sTy/dangerous_event_1"
success, dsl_path = generator.process_single_event(event_id)

# Force regeneration (ignore cache)
success, dsl_path = generator.process_single_event(event_id, force_regenerate=True)

# Check cache status
if generator.is_dsl_cached(event_id):
    dsl_data = generator.get_cached_dsl(event_id)
    cached_path = generator.get_cached_dsl_path(event_id)

# Print DSL summary
generator.print_dsl_summary(dsl_data)
```

### Arguments:
- `event_id` (str): Event identifier in format "hash_id/event_name"
- `force_regenerate` (bool): If True, ignore cache and regenerate DSL

### Output Structure:
```python
{
    "scenario_id": "narrow_passage_near_miss",
    "scenario_type": "narrow_passage_near_miss",
    "environment": {
        "location_type": "urban",
        "infrastructure": "narrow_street_with_parking",
        "space_constraints": "tight",
        "weather": "clear",
        "time_of_day": "day"
    },
    "ego_vehicle": {
        "initial_speed_ms": 10.5,
        "target_speed_ms": 15.0,
        "behavior_sequence": ["approach_intersection", "detect_hazard", ...],
        "motion_pattern": "low_speed_navigation"
    },
    "scenario_actors": [...],
    "critical_events": {...},
    "success_criteria": {...},
    "carla_specifics": {
        "recommended_map": "Town01",
        "spawn_points": [...],
        "weather_preset": "clear"
    },
    "_metadata": {
        "source_event_id": "hash/event",
        "generation_timestamp": "...",
        "risk_score": 1.9,
        "complexity": "high"
    }
}
```

### Dependencies:
- `data_extractor.py` and `fusion_analyzer.py`
- `llm_warmup.py` (LLM handling utilities)
- `utils.model_loader` (Mistral-7B loading)
- Output: `dsl_scenarios/` directory

### Caching System:
- **Auto-loads** existing DSL files on initialization
- **Cache-first** processing (only regenerates if not cached)
- **Force regeneration** option for development
- **File-based persistence** in `dsl_scenarios/` directory

---

## 🧪 llm_warmup.py

**Purpose:** LLM model testing, JSON template validation, and token management utilities.

### Key Classes:
- `TokenManager` - 4K context window management
- `JSONTemplateValidator` - Strict JSON template validation
- `LLMTester` - Clean model loading/unloading with VRAM monitoring

### Usage:

#### Standalone Testing:
```bash
python src/llm_warmup.py
```

#### Programmatic Usage:
```python
from llm_warmup import LLMTester, TokenManager, JSONTemplateValidator

# Initialize LLM tester
tester = LLMTester()

# Load Mistral model
tester.load_model("mistral")

# Test JSON template generation
json_template = {"field1": "string", "field2": "float"}
prompt = "Generate JSON matching this template..."

success, result = tester.test_json_template(prompt, json_template, max_new_tokens=512)

# Unload model (important for memory management)
tester.unload_model()

# Token counting
token_manager = TokenManager()
token_count = token_manager.count_tokens(tester.tokenizer, prompt)
is_valid = token_manager.validate_input_length(tester.tokenizer, prompt)

# JSON validation
validator = JSONTemplateValidator()
is_valid, parsed = validator.validate_response(response_text, template)
```

### Features:
- **Clean model loading/unloading** with VRAM monitoring
- **Token counting** and 4K context window management
- **Strict JSON template validation**
- **Colored logging** for request/response segregation
- **Support for Mistral-7B** and Nous-Hermes fallback

### Dependencies:
- `utils.model_loader` (model loading utilities)
- Mistral-7B or compatible models

---

## 🔄 Module Integration

### Complete Pipeline Usage:
```python
from data_extractor import TriageBrainDataExtractor
from fusion_analyzer import BEVFusionAnalyzer
from dsl_generator import DSLGenerator

# Initialize all components
extractor = TriageBrainDataExtractor()
analyzer = BEVFusionAnalyzer()
generator = DSLGenerator()

# Process single event through complete pipeline
event_id = "04akO6mLeIFQRjbq9XwT71QNx0IJ0sTy/dangerous_event_1"

# Step 1: Extract semantic features
complete_data = extractor.load_complete_event_data(event_id)
semantic_features = complete_data["semantic_features"]

# Step 2: Analyze spatial context
scene_dynamics = analyzer.analyze_scene_dynamics(
    event_id, 
    semantic_features["motion_behavior"], 
    semantic_features["scenario_type"]
)
fusion_insights = analyzer.compress_fusion_insights(scene_dynamics)

# Step 3: Generate DSL (or use complete pipeline)
success, dsl_path = generator.process_single_event(event_id)
```

### Or use the integrated pipeline:
```python
# Single function call for complete pipeline
generator = DSLGenerator()
success, dsl_path = generator.process_single_event(event_id)
```

---

## 📋 Prerequisites

### Data Requirements:
- `analysis_output/usable_clips_for_multimodal.json` - List of processable events
- `/home/jainy007/PEM/triage_brain/llm_input_dataset/` - Triage brain event data
- `bev_visualizations/analysis/` - BEV analysis files (optional)

### Model Requirements:
- Mistral-7B model (loaded via `utils.model_loader`)
- CUDA-capable GPU (recommended)
- 8GB+ VRAM for model loading

### Python Dependencies:
```bash
pip install torch transformers pandas numpy opencv-python termcolor pathlib glob json
```

---

## 🎯 Output Files

### Generated Artifacts:
- **DSL Scenarios**: `dsl_scenarios/*.json` - CARLA-ready scenario descriptions
- **Cache Files**: Automatic caching in `dsl_scenarios/` for reuse
- **Debug Logs**: Colored terminal output for all operations

### Next Steps:
DSL files are ready for:
1. **OpenAI API processing** → CARLA Python code generation
2. **CARLA simulator execution** → Scenario recreation
3. **Validation testing** → Scenario verification

---

## 🚀 Quick Start

```bash
# Test individual modules
python src/data_extractor.py
python src/fusion_analyzer.py  
python src/dsl_generator.py

# Or use the complete pipeline (coming soon)
python pipeline.py --event_id 04akO6mLeIFQRjbq9XwT71QNx0IJ0sTy/dangerous_event_1
```

Each module is designed to work independently or as part of the integrated pipeline for maximum flexibility and debugging capability.