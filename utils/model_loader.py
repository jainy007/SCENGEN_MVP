import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Tuple
import os
import sys

# Add current directory to path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from gpu_utils import wait_for_vram

# Simple logging since we don't have common.logger
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# === Phi-3.5 Model (Resident) ===
def load_phi_model() -> Tuple:
    PHI_MODEL = "microsoft/phi-3.5-mini-instruct"
    tokenizer = AutoTokenizer.from_pretrained(PHI_MODEL, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        PHI_MODEL,
        device_map="cuda",
        torch_dtype=torch.float16
    ).eval()
    return tokenizer, model

# === Mistral-7B (Ephemeral) ===
def load_mistral_model() -> Tuple:
    MODEL_PATH = "models/mistral-7b/Mistral-7B-Instruct-v0.1"
    logger.info(f"Loading model from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    )
    logger.info("Model loaded successfully.")
    return tokenizer, model.eval()


# === NOUS HERMES Mistral-7B (Flavor) ===
def load_nous_hermes_7b_4bit():
    model_name = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
    print(f"[MODEL LOADER] Loading {model_name} in 4-bit...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant_config
    )

    return tokenizer, model

def unload_model(model):
    import gc
    import torch

    try:
        model.cpu()
    except Exception as e:
        logger.warning(f"Failed to move model to CPU before delete: {e}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    wait_for_vram(required_gb=11.0)