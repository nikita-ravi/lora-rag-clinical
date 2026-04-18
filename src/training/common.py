"""Shared QLoRA setup and utilities for LoRA training recipes.

Provides common functions for model setup, data loading, and training.
Supports both Unsloth (GPU) and plain HuggingFace (CPU smoke test) paths.
"""

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

# Conditional imports for Unsloth (only available on CUDA)
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

# Conditional import for peft
from peft import LoraConfig, get_peft_model

# Use standard Trainer since we have pre-tokenized data with label masking
from transformers import Trainer


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Sets seeds in: torch, numpy, random, transformers.
    Also configures CUDNN for deterministic behavior.

    Note: cuBLAS operations may still have non-determinism on GPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str, base_config_path: str = "configs/base.yaml") -> dict:
    """Load and merge base config with recipe-specific config.

    Args:
        config_path: Path to recipe-specific config (e.g., configs/lora_a.yaml)
        base_config_path: Path to base config with shared settings

    Returns:
        Merged config dict with base settings + recipe overrides
    """
    # Load base config
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Load recipe config
    with open(config_path, "r") as f:
        recipe_config = yaml.safe_load(f)

    # Merge: recipe config overrides base config
    merged = {**base_config, **recipe_config}

    # Deep merge nested dicts (lora, training, etc.)
    for key in ["lora", "training", "model", "input", "logging", "paths"]:
        if key in base_config and key in recipe_config:
            merged[key] = {**base_config.get(key, {}), **recipe_config.get(key, {})}
        elif key in base_config:
            merged[key] = base_config[key]
        elif key in recipe_config:
            merged[key] = recipe_config[key]

    return merged


def load_training_data(data_path: str) -> list[dict]:
    """Load training data from JSONL file.

    Args:
        data_path: Path to JSONL file with prompt/target pairs

    Returns:
        List of dicts with 'prompt', 'target', and 'metadata' fields
    """
    examples = []
    with open(data_path, "r") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def setup_model_and_tokenizer(
    model_name: str,
    max_seq_length: int,
    use_unsloth: bool = True,
) -> tuple[Any, Any]:
    """Set up model and tokenizer with optional 4-bit quantization.

    Args:
        model_name: HuggingFace model name or path
        max_seq_length: Maximum sequence length for training
        use_unsloth: If True, use Unsloth for efficient QLoRA (requires CUDA).
                     If False, use plain HuggingFace for CPU smoke testing.

    Returns:
        Tuple of (model, tokenizer)
    """
    if use_unsloth:
        if not UNSLOTH_AVAILABLE:
            raise RuntimeError(
                "Unsloth not available. Install with: pip install unsloth\n"
                "Or use --smoke-test flag for CPU testing without Unsloth."
            )

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=True,
        )
    else:
        # CPU-compatible path for smoke testing
        # Use tiny model without quantization
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # CPU doesn't support bf16
            device_map="cpu",
        )

    return model, tokenizer


def apply_lora(
    model: Any,
    config: dict,
    use_unsloth: bool = True,
) -> Any:
    """Apply LoRA adapters to the model.

    Args:
        model: Base model to apply LoRA to
        config: Full config dict containing 'lora' section
        use_unsloth: If True, use Unsloth's get_peft_model.
                     If False, use standard peft.

    Returns:
        Model with LoRA adapters applied
    """
    lora_config = config["lora"]

    if use_unsloth:
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_config["r"],
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config["lora_dropout"],
            target_modules=lora_config["target_modules"],
            bias=lora_config["bias"],
            use_gradient_checkpointing="unsloth",
            random_state=config.get("seed", 42),
        )
    else:
        # Standard peft for CPU smoke test
        # Note: tiny-gpt2 has different module names than Llama
        # We'll try to apply LoRA to whatever attention modules exist
        peft_config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config["lora_dropout"],
            # For tiny-gpt2, target the attention modules it actually has
            target_modules=["c_attn", "c_proj"],  # GPT-2 attention module names
            bias=lora_config["bias"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    return model


def create_training_args(
    config: dict,
    output_dir: str,
    seed: int,
    smoke_test: bool = False,
    offline: bool = False,
) -> TrainingArguments:
    """Create HuggingFace TrainingArguments from config.

    Args:
        config: Full config dict containing 'training' section
        output_dir: Directory to save checkpoints and final model
        seed: Random seed for training
        smoke_test: If True, override to minimal training (2 steps)
        offline: If True, disable W&B logging

    Returns:
        TrainingArguments instance
    """
    train_config = config["training"]

    if smoke_test:
        # Minimal training for smoke test
        return TrainingArguments(
            output_dir=output_dir,
            max_steps=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            logging_steps=1,
            save_steps=10000,  # Don't save during smoke test
            learning_rate=train_config["learning_rate"],
            seed=seed,
            report_to="none",
            fp16=False,
            bf16=False,
            optim="adamw_torch",  # Standard optimizer for CPU
            remove_unused_columns=False,
        )

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_config["num_epochs"],
        per_device_train_batch_size=train_config["per_device_train_batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        learning_rate=train_config["learning_rate"],
        lr_scheduler_type=train_config["lr_scheduler_type"],
        warmup_ratio=train_config["warmup_ratio"],
        weight_decay=train_config["weight_decay"],
        max_grad_norm=train_config["max_grad_norm"],
        logging_steps=train_config["logging_steps"],
        save_steps=train_config["save_steps"],
        save_total_limit=train_config["save_total_limit"],
        seed=seed,
        bf16=train_config.get("bf16", True),
        fp16=train_config.get("fp16", False),
        optim=train_config.get("optim", "adamw_8bit"),
        report_to="none" if offline else train_config.get("report_to", "wandb"),
        remove_unused_columns=False,
    )


def format_for_trainer(
    examples: list[dict],
    tokenizer: Any,
    max_seq_length: int,
) -> Dataset:
    """Convert examples to HuggingFace Dataset with proper label masking.

    CRITICAL: Labels must mask prompt tokens with -100 so loss is only
    computed on the target (completion). If this is wrong, the model
    learns to reproduce prompts instead of generating completions.

    Args:
        examples: List of dicts with 'prompt' and 'target' fields
        tokenizer: Tokenizer for encoding
        max_seq_length: Maximum sequence length (truncate longer sequences)

    Returns:
        HuggingFace Dataset ready for Trainer
    """
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for example in examples:
        prompt = example["prompt"]
        target = example["target"]
        full_text = prompt + target

        # Tokenize full text
        full_encoding = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None,
        )

        # Tokenize just the prompt to find its length
        prompt_encoding = tokenizer(
            prompt,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = full_encoding["input_ids"]
        attention_mask = full_encoding["attention_mask"]
        prompt_length = len(prompt_encoding["input_ids"])

        # Create labels: -100 for prompt tokens, actual IDs for target tokens
        labels = input_ids.copy()
        labels[:prompt_length] = [-100] * prompt_length

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return Dataset.from_dict({
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    })


def train_recipe(
    data_path: str,
    config_path: str,
    output_dir: str,
    seed: int = 42,
    smoke_test: bool = False,
    offline: bool = False,
) -> dict:
    """Main training function shared by all LoRA recipes.

    Args:
        data_path: Path to training JSONL file (from M5 materialization)
        config_path: Path to recipe YAML config
        output_dir: Directory to save adapter and logs
        seed: Random seed for reproducibility
        smoke_test: If True, use tiny model on CPU for 2 steps
        offline: If True, skip W&B logging

    Returns:
        Dict with training results: steps, final_loss, time_elapsed, output_path
    """
    start_time = time.time()

    # Set seed first
    set_seed(seed)

    # Load config
    config = load_config(config_path)
    recipe_name = config.get("recipe", "unknown")

    # Determine model settings based on smoke_test flag
    if smoke_test:
        model_name = "sshleifer/tiny-gpt2"
        # LoRA-A'/B prompts are 500-900 tokens, need higher limit to avoid truncating targets
        max_seq_length = 1024
        use_unsloth = False
        print(f"[SMOKE TEST] Using {model_name} on CPU")
    else:
        model_name = config["model"]["name"]
        max_seq_length = config["model"]["max_seq_len"]
        use_unsloth = True
        print(f"[TRAINING] Using {model_name} with Unsloth")

    # Load data
    print(f"Loading data from {data_path}...")
    examples = load_training_data(data_path)
    print(f"Loaded {len(examples)} examples")

    # For smoke test, use only 4 examples
    if smoke_test:
        examples = examples[:4]
        print(f"[SMOKE TEST] Using {len(examples)} examples")

    # Setup model and tokenizer
    print(f"Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(
        model_name=model_name,
        max_seq_length=max_seq_length,
        use_unsloth=use_unsloth,
    )

    # Apply LoRA
    print(f"Applying LoRA adapters...")
    model = apply_lora(model, config, use_unsloth=use_unsloth)

    # Prepare dataset
    print(f"Preparing dataset with label masking...")
    dataset = format_for_trainer(examples, tokenizer, max_seq_length)

    # Create training arguments
    training_args = create_training_args(
        config=config,
        output_dir=output_dir,
        seed=seed,
        smoke_test=smoke_test,
        offline=offline,
    )

    # Create data collator for padding
    from transformers import DataCollatorForSeq2Seq
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    # Create trainer (standard Trainer since we have pre-tokenized data)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Train
    print(f"Starting training...")
    train_result = trainer.train()

    # Save adapter
    adapter_dir = Path(output_dir) / f"{recipe_name}_seed_{seed}"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved adapter to {adapter_dir}")

    # Compute results
    elapsed = time.time() - start_time
    total_steps = train_result.global_step
    final_loss = train_result.training_loss

    results = {
        "recipe": recipe_name,
        "seed": seed,
        "total_steps": total_steps,
        "final_loss": final_loss,
        "time_elapsed": elapsed,
        "output_path": str(adapter_dir),
        "smoke_test": smoke_test,
    }

    # Print summary
    print()
    print("=" * 60)
    print(f"TRAINING COMPLETE: {recipe_name}")
    print("=" * 60)
    print(f"  Seed:        {seed}")
    print(f"  Steps:       {total_steps}")
    print(f"  Final loss:  {final_loss:.4f}")
    print(f"  Time:        {elapsed:.1f}s")
    print(f"  Output:      {adapter_dir}")
    print("=" * 60)

    return results
