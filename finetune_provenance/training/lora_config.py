"""PEFT LoRA configuration for fine-tuning LLMs."""

import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


def build_lora_model(
    model,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
):
    """Wrap a HuggingFace model with PEFT LoRA adapters.

    Args:
        model: A pretrained HuggingFace CausalLM model.
        r: LoRA rank (lower = fewer parameters).
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout probability on LoRA layers.
        target_modules: Which module names to apply LoRA to.
            Defaults to ['q_proj', 'v_proj'] for attention-only LoRA.
            For GPT-2 use ['c_attn'] or None for auto-detection.
        bias: Whether to train bias parameters.
        task_type: PEFT task type string.

    Returns:
        PEFT-wrapped model.
    """
    try:
        from peft import get_peft_model, LoraConfig, TaskType
    except ImportError as e:
        raise ImportError("peft package required: pip install peft") from e

    peft_task = getattr(TaskType, task_type, TaskType.CAUSAL_LM)

    if target_modules is None:
        # Auto-detect: check model architecture
        module_names = {name for name, _ in model.named_modules()}
        if any("c_attn" in n for n in module_names):
            # GPT-2 style
            target_modules = ["c_attn"]
        elif any("q_proj" in n for n in module_names):
            # LLaMA / Phi-3 style
            target_modules = ["q_proj", "v_proj"]
        else:
            target_modules = None  # Let PEFT pick defaults

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=peft_task,
    )

    peft_model = get_peft_model(model, lora_cfg)
    trainable, total = peft_model.get_nb_trainable_parameters()
    logger.info(
        "LoRA applied: %d trainable params / %d total (%.2f%%)",
        trainable,
        total,
        100 * trainable / max(total, 1),
    )
    return peft_model
