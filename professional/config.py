"""Shared constants for the orchestrator/personalization LLM, used by
ai_automation.py and personalization.py -- kept in one place so the two
don't drift out of sync on model/provider choice."""

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"  # MoE (235B total, 22B active per token)
MODEL_PROVIDER = "novita"  # this model has no chat-capable "auto" route; provider must be pinned
