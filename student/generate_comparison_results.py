"""
Runs the SAME freeform teacher prompt through both orchestrators
(ai_automation.py for Hugging Face, openrouter_pipeline.py for OpenRouter),
each deciding for itself which tools to call ("the brain" -- real LLM
tool-selection, not a hardcoded script), and saves the results into two
separate folders for a side-by-side, showable comparison:

    results_huggingface/   <- ai_automation.py's orchestrator + tools
    results_openrouter/    <- openrouter_pipeline.py's orchestrator + tools

Each folder gets a cost_summary.json with what was generated and its real
cost. A combined comparison_summary.json is written at the project root.

HF per-unit prices are fixed/published (no per-call cost in the API
response), so those are computed from the known rates. OpenRouter returns
real billed cost per call in its API response, so those numbers are read
directly from what was actually charged, not estimated.

Run:
    python generate_comparison_results.py
"""

import json
from pathlib import Path

import student.ai_automation as ai_automation
import student.openrouter_pipeline as openrouter_pipeline

TEACHER_PROMPT = (
    "I need a cartoon picture of a cow, a natural realistic photo of a duck, "
    "an audio clip saying 'The sheep says baa', and a short video of a cow "
    "on a farm, for our English animals lesson."
)

HF_PRICES = {"generate_image": 0.025, "generate_audio": 0.0, "generate_video": 0.15, "animate_image": 0.40}


def run_huggingface():
    out_dir = "results_huggingface"
    Path(out_dir).mkdir(exist_ok=True)

    print("=" * 70)
    print("HUGGING FACE ORCHESTRATOR (Qwen3-235B-A22B via novita)")
    print("=" * 70)
    results = ai_automation.run_automation(TEACHER_PROMPT, out_dir=out_dir)

    total_cost = 0.0
    summary = []
    for r in results:
        cost = HF_PRICES.get(r["tool"], 0.0) if r["status"] == "success" else 0.0
        total_cost += cost
        summary.append({**r, "cost_usd": cost})

    with open(Path(out_dir) / "cost_summary.json", "w", encoding="utf-8") as f:
        json.dump({"platform": "huggingface", "teacher_prompt": TEACHER_PROMPT,
                    "total_cost_usd": round(total_cost, 4), "calls": summary}, f, indent=2)

    print(f"\nHF total cost: ${total_cost:.4f}\n")
    return total_cost, summary


def run_openrouter():
    out_dir = "results_openrouter"
    Path(out_dir).mkdir(exist_ok=True)

    print("=" * 70)
    print("OPENROUTER ORCHESTRATOR (Qwen3-235B-A22B via novita, same model)")
    print("=" * 70)
    results = openrouter_pipeline.run_automation(TEACHER_PROMPT, out_dir=out_dir)

    total_cost = 0.0
    summary = []
    for r in results:
        cost = 0.0
        if r["status"] == "success" and "cost: $" in r["result"]:
            cost_str = r["result"].split("cost: $")[1].rstrip(")")
            try:
                cost = float(cost_str)
            except ValueError:
                cost = 0.0
        total_cost += cost
        summary.append({**r, "cost_usd": cost})

    with open(Path(out_dir) / "cost_summary.json", "w", encoding="utf-8") as f:
        json.dump({"platform": "openrouter", "teacher_prompt": TEACHER_PROMPT,
                    "total_cost_usd": round(total_cost, 4), "calls": summary}, f, indent=2)

    print(f"\nOpenRouter total cost: ${total_cost:.4f}\n")
    return total_cost, summary


if __name__ == "__main__":
    hf_cost, hf_summary = run_huggingface()
    or_cost, or_summary = run_openrouter()

    comparison = {
        "teacher_prompt": TEACHER_PROMPT,
        "huggingface": {"total_cost_usd": round(hf_cost, 4), "output_dir": "results_huggingface"},
        "openrouter": {"total_cost_usd": round(or_cost, 4), "output_dir": "results_openrouter"},
    }
    with open("comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("=" * 70)
    print(f"HuggingFace: ${hf_cost:.4f}  |  OpenRouter: ${or_cost:.4f}")
    print("Saved: results_huggingface/, results_openrouter/, comparison_summary.json")
    print("=" * 70)
