"""
Optional LLM narrative layer — turns one case's numbers (agent
probabilities, confidence, SHAP top feature, uncertainty, resolution
protocol) into a short plain-English note.

This is NOT required for the pipeline to work — every other module in this
project (Imaging/Texture/Metadata/Orchestrator/Uncertainty/Explainability)
is classical CNN/ML, not LLM-based, and runs with no API key at all. This
module exists only because it was explicitly requested as an add-on.

No API key is embedded here, and none can be — getting one is something
only you can do:

  Google Gemini (recommended, generous free tier):
    1. https://aistudio.google.com/apikey  -> "Create API key" (needs a
       Google account, no payment method required for the free tier)
    2. set it as an environment variable before running Python:
         Windows (PowerShell):  $env:GOOGLE_API_KEY = "your-key-here"
         Windows (cmd.exe)   :  set GOOGLE_API_KEY=your-key-here

  Groq (alternative, also free-tier, OpenAI-compatible API):
    1. https://console.groq.com/keys -> "Create API Key"
    2. set GROQ_API_KEY the same way

If neither environment variable is set, `generate_case_narrative` silently
falls back to a deterministic, template-based summary — the pipeline never
hard-fails or blocks on a missing key.
"""

import os
import math
import json
from typing import Dict, Optional

import requests


def _valid(x) -> bool:
    """True for a real value; False for None or CSV-round-tripped NaN
    (pandas writes an empty field as NaN, which reads back as a float and
    is truthy — `x is not None` alone doesn't catch it)."""
    if x is None:
        return False
    if isinstance(x, float) and math.isnan(x):
        return False
    return True

GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
              "gemini-2.0-flash:generateContent")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"
TIMEOUT_S = 20


def _prompt_for(case: Dict) -> str:
    lines = [
        "You are summarizing the output of a research breast-thermography "
        "triage pipeline for a clinician. This is a research prototype, "
        "NOT a diagnostic device — never state a diagnosis as fact.",
        "Write 2-4 plain-English sentences summarizing ONLY the numbers "
        "given below. Do not invent clinical facts, do not mention "
        "features not listed, and end by naming the recommended next "
        "action implied by the routing tier.",
        "",
        f"Subject: {case.get('subject_id')}",
        f"Imaging Agent P(patient): {case.get('p_imaging')}",
        f"Texture Agent P(patient): {case.get('p_texture')}",
    ]
    if _valid(case.get("p_metadata_malignancy")):
        lines.append(f"Metadata Agent malignancy risk (patients only): {case.get('p_metadata_malignancy')}")
    lines.append(f"Weighted consensus score (Step 1, C_final): {case.get('c_final')}")
    lines.append(f"Routing tier: {case.get('tier')}")
    if _valid(case.get("bayesian_uncertainty")):
        lines.append(f"Combined epistemic uncertainty: {case.get('bayesian_uncertainty')}")
    if case.get("is_ood") is True:
        lines.append("Flagged as out-of-distribution by Mahalanobis-distance OOD detection.")
    if _valid(case.get("top_shap_feature")):
        lines.append(f"Most influential texture feature (SHAP): {case.get('top_shap_feature')}")
    if isinstance(case.get("resolution_protocol"), list) and case["resolution_protocol"]:
        lines.append("Disagreement resolution protocol steps: " + " / ".join(case["resolution_protocol"]))
    return "\n".join(lines)


def _template_narrative(case: Dict) -> str:
    sid = case.get("subject_id", "this subject")
    tier = case.get("tier", "uncertain")
    c = case.get("c_final")
    c_txt = f"{c:.2f}" if isinstance(c, (int, float)) else "n/a"

    if tier == "confident":
        label = "patient (abnormal)" if case.get("predicted_label") == 1 else "healthy"
        body = (f"{sid}: Imaging and Texture agents agree, weighted consensus "
                f"C={c_txt} (> 0.85) -> Auto-Classified as {label}. "
                f"Grad-CAM++ heatmap available for visual review.")
    elif tier == "uncertain":
        u = case.get("bayesian_uncertainty")
        u_txt = f", combined epistemic uncertainty={u:.3f}" if _valid(u) else ""
        body = (f"{sid}: Partial agent agreement, weighted consensus C={c_txt} "
                f"(0.60-0.85) -> routed to Uncertainty Triage{u_txt}. "
                f"Flagged for radiologist review.")
    else:
        steps = case.get("resolution_protocol")
        steps = steps if isinstance(steps, list) else []
        step_txt = " ".join(steps) if steps else "Escalated for structured disagreement resolution."
        body = f"{sid}: weighted consensus C={c_txt} (< 0.60) -> Disagreement Resolution. {step_txt}"

    if _valid(case.get("p_metadata_malignancy")):
        body += f" Metadata malignancy-risk annotation: {case['p_metadata_malignancy']:.2f}."
    body += " (Research prototype — not a diagnostic device.)"
    return body


def _call_gemini(prompt: str, api_key: str) -> Optional[str]:
    try:
        resp = requests.post(
            f"{GEMINI_URL}?key={api_key}",
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"[llm_narrative] Gemini call failed ({e}) — falling back to template.")
        return None


def _call_groq(prompt: str, api_key: str) -> Optional[str]:
    try:
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": GROQ_MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.2, "max_tokens": 200},
            timeout=TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[llm_narrative] Groq call failed ({e}) — falling back to template.")
        return None


def generate_case_narrative(case: Dict) -> Dict:
    """
    `case` is a dict of the fields on OrchestratorResult (plus optionally
    `top_shap_feature`, `predicted_label`). Returns
    {"text": str, "source": "gemini" | "groq" | "template"}.
    """
    google_key = os.environ.get("GOOGLE_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")
    prompt = _prompt_for(case)

    if google_key:
        text = _call_gemini(prompt, google_key)
        if text:
            return {"text": text, "source": "gemini"}
    if groq_key:
        text = _call_groq(prompt, groq_key)
        if text:
            return {"text": text, "source": "groq"}

    return {"text": _template_narrative(case), "source": "template"}


def generate_batch_narratives(cases) -> "list[Dict]":
    """`cases`: iterable of dicts (e.g. DataFrame.to_dict('records'))."""
    return [dict(subject_id=c.get("subject_id"), **generate_case_narrative(c)) for c in cases]


if __name__ == "__main__":
    demo_case = {
        "subject_id": "DMR-IR::PAC_16", "p_imaging": 0.92, "p_texture": 0.82,
        "p_metadata_malignancy": None, "c_final": 0.78, "tier": "uncertain",
        "bayesian_uncertainty": 0.041, "is_ood": False, "predicted_label": None,
        "resolution_protocol": None,
    }
    result = generate_case_narrative(demo_case)
    print(f"[{result['source']}] {result['text']}")
    if result["source"] == "template":
        print("\nNo GOOGLE_API_KEY or GROQ_API_KEY found in the environment — "
              "see this file's docstring for how to get a free one.")
