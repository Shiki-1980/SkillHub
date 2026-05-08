"""
Rule-based weak labeling (8 rules).
Used as fallback when LLM labels unavailable.
"""
import numpy as np
from src.features.structural_features import extract as extract_features


def label(df, structural=None):
    """
    Apply 8 rule-based heuristics to label skills.
    Returns (labels, confidence) arrays.
    """
    if structural is None:
        structural = extract_features(df)

    n = len(df)
    labels = np.full(n, "normal", dtype=object)
    confidence = np.ones(n) * 0.5

    # Rule 1: Obfuscation + hidden behavior + dangerous execution → malicious
    m = (structural["sig_obfuscation"] + structural["sig_hidden_behavior"] >= 1) & \
        (structural[["sig_dangerous_exec", "sig_dangerous_net", "sig_data_exfil"]].sum(axis=1) >= 1)
    labels[m] = "malicious"; confidence[m] = 0.85

    # Rule 2: Prompt injection → malicious
    m = structural["sig_prompt_injection"] >= 1
    labels[m] = "malicious"; confidence[m] = 0.90

    # Rule 3: Reverse shell / credential theft → malicious
    m = (structural["sig_reverse_shell"] + structural["sig_credential_theft"] >= 1)
    labels[m] = "malicious"; confidence[m] = 0.80

    # Rule 4: High danger + dangerous signals → unsafe
    m = (structural["n_danger"] >= 2) & \
        (structural[["sig_dangerous_exec", "sig_dangerous_net",
                     "sig_sensitive_paths", "sig_disable_safety"]].sum(axis=1) >= 1)
    m &= (labels == "normal")
    labels[m] = "unsafe"; confidence[m] = 0.75

    # Rule 5: High danger ratio + urgency → unsafe
    m = (structural["danger_ratio"] > 0.6) & (structural["urgency_density"] > 0.01)
    m &= (labels == "normal")
    labels[m] = "unsafe"; confidence[m] = 0.70

    # Rule 6: Security category + multi-danger → unsafe
    m = (structural.get("cat_security", 0) == 1) & (structural["n_danger"] >= 2)
    m &= (labels == "normal")
    labels[m] = "unsafe"; confidence[m] = 0.65

    # Rule 7: Many permissions + short actions → over-privileged (unsafe)
    m = (structural["total_perm_items"] >= 8) & (structural["actions_len"] < 120)
    m &= (labels == "normal")
    labels[m] = "unsafe"; confidence[m] = 0.60

    # Rule 8: Very short → useless
    m = (structural["actions_len"] < 70) & (structural["desc_len"] < 70)
    m &= (labels == "normal")
    labels[m] = "useless"; confidence[m] = 0.75

    return labels, confidence
