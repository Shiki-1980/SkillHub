"""
Structural feature extraction (45 features).
Based on SkillSieve Layer 1: regex-based security signals, risk metadata, permission analysis.
"""
import re, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore", message=".*has match groups.*")


def parse_risks(risk_str):
    if pd.isna(risk_str):
        return {"danger": 0, "warn": 0, "medium": 0, "safe": 0}
    s = str(risk_str)
    return {
        "danger": len(re.findall(r"\[danger\]", s)),
        "warn": len(re.findall(r"\[warn\]", s)),
        "medium": len(re.findall(r"\[medium\]", s)),
        "safe": len(re.findall(r"\[safe\]", s)),
    }


def extract(df):
    """Extract 45 structural features from skill DataFrame."""
    feats = pd.DataFrame(index=df.index)

    # Risk metadata
    risk_parsed = df["risks"].apply(parse_risks)
    feats["n_danger"] = risk_parsed.apply(lambda x: x["danger"])
    feats["n_warn"] = risk_parsed.apply(lambda x: x["warn"])
    feats["n_medium"] = risk_parsed.apply(lambda x: x["medium"])
    feats["n_safe"] = risk_parsed.apply(lambda x: x["safe"])
    feats["danger_ratio"] = feats["n_danger"] / (feats["n_danger"] + feats["n_safe"] + 1e-6)
    feats["total_risks"] = feats["n_danger"] + feats["n_warn"] + feats["n_medium"] + feats["n_safe"]

    # Text lengths
    feats["name_len"] = df["name"].fillna("").apply(len)
    feats["desc_len"] = df["description"].fillna("").apply(len)
    feats["actions_len"] = df["actions"].fillna("").apply(len)
    feats["perm_len"] = df["permissions"].fillna("").apply(len)
    feats["desc_ratio"] = feats["desc_len"] / (feats["actions_len"] + 1e-6)

    # Permission counts
    perm_str = df["permissions"].fillna("").str.lower()
    feats["perm_os_count"] = perm_str.str.count(r"\[warn\].*?(?:linux|macos|windows)")
    feats["perm_shell_count"] = perm_str.str.count(r"shell")
    feats["perm_network_count"] = perm_str.str.count(r"网络|network")
    feats["perm_file_count"] = perm_str.str.count(r"文件系统|filesystem|文件")
    feats["total_perm_items"] = perm_str.str.count(r"\[warn\]|\[ok\]|\[danger\]")

    # Security signal patterns
    text_all = (df["name"].fillna("") + " " + df["description"].fillna("") + " "
                + df["actions"].fillna("")).str.lower()

    patterns = {
        "sig_reverse_shell": r"\b(reverse\s*shell|bind\s*shell)\b",
        "sig_credential_theft": r"\b(?:password|token|secret|credential|api[_\s]?key)\s*(?:steal|leak|exfiltrat|extract|send|transmit)",
        "sig_data_exfil": r"\b(?:exfiltrat|send\s*(?:sensitive|private|secret)\s*(?:data|info|file))",
        "sig_obfuscation": r"\b(obfuscat|base64\s*(encode|decode)|conceal\s*(code|payload|script))",
        "sig_prompt_injection": r"\b(prompt\s*inject|jailbreak|ignore\s*(previous|above|all)\s*instructions?)\b",
        "sig_dangerous_exec": r"\b(eval\s*\(|exec\s*\(|subprocess|os\.system|rm\s+-rf|sudo\s+)",
        "sig_dangerous_net": r"\b(curl\s+|wget\s+|nc\s+\w|netcat\s+)",
        "sig_sensitive_paths": r"(~?/\.env|~?/\.ssh|~?/\.aws|/etc/passwd|/etc/shadow|keychain|keyring)",
        "sig_urgency": r"\b(immediately|do\s+not\s+tell|must\s+not\s+(tell|reveal|show|disclose))",
        "sig_disable_safety": r"\b(disable|bypass|skip)\s+(security|safety|verification|check|review|audit)",
        "sig_hidden_behavior": r"\b(hidden|silently|secretly|invisible|undetected|covert)\b",
    }
    for name, pattern in patterns.items():
        feats[name] = text_all.str.contains(pattern, na=False, regex=True).astype(int)

    # Urgency density
    urgency_words = ["must", "immediately", "never", "always", "critical", "urgent", "mandatory"]
    feats["urgency_density"] = text_all.apply(
        lambda x: sum(x.count(w) for w in urgency_words)
    ) / (feats["actions_len"] + 1e-6)

    # Metadata
    feats["stars_log"] = np.log1p(df["stars"].fillna(0))
    feats["score"] = df["score"].fillna(0)
    feats["form_workflow"] = (df["form"] == "workflow").astype(int)
    feats["form_prompt"] = (df["form"] == "prompt").astype(int)
    feats["form_reference"] = (df["form"] == "reference").astype(int)
    feats["form_config"] = (df["form"] == "tool-config").astype(int)

    # Category one-hot (top 10)
    top_cats = df["category"].value_counts().head(10).index
    for cat in top_cats:
        safe_name = "cat_" + re.sub(r"[^a-z0-9]", "_", cat.lower())
        feats[safe_name] = (df["category"] == cat).astype(int)

    feats["non_english"] = (df["language"] != "en").astype(int)
    return feats
