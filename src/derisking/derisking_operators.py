"""
7 de-risking operators. Inverse of Task C evasion operators.
Based on: SoK Debloating (Alhanahnah 2024), Auto-SPT (2025).
"""
import random, re

SAFE_GUARDS = [
    "[SAFETY] The user must explicitly approve this action before execution.",
    "[CONFIRM] Present confirmation dialog before proceeding with this operation.",
    "[AUDIT] This action will be logged for security review.",
    "[SANDBOX] This operation runs in a restricted environment.",
    "[REVIEW] Output will be displayed for user inspection before any changes take effect.",
    "[CONSENT] Obtain user consent before accessing external resources.",
]

RISK_DECLARATIONS = [
    "⚠️ Note: This operation accesses sensitive resources. Review before proceeding.",
    "Security notice: This skill requires network access. Ensure you trust the target endpoint.",
    "Warning: This skill can modify files. Backup important data before use.",
    "Caution: This skill handles authentication credentials. Use with trusted services only.",
    "Attention: This skill executes shell commands. Verify commands before approval.",
    "Important: This skill accesses the file system. Confirm file paths before execution.",
]

PERMISSION_SCOPE_DOWNGRADE = {
    "需要 linux, macos, windows": "需要 linux or macos",
    "需要 bash, zsh, sh": "需要 bash",
    "需要 curl, wget": "需要 curl (read-only HTTP)",
    "网络访问": "local network access only",
    "文件系统访问": "read-only file access",
}


class DeRiskingOperators:
    def d1_safety_guard_insert(self, text, strength=0.7):
        if random.random() < strength:
            g = random.choice(SAFE_GUARDS)
            sents = re.split(r"(?<=[.!;])\s+", text)
            if len(sents) > 2:
                sents.insert(min(2, len(sents)-1), g)
                return " ".join(sents)
            return g + " " + text
        return text

    def d2_permission_minimization(self, text, strength=0.5):
        if random.random() < strength:
            for old, new in PERMISSION_SCOPE_DOWNGRADE.items():
                if old.lower() in text.lower():
                    text = re.sub(re.escape(old), new, text, count=1, flags=re.IGNORECASE)
                    break
            if random.random() < 0.3:
                text += " [This skill follows the principle of least privilege.]"
        return text

    def d3_risk_transparency(self, text, strength=0.6):
        return random.choice(RISK_DECLARATIONS) + " " + text if random.random() < strength else text

    def d4_action_justification(self, text, strength=0.5):
        if random.random() < strength:
            for kw in ["curl", "wget", "api", "execute", "run", "read", "write", "delete", "download", "upload"]:
                if kw in text.lower():
                    return text.rstrip() + f" [Purpose: legitimate {kw} operation for authorized functionality.]"
        return text

    def d7_remove_dangerous_flags(self, text, strength=0.7):
        if random.random() < strength:
            text = re.sub(r'\[danger\]\s*([^;:\n]+)', r'[warn] \1 [mitigated by user confirmation]', text)
            text = re.sub(r'https?://[^\s]+', '[URL redacted for safety]', text)
        return text

    def d8_command_sanitization(self, text, strength=0.6):
        if random.random() < strength:
            for cmd in ['curl', 'wget', 'eval', 'exec', 'rm -rf', 'sudo', 'nc ', 'netcat', 'bash -c']:
                if cmd in text.lower():
                    text = re.sub(r'(\b' + re.escape(cmd) + r'\b)',
                                  r'\1 [EXECUTED ONLY AFTER USER CONFIRMATION]',
                                  text, count=2, flags=re.IGNORECASE)
        return text

    def d6_audit_trail_addition(self, text, strength=0.5):
        if random.random() < strength:
            phrases = ["All operations are logged for compliance auditing.",
                       "A detailed operation log is maintained for security review.",
                       "Actions are recorded with timestamps for traceability.",
                       "An immutable audit trail is preserved for all executions."]
            return text.rstrip() + " " + random.choice(phrases)
        return text

    def apply_all(self, text):
        text = self.d7_remove_dangerous_flags(text)
        text = self.d1_safety_guard_insert(text)
        text = self.d2_permission_minimization(text)
        text = self.d3_risk_transparency(text)
        text = self.d4_action_justification(text)
        text = self.d8_command_sanitization(text)
        text = self.d6_audit_trail_addition(text)
        return text
