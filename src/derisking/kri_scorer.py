"""
KRI Composite Risk Scoring Model.
KRI = w_t*Threat + w_i*Impact + w_e*Exposure - SafetyBonus
Based on: Vuln Prioritization Survey (Jiang 2025), KRI Framework (2025).
"""
import re


class KRIRiskScorer:
    def __init__(self, w_threat=0.55, w_impact=0.25, w_exposure=0.20):
        self.w_threat = w_threat
        self.w_impact = w_impact
        self.w_exposure = w_exposure

        self.threat_patterns = {
            "eval_exec": r"\b(?:eval\s*\(|exec\s*\(|subprocess|os\.system|rm\s+-rf|sudo\s+)",
            "reverse_shell": r"\b(?:reverse\s*shell|bind\s*shell|shell\s*connect)\b",
            "credential_theft": r"\b(?:password|token|secret|credential|api[_\s]?key)\s*(?:steal|leak|exfiltrat|extract|send|transmit)",
            "data_exfil": r"\b(?:exfiltrat|send\s*(?:sensitive|private|secret)\s*(?:data|info|file))",
            "prompt_injection": r"\b(?:prompt\s*inject|jailbreak|ignore\s*(?:previous|above|all)\s*instructions?)",
            "obfuscation": r"\b(?:obfuscat|base64\s*(?:encode|decode)|conceal\s*(?:code|payload))",
            "backdoor": r"\b(?:backdoor|trojan|rootkit|persist\s*(?:access|connection))",
            "danger_network": r"\b(?:curl\s+\S+\s*\|.*sh|wget.*-O.*\|.*sh|nc\s+-[nlvp])",
            "disable_safety": r"\b(?:disable|bypass|skip)\s+(?:security|safety|verification|check|audit)",
            "hidden_behavior": r"\b(?:hidden|silently|secretly|invisible|undetected|covert|stealth)",
        }
        self.impact_patterns = {
            "os_access": r"(?:linux|macos|windows)",
            "shell_access": r"(?:bash|zsh|sh|powershell|shell)",
            "network_access": r"(?:网络|network|api|http|curl|wget)",
            "filesystem_access": r"(?:文件系统|filesystem|file|directory|read|write|delete)",
            "env_vars": r"(?:环境变量|env|environment|\.env|API_KEY|TOKEN|SECRET)",
            "external_service": r"(?:外部服务|external|api\.|cloud|saas)",
        }
        self.exposure_patterns = {
            "url_reference": r"https?://[^\s]+",
            "external_dep": r"(?:依赖.*外部|depends on|requires.*service|needs.*api)",
            "user_data": r"(?:user.*(?:data|input|file|content)|process.*(?:file|data))",
            "multi_step": r"(?:then|after.*do|next|finally|step\s*\d)",
        }
        self.danger_level_re = r"\[danger\]"

    def _count(self, text, patterns):
        if not text: return 0.0
        return min(1.0, sum(len(re.findall(p, str(text).lower())) for p in patterns.values()) / max(len(patterns), 1))

    def score_threat(self, text, risks_str=""):
        ts = self._count(text, self.threat_patterns)
        dn = len(re.findall(self.danger_level_re, str(risks_str)))
        return 0.7 * ts + 0.3 * min(1.0, dn * 0.25)

    def score_impact(self, text, permissions_str=""):
        return 0.5 * self._count(text, self.impact_patterns) + 0.5 * self._count(str(permissions_str), self.impact_patterns)

    def score_exposure(self, text):
        return self._count(text, self.exposure_patterns)

    def _safety_bonus(self, text):
        signals = {
            "user_confirm": r"\[SAFETY\]|\[CONFIRM\]|\[CONSENT\]|user must explicitly|confirmation dialog",
            "audit_log": r"\[AUDIT\]|logged for security|audit trail|operation.*log|immutable.*audit",
            "transparency": r"⚠️ Note|Security notice|Warning:|Caution:|Attention:",
            "least_privilege": r"least privilege|minimum.*necessary|read-only|restricted environment",
            "justification": r"\[Purpose:|legitimate.*operation|authorized functionality",
            "sandbox": r"\[SANDBOX\]|restricted environment|sandboxed",
        }
        return min(0.15, sum(0.03 for p in signals.values() if re.search(p, str(text), re.IGNORECASE)))

    def compute(self, text, risks_str="", permissions_str=""):
        t = self.score_threat(text, risks_str)
        i = self.score_impact(text, permissions_str)
        e = self.score_exposure(text)
        s = self._safety_bonus(text)
        kri = max(0, self.w_threat * t + self.w_impact * i + self.w_exposure * e - s)
        level = "critical" if kri > 0.7 else "high" if kri > 0.5 else "medium" if kri > 0.3 else "low" if kri > 0.1 else "minimal"
        return {"kri": round(kri, 4), "threat": round(t, 4), "impact": round(i, 4),
                "exposure": round(e, 4), "safety_bonus": round(s, 4), "risk_level": level}
