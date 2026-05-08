"""
6 perturbation operators for adversarial skill generation.
Based on: AOP-Mal (Xu 2025), GA Text Attack (2020), ICLT (2025).
"""
import random, re

RISK_DOWNGRADE_MAP = {
    "danger": "caution", "warning": "notice", "critical": "important",
    "must not": "should avoid", "immediately": "promptly", "urgent": "timely",
    "hidden": "internal", "silent": "quiet", "bypass": "streamline",
    "disable": "adjust", "inject": "insert", "exploit": "utilize",
    "payload": "content", "backdoor": "alternate access", "exfiltrate": "transfer",
    "steal": "collect", "hack": "access", "attack": "test",
    "malicious": "custom", "obfuscate": "reorganize",
}

BENIGN_CONTEXTS = [
    "This skill follows best practices for security and transparency.",
    "All operations are logged and can be reviewed by the user.",
    "Designed for legitimate development and testing purposes only.",
    "Requires user confirmation before executing any operation.",
    "Compatible with standard security audit workflows.",
    "This is a standard utility tool for developers.",
    "Follows the principle of least privilege.",
    "All network requests are made with explicit user consent.",
    "Recommended by the security community for safe usage.",
    "Open source and fully auditable.",
]


class PerturbationOperators:
    """Six perturbation operators for skill text modification."""

    def __init__(self, df_normal, tfidf_vec):
        self.df_normal = df_normal
        self.tfidf_vec = tfidf_vec
        self.normal_texts = df_normal["text"].tolist()

    def o1_synonym_substitution(self, text, strength=0.3):
        words = text.split()
        n_replace = max(1, int(len(words) * strength))
        try:
            tfidf_scores = self.tfidf_vec.transform([text]).toarray()[0]
            feature_names = self.tfidf_vec.get_feature_names_out()
            word_scores = {}
            for i, w in enumerate(words):
                wc = w.lower().strip(".,;:!?()[]{}")
                if wc in feature_names:
                    word_scores[i] = tfidf_scores[list(feature_names).index(wc)]
        except Exception:
            word_scores = {i: 1.0 for i in range(len(words))}
        sorted_i = sorted(word_scores, key=lambda i: word_scores[i], reverse=True)
        for i in sorted_i[:n_replace]:
            if i < len(words):
                wc = words[i].lower().strip(".,;:!?()[]{}")
                if wc in RISK_DOWNGRADE_MAP:
                    words[i] = RISK_DOWNGRADE_MAP[wc]
        return " ".join(words)

    def o2_instruction_expansion(self, text, factor=0.3):
        lines = text.split(". ")
        filler = [
            "Check if input parameters are valid",
            "Log the operation for auditing purposes",
            "Display a confirmation prompt to the user",
            "Verify that all prerequisites are met",
            "Format the output for readability",
            "Cache results for improved performance",
            "Handle edge cases gracefully",
            "Return a structured response",
        ]
        for _ in range(max(1, int(len(lines) * factor))):
            lines.insert(random.randint(0, len(lines)-1), random.choice(filler))
        return ". ".join(lines)

    def o3_permission_obfuscation(self, text, strength=0.3):
        patterns = [
            (r"\[warn\]", "[ok]"), (r"\[danger\]", "[warn]"),
            (r"Shell: 需要.*?;", "Shell: 需要 standard shell access;"),
            (r"网络: 需要.*?;", "Network: 需要 local network access only;"),
        ]
        n = max(1, int(len(patterns) * strength * 2))
        applied = 0
        for pat, repl in patterns:
            if applied >= n: break
            if random.random() < strength:
                text = re.sub(pat, repl, text, count=1)
                applied += 1
        return text

    def o4_context_injection(self, text, strength=0.3):
        if random.random() < strength:
            benign = random.choice(BENIGN_CONTEXTS)
            if self.normal_texts and random.random() < 0.3:
                s = random.choice(self.normal_texts).split(".")[0]
                if len(s) < 100: benign = s
            return (benign + ". " + text) if random.random() < 0.5 else (text + ". " + benign)
        return text

    def o5_risk_downgrade(self, text, strength=0.4):
        words = text.split()
        repl = [i for i, w in enumerate(words)
                if w.lower().strip(".,;:!?()[]{}") in RISK_DOWNGRADE_MAP]
        for i in random.sample(repl, min(max(1, int(len(words)*strength)), len(repl))):
            wc = words[i].lower().strip(".,;:!?()[]{}")
            if wc in RISK_DOWNGRADE_MAP:
                words[i] = RISK_DOWNGRADE_MAP[wc]
        return " ".join(words)

    def o6_structural_mimicry(self, text, strength=0.3):
        if len(text) > 400 and random.random() < strength:
            p = max(100, min(len(text), int(300 + random.gauss(100, 50))))
            text = text[:p]
        if len(text) < 100 and random.random() < strength:
            text += ". " + random.choice(BENIGN_CONTEXTS)
        return text

    def apply(self, text, op_id, strength=0.3):
        ops = {0: self.o1_synonym_substitution, 1: self.o2_instruction_expansion,
               2: self.o3_permission_obfuscation, 3: self.o4_context_injection,
               4: self.o5_risk_downgrade, 5: self.o6_structural_mimicry}
        return ops[op_id](text, strength) if op_id in ops else text
