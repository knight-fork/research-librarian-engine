#!/usr/bin/env python3
"""Refuse to commit secrets. Scans staged files (or given paths) for:
  * the literal values of every variable in .env (API keys, passwords, ids, emails);
  * common credential patterns (Google / Anthropic / GitHub / AWS / Slack keys, private keys, key=value secrets).
Exit status 1 if anything is found. Installed as a pre-commit hook by scripts/install_hooks.sh."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATTERNS = {
    "Google API key": r"AIza[0-9A-Za-z_\-]{35}",
    "Google AI Studio key": r"\bAQ\.[0-9A-Za-z_\-]{20,}",
    "Anthropic key": r"sk-ant-[0-9A-Za-z_\-]{20,}",
    "OpenAI-style key": r"\bsk-[A-Za-z0-9]{32,}",
    "GitHub token": r"\bgh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}",
    "AWS access key": r"\bAKIA[0-9A-Z]{16}\b",
    "Slack token": r"\bxox[abprs]-[0-9A-Za-z\-]{10,}",
    "private key": r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    "quoted secret literal": r"(?i)\b(?:api[_-]?key|secret|password|token)\w*['\"]?\s*[:=]\s*['\"][A-Za-z0-9_\-\.+/]{16,}['\"]",
    "env-style secret": r"(?m)^\s*[A-Z0-9_]*(?:API_KEY|SECRET|PASSWORD|TOKEN)[A-Z0-9_]*\s*=\s*[A-Za-z0-9_\-\.+/]{12,}\s*$",
}
ALLOW_FILES = {"scripts/check_secrets.py"}  # contains the patterns themselves


def env_values():
    """Values from this repo's .env plus any files in CHECK_SECRETS_ENV_FILES (colon-separated paths)."""
    import os
    files = [ROOT / ".env"] + [Path(x).expanduser() for x in os.environ.get("CHECK_SECRETS_ENV_FILES", "").split(":") if x]
    vals = {}
    for env in files:
        if not env.exists():
            continue
        for line in env.read_text().splitlines():
            m = re.match(r"\s*([A-Z0-9_]+)\s*=\s*(.*)$", line)
            if m:
                v = m.group(2).split(" #")[0].strip().strip("'\"")
                if len(v) >= 6 and v.lower() not in ("user", "group", "true", "false"):
                    vals[m.group(1)] = v
    return vals


def staged_files():
    out = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if f]


def content(path, staged):
    if staged:
        r = subprocess.run(["git", "show", f":{path}"], cwd=ROOT, capture_output=True)
        return r.stdout.decode("utf-8", "replace")
    return (ROOT / path).read_text("utf-8", "replace")


def main(argv):
    staged = not argv
    files = staged_files() if staged else argv
    values = env_values()
    problems = []
    for f in files:
        if f in (".env",) or (f.startswith(".env") and f != ".env.example"):
            problems.append(f"{f}: environment file must never be committed")
            continue
        text = content(f, staged)
        for name, v in values.items():
            if v in text:
                problems.append(f"{f}: contains the value of {name} from .env")
        if f in ALLOW_FILES:
            continue
        for label, rx in PATTERNS.items():
            for m in re.finditer(rx, text):
                line = text.count("\n", 0, m.start()) + 1
                problems.append(f"{f}:{line}: looks like a {label}")
    if problems:
        print("Secret check FAILED - nothing was committed:")
        for p in problems:
            print("  " + p)
        return 1
    print(f"Secret check passed ({len(files)} file(s), {len(values)} .env value(s) checked).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
