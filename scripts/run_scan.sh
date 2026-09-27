#!/bin/zsh
# Scheduled entry point: run daily. The broad scan runs every scan.interval_days; each alert on its own cadence.
cd "$(dirname "$0")/.." || exit 1
mkdir -p data/logs
./.venv/bin/python -m src.cli scan --if-due >> "data/logs/scan-$(date +%Y-%m).log" 2>&1
./.venv/bin/python -m src.cli watch >> "data/logs/scan-$(date +%Y-%m).log" 2>&1
# Saved alerts: each runs on its own cadence (default every 30 days); only due alerts execute.
./.venv/bin/python -m src.cli alerts run-due >> "data/logs/scan-$(date +%Y-%m).log" 2>&1
