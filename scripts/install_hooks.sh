#!/bin/sh
# Install the secret-scanning pre-commit hook (run once after cloning).
cd "$(dirname "$0")/.." || exit 1
printf '#!/bin/sh\npython3 scripts/check_secrets.py || exit 1\n' > .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
echo "pre-commit secret check installed"
