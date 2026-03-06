#!/bin/bash
# =============================================================================
# PTCV Secrets Loader (Bash)
#
# Loads environment variables from .secrets file.
#
# Usage:
#   source ./load-secrets.sh       # Load secrets into current session
#   . ./load-secrets.sh            # Alternative syntax
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECRETS_FILE="$SCRIPT_DIR/.secrets"

if [ ! -f "$SECRETS_FILE" ]; then
    echo "[ERROR] Secrets file not found: $SECRETS_FILE"
    echo "[INFO] Copy .secrets.example to .secrets and fill in your credentials"
    return 1 2>/dev/null || exit 1
fi

echo "[INFO] Loading secrets from: $SECRETS_FILE"

loaded=0
while IFS= read -r line || [ -n "$line" ]; do
    # Trim whitespace
    line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

    # Skip empty lines and comments
    if [ -n "$line" ] && [ "${line:0:1}" != "#" ]; then
        # Extract name and value
        name=$(echo "$line" | cut -d'=' -f1)
        value=$(echo "$line" | cut -d'=' -f2-)

        # Export the variable
        export "$name"="$value"

        # Show masked value for confirmation
        if [ ${#value} -gt 8 ]; then
            masked="${value:0:4}****${value: -4}"
        else
            masked="****"
        fi
        echo "  $name = $masked"
        ((loaded++))
    fi
done < "$SECRETS_FILE"

echo "[INFO] Loaded $loaded environment variables"
