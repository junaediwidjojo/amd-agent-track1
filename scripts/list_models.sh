#!/usr/bin/env bash
# List chat-capable Fireworks models available to your API key.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  echo "Copy .env.example to .env and set FIREWORKS_API_KEY"
  exit 1
fi

set -a
# shellcheck disable=SC1091
source "$ROOT/.env"
set +a

curl -sS -H "Authorization: Bearer $FIREWORKS_API_KEY" \
  "${FIREWORKS_BASE_URL}/models" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
chat = [m['id'] for m in data.get('data', []) if m.get('supports_chat')]
if not chat:
    print('No chat models found for this API key.')
    sys.exit(1)
print('Chat models available to your key:\n')
for m in chat:
    print(f'  {m}')
print('\nSuggested ALLOWED_MODELS for .env (starter):')
print(f'ALLOWED_MODELS={chat[0]}')
if len(chat) > 1:
    print(f'# with fallback:')
    print(f'ALLOWED_MODELS={chat[0]},{chat[1]}')
"
