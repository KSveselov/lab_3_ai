#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
MLFLOW_URL="http://127.0.0.1:5000"

if [[ ! -x "$VENV_DIR/bin/mlflow" ]]; then
  echo "MLflow не найден в $VENV_DIR. Сначала установите зависимости:"
  echo "  source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

cd "$PROJECT_DIR"
export PATH="$VENV_DIR/bin:$PATH"

cleanup() {
  kill "$MLFLOW_PID" 2>/dev/null || true
}

mlflow ui \
  --backend-store-uri "$PROJECT_DIR/src/mlruns" \
  --host 127.0.0.1 \
  --port 5000 &
MLFLOW_PID=$!
trap cleanup EXIT INT TERM

for _ in {1..30}; do
  if curl --silent --fail "$MLFLOW_URL" >/dev/null; then
    echo "Открываю $MLFLOW_URL"
    xdg-open "$MLFLOW_URL" >/dev/null 2>&1 || echo "Откройте вручную: $MLFLOW_URL"
    wait "$MLFLOW_PID"
    exit 0
  fi
  sleep 1
done

echo "MLflow UI не запустился. Проверьте сообщения выше."
exit 1
