# shared by the scripts: project root and interpreter
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
PY="${PYTHON:-$HOME/.venvs/odysseys/bin/python}"
if [ ! -x "$PY" ]; then
  echo "Python interpreter not found at $PY. Create it with:" >&2
  echo "  python3 -m venv ~/.venvs/odysseys && ~/.venvs/odysseys/bin/pip install -r requirements.txt && ~/.venvs/odysseys/bin/playwright install chromium" >&2
  exit 1
fi
