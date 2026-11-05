#!/usr/bin/env bash
# Render every docs/diagrams/NN-*.mmd to SVG (for ARCHITECTURE.md) and PNG (for slides).
#   docs/diagrams/render.sh            # all diagrams
#   docs/diagrams/render.sh 01-etl     # one diagram
# Uses mermaid-cli through npx (nothing installed globally) and a local Chrome:
# PUPPETEER_EXECUTABLE_PATH overrides the macOS default below.
set -euo pipefail
cd "$(dirname "$0")"

chrome="${PUPPETEER_EXECUTABLE_PATH:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
if [ -x "$chrome" ]; then
  printf '{"executablePath": "%s", "args": ["--no-sandbox"]}\n' "$chrome" > "$tmp/puppeteer.json"
else
  echo '{"args": ["--no-sandbox"]}' > "$tmp/puppeteer.json"   # let puppeteer fetch its own Chromium
fi
mmdc=(npx -y -p @mermaid-js/mermaid-cli@11 mmdc -c mermaid.config.json -p "$tmp/puppeteer.json" -b white)

if [ $# -gt 0 ]; then names=("$@"); else names=(); for f in [0-9]*.mmd; do names+=("${f%.mmd}"); done; fi
for name in "${names[@]}"; do
  src="$name.mmd"
  # Mermaid has no include: put the shared classDefs right after a flowchart's `flowchart ...`
  # line (after any front matter); sequence diagrams are copied as they are.
  awk -v classes=_classes.mmd '
    { print }
    !done && /^flowchart / { while ((getline line < classes) > 0) print line; done = 1 }
  ' "$src" > "$tmp/$src"
  "${mmdc[@]}" -i "$tmp/$src" -o "$name.svg" -q
  "${mmdc[@]}" -i "$tmp/$src" -o "$name.png" -s 2 -q
  echo "rendered $name (.svg, .png)"
done
