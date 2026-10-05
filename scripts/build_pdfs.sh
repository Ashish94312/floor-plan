#!/usr/bin/env bash
# Render the submission docs (brief's Deliverables 1, 2, 5, 6, 7) from Markdown to PDF in deliverables/, and print
# each PDF's page count against the brief's caps (protocol 1 page, fix declaration 1 page, technical report 6 pages).
# Needs pandoc, weasyprint and poppler (brew install pandoc poppler; pip install weasyprint). Style: scripts/pdf/style.css;
# one-page docs also get scripts/pdf/compact.css.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=deliverables
mkdir -p "$OUT"
TMP=$(mktemp -d)
trap 'rm -rf "${TMP:?}"' EXIT

DOCS=(  # file, page cap (0 = none)
  "docs/COMPLIANCE_MATRIX.md 0"
  "docs/CAPTURE_PROTOCOL.md 1"
  "docs/DEVICE_MATRIX.md 0"
  "reports/benchmark.md 0"
  "docs/FIX_DECLARATION.md 1"
  "reports/fix_loop.md 0"
  "docs/TECHNICAL_REPORT.md 6"
)

for row in "${DOCS[@]}"; do
  read -r md cap <<< "$row"
  name=$(basename "$md" .md)
  pdf="$OUT/$name.pdf"
  css=(--css scripts/pdf/style.css)
  if [ "$cap" -eq 1 ]; then css+=(--css scripts/pdf/compact.css); fi
  pandoc "$md" -f gfm -t html5 --standalone --embed-resources \
    --metadata pagetitle="$(sed -n '1s/^# //p' "$md")" \
    "${css[@]}" --lua-filter scripts/pdf/links.lua -o "$TMP/$name.html"
  weasyprint -q "$TMP/$name.html" "$pdf"
  pages=$(pdfinfo "$pdf" | awk '/^Pages:/ {print $2}')
  verdict=""
  if [ "$cap" -gt 0 ]; then
    if [ "$pages" -le "$cap" ]; then verdict="  (cap $cap: ok)"; else verdict="  (cap $cap: OVER)"; fi
  fi
  printf '%-28s %2s pages%s\n' "$pdf" "$pages" "$verdict"
done
