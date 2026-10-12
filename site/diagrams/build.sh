#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ../media
for f in arch-*.tex; do
  case "$f" in arch-parkour.tex) continue ;; esac
  pdflatex -interaction=nonstopmode -halt-on-error "$f" >/dev/null
  dvisvgm --pdf --no-fonts -o "../media/${f%.tex}.svg" "${f%.tex}.pdf"
done
rm -f *.aux *.log *.pdf
echo "wrote site/media/arch-*.svg"
