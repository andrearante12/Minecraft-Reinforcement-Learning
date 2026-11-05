#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ../media
for f in arch-*.tex; do
  case "$f" in arch-parkour.tex) continue ;; esac
  pdflatex -interaction=nonstopmode -halt-on-error "$f" >/dev/null
  if grep -q includegraphics "$f"; then
    pdftocairo -svg "${f%.tex}.pdf" "../media/${f%.tex}.svg"   # dvisvgm misplaces embedded images
  else
    dvisvgm --pdf --no-fonts -o "../media/${f%.tex}.svg" "${f%.tex}.pdf"
  fi
done
rm -f *.aux *.log *.pdf
echo "wrote site/media/arch-*.svg"
