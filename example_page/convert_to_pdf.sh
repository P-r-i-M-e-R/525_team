pandoc page.md \
  --pdf-engine=xelatex \
  -V mainfont="DejaVu Serif" \
  -V monofont="DejaVu Sans Mono" \
  -V geometry:margin=1in \
  --highlight-style=tango \
  -f markdown \
  -t pdf \
  -o page.pdf