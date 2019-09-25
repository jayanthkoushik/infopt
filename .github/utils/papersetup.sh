cp paper/main.md /www/
cp paper/references.bib /www/_includes/
cp paper/commands.md /www/_includes/
mkdir -p /www/fig
convert -antialias -density 300 -quality 100 paper/fig/alg.pdf /www/fig/alg.png
convert -antialias -density 300 -quality 100 paper/fig/synth.pdf /www/fig/synth.png
convert -antialias -density 300 -quality 100 paper/fig/mnist.pdf /www/fig/mnist.png
