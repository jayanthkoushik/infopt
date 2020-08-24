#!/usr/bin/env sh

set -e
cd paper/

if [ ! -z $(git diff --cached --name-only --raw -- "paper/fig/alg.pdf") ]; then
    (set -x ; mdc fig/src/alg.md -o fig/alg.pdf -i commands.md -t standalone)
fi

if [ ! -z $(git diff --cached --name-only --raw -- "paper/main.pdf") ]; then
    (set -x ; mdc main.md -o main.pdf -i commands.md -b references.bib -t simple)
fi
