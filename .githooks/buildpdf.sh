#!/usr/bin/env sh -e

read -ra diffiles <<< $(git diff --cached --name-only)

elementin () {
    local e match="$1"
    shift
    for e; do [[ "$e" == "$match" ]] && return 0; done
    return 1
}

cd paper/
if elementin "paper/fig/_src/alg.md" "${diffiles[@]}"; then
    mdc fig/_src/alg.md -o fig/alg.pdf -i commands.md -t standalone
fi

if ! elementin "paper/main.pdf" "${diffiles[@]}"; then
    mdc main.md -o main.pdf -i commands.md -b references.bib -t simple
fi
