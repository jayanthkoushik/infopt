#!/usr/bin/env sh

for f in "$@"; do
    isort $f
done
