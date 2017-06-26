#!/bin/bash

head -n$1 $2 | cut -d':' -f2 | sort -nr | head -n10
echo "lol"
cut -d':' -f2 $2 | sort -nr | head -n10
