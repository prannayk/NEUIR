#!/bin/bash

t=$(ls -la ../results/$1/$2/$3/ | grep -c ",*")
echo $t
