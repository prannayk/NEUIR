#!/bin/bash

hm=$(echo "2/((1/$1) + (1/$2))" | bc -l)
echo $hm
