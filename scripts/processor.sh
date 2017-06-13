#!/bin/bash

for i in ../results/$1/$2/$3/tweet_list_* ; do sed 's/nepal/Nepal/g' $i > tmp.txt | mv tmp.txt $i ; echo $1 ; done
echo "Completed"
