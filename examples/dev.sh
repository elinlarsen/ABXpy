#!/bin/bash
#
# This test contains a full run of the ABX pipeline in command line
# with randomly created database and features

data=$(dirname $(realpath "${BASH_SOURCE[0]}"))/example_items/data

# input files already here (if they are not here, please run
# complete_run.py)
item=$data.item
features=$data.features

# output files produced by ABX
task=$data.abx
distance=$data.distance
score=$data.score
analyze=$data.csv

rm -rf $task $distance $score $analyze

# generating task file
#args="--on c0 --across c1"
#args="--on c0 --across c1 -K 100"
args="--on c0 --across c1 --by c2 -K 100"
abx-task --stats-only $item $args || exit 1
abx-task $item $task --verbose $args || exit 1

# computing distances
abx-distance $features $task $distance --normalization 1 --njobs 1

# calculating the score
abx-score $task $distance $score

# collapsing the results
abx-analyze $score $task $analyze

wc -l $analyze
cat $analyze | sed '1d' | awk '{ print $7 }' | sort -n | uniq | xargs
