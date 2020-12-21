#!/bin/bash
NUM_CLIENTS=3

echo "Starting $NUM_CLIENTS clients."
for ((i = 1; i < $NUM_CLIENTS+1; i++))
do
    echo "Starting client(cid=$i) with partition $i out of $NUM_CLIENTS clients."
    python -m client --partition=$i --monet=False &
done
echo "Started $NUM_CLIENTS clients."