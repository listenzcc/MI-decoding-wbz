#!/bin/sh

script=summary-decoding-results.py

python $script -n fbcnet -m fbcnet
python $script -n fbcsp-vote -m softvote
python $script -n fbcsp-vote -m hardvote
python $script -n fbcsp-info -m default