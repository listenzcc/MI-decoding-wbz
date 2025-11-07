#!/bin/sh

# script=decode-fbcsp.py
script=decode-fbcnet.py

python $script -s bxy &
python $script -s bxy2 &
python $script -s gc &
python $script -s gc2 &
python $script -s ljy &
python $script -s ljy2 &
python $script -s lcc &
python $script -s lzb

# python $script -s S1 &
# python $script -s S2 &
# python $script -s S3 &
# python $script -s S4 &
# python $script -s S5 &
# python $script -s S6 &
# python $script -s S7 &
# python $script -s S8 &
# python $script -s S9 &
# python $script -s S10

# python $script -s S11 &
# python $script -s S12 &
# python $script -s S13 &
# python $script -s S14 &
# python $script -s S15 &
# python $script -s S16 &
# python $script -s S17 &
# python $script -s S18 &
# python $script -s S19 &
# python $script -s S20