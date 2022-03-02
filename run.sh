#!/bin/bash
cd build
cmake ..
make install
cd ..

for v in {1..2..1}
do
    ./bin/verify -k 0 -v $v
    ./bin/speedtest -k 0 -v $v
done