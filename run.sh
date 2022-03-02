#!/bin/bash
cd build
cmake ..
make install
cd ..

for v in {3..3..1}
do
    ./bin/verify -k 0 -v $v
    ./bin/speedtest -k 0 -v $v
done