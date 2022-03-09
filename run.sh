#!/bin/bash
cd build
cmake ..
make install
cd ..

./bin/print

# for v in {4..4..1}
# do
#     ./bin/verify -k 0 -v $v
#     ./bin/speedtest -k 0 -v $v
# done