#!/bin/bash
exec /usr/local/hygon/dtk-25.04/bin/hipprof --hip-trace --trace-off --session strata-deep-baseline -d /root/code/strata-dcu/deep-baseline-20261021 -o trace.csv /root/code/strata-dcu/build-dcu/strata "$@"
