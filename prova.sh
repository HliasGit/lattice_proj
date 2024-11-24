#!/bin/bash

echo ""
echo "Sequential "
./sequential
echo ""

echo "Parallel generic algorithm "
./parallel_mio
echo ""

echo "Parallel fixpoint "
./fixpoint
echo ""