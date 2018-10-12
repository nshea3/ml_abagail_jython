#!/bin/bash
# edit the classpath to to the location of your ABAGAIL jar file
#
export CLASSPATH=ABAGAIL.jar:$CLASSPATH
mkdir -p data/plot logs image

# abalone test
echo "Running abalone test"
jython well_test.py
