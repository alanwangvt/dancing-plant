#!/bin/bash
git pull
python launch/batchTrace.py /projects/deep4cbia/datasetlights 4
python launch/read_trace.py /projects/deep4cbia/datasetlights 4