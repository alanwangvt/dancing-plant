#!/bin/bash
git pull
python launch/gen_trace.py
python launch/draw_trace.py
python launch/cluster_trace.py