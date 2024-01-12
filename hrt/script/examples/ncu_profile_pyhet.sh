#!/usr/bin/env bash

# use --kernel-name regex:HET to profile only kernels (implemented in this repo and exposed to pyhet module)
nv-nsight-cu-cli --export "myhgt" --force-overwrite --target-processes all --kernel-name-base function --kernel-name regex:HET --launch-skip-before-match 0 --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section WarpStateStats --profile-from-start 1 --cache-control all --clock-control base python -m python.HGT.train -d am --num_layers 1 --full_graph_training --num_classes 64 --n_infeat 64
