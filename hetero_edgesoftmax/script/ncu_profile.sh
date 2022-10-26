nv-nsight-cu-cli --export "EdgeSoftmax" --force-overwrite --target-processes application-only --kernel-name-base function --kernel-name regex:EdgeSoftmax --launch-skip-before-match 0 --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section WarpStateStats --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --profile-from-start 1 --cache-control all --clock-control base ./build/exe/src/kernel.cu.exe

nv-nsight-cu-cli --export "HGT" --force-overwrite --target-processes application-only --kernel-name-base function --kernel-name regex:HGT --launch-skip-before-match 0 --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section WarpStateStats --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --profile-from-start 1 --cache-control all --clock-control base ./build/exe/src/test_DGLHackKernel.cu.exe

nv-nsight-cu-cli --export "EdgeSoftmax" --force-overwrite --target-processes application-only --kernel-name-base function --kernel-name regex:RgcnLayer1 --launch-skip-before-match 0 --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section WarpStateStats --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --profile-from-start 1 --cache-control all --clock-control base ../build/hetero_edgesoftmax/test_DGLHackSimpleLoadBalance_cu_cc