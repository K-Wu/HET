# Find the script path
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

source "${DIR}/_assert_in_het_dev_root.sh"
assert_in_het_dev_root

source "${DIR}/_do_all_cases.sh"
do_all_cases "misc/artifacts/ncu_breakdown_$(date +%Y%m%d%H%M)" \
  "ncu --export "\$OUTPUT_DIR/\$m.\$d.\$mf.\${c//[[:blank:]]/}.\$dimx.\$dimy.1" --force-overwrite --target-processes all --kernel-name-base function --kernel-name regex:HET --launch-skip-before-match 0 --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_HierarchicalDoubleRooflineChart --section SpeedOfLight_HierarchicalHalfRooflineChart --section SpeedOfLight_HierarchicalSingleRooflineChart --section SpeedOfLight_HierarchicalTensorRooflineChart --section SpeedOfLight_RooflineChart  --section WarpStateStats --profile-from-start 1 --cache-control all --clock-control base" \
  "-e 1 --no_warm_up"
