# source this file to execute
declare -a MODELS=("HGT" "RGAT")
declare -a CompactFlag=("--compact_as_of_node_flag" "" "--compact_as_of_node_flag --compact_direct_indexing_flag")
declare -a MulFlag=("--multiply_among_weights_first_flag" "")
declare -a Datasets=("aifb" "mutag" "bgs" "am" "mag" "wikikg2" "fb15k" "biokg")

OUTPUT_DIR="misc/artifacts/ncu_breakdown_`date +%Y%m%d%H%M`"
mkdir -p ${OUTPUT_DIR}

for m in ${MODELS[@]}
do
    for d in ${Datasets[@]}
    do
        for c_idx in {0..2}
        do
            for mf_idx in {0..1}
            do
                c=${CompactFlag[$c_idx]}
                mf=${MulFlag[$mf_idx]}
                # skip if c and mf both equal 0
                #if [ $c_idx -eq 0 ] && [ $mf_idx -eq 0 ]
                #then
                #    continue
                #fi
                ncu --export "${OUTPUT_DIR}/$m.$d.$mf.${c//[[:blank:]]/}" --force-overwrite --target-processes all --kernel-name-base function --kernel-name regex:HET --launch-skip-before-match 0 --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_HierarchicalSingleRooflineChart --section WarpStateStats --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --profile-from-start 1 --cache-control all --clock-control base python -m python.$m.train -d $d --num_layers 1  --full_graph_training --num_classes 64 --n_infeat 64 $c $mf -e 1 --no_warm_up
            done
        done
    done
done

for d in ${Datasets[@]}
do
    for c_idx in {0..2}
    do
        c=${CompactFlag[$c_idx]}
        # two dots in the name because mul flag is not set
        ncu --export "$OUTPUT_DIR/RGCN.$d..${c//[[:blank:]]/}" --force-overwrite --target-processes all --kernel-name-base function --kernel-name regex:HET --launch-skip-before-match 0 --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_HierarchicalSingleRooflineChart --section WarpStateStats --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --profile-from-start 1 --cache-control all --clock-control base python -m python.RGCN.RGCNSingleLayerSeparateCOO -d $d $c --num_layers 1  --full_graph_training --num_classes 64 --n_infeat 64 -e 1 --no_warm_up
    done
done
