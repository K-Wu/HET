declare -a MODELS=("HGT.train" "RGAT.train")
declare -a CompactFlag=("--compact_as_of_node_flag" "")
declare -a MulFlag=("--multiply_among_weights_first_flag" "")
declare -a Datasets=("aifb" "mutag" "bgs" "am" "mag" "wikikg2" "fb15k" "biokg")

for m in ${MODELS[@]}
do
    for d in ${Datasets[@]}
    do
        for c_idx in {0..1}
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
                nv-nsight-cu-cli --export "$m.$d.$mf.$c" --force-overwrite --target-processes all --kernel-name-base function --kernel-name regex:HET --launch-skip-before-match 0 --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section WarpStateStats --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --profile-from-start 1 --cache-control all --clock-control base python -m python.$m -d $d --num_layers 1  --full_graph_training --num_classes 64 --n_infeat 64 $c $mf -e 1 --no_warm_up
            done
        done
    done
done

for d in ${Datasets[@]}
do
    nv-nsight-cu-cli --export "RGCNSingleLayer.$d" --force-overwrite --target-processes all --kernel-name-base function --kernel-name regex:HET --launch-skip-before-match 0 --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section WarpStateStats --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --profile-from-start 1 --cache-control all --clock-control base python -m python.RGCN.RGCNSingleLayer -d $d --num_layers 1  --full_graph_training --num_classes 64 --n_infeat 64 -e 1 --no_warm_up
done