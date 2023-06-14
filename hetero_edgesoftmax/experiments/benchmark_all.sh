# source this file to execute
# declare -a MODELS=("HGT.train" "RGAT.train")
declare -a MODELS=("RGAT.train")
declare -a CompactFlag=("--compact_as_of_node_flag" "")
declare -a MulFlag=("--multiply_among_weights_first_flag" "")
declare -a Datasets=("aifb" "mutag" "bgs" "am" "mag" "wikikg2" "fb15k" "biokg")

OUTPUT_DIR="misc/artifacts/benchmark_all_`date +%Y%m%d%H%M`"
mkdir -p ${OUTPUT_DIR}

for m in ${MODELS[@]}
do
    for d in ${Datasets[@]}
    do
        for c_idx in {0..1}
        do
            for mf_idx in {0..1}
            do
                #skip if c == 0
                # if [ $c_idx -eq 1 ]
                # then
                #     continue
                # fi
                c=${CompactFlag[$c_idx]}
                mf=${MulFlag[$mf_idx]}
                # print command to the log file
                echo "python -m python.$m -d $d --num_layers 1  --full_graph_training --num_classes 64 --n_infeat 64 $c $mf" 
                echo "python -m python.$m -d $d --num_layers 1  --full_graph_training --num_classes 64 --n_infeat 64 $c $mf" >>"${OUTPUT_DIR}/$m.$mf.$c.log"
                python -m python.$m -d $d --num_layers 1  --full_graph_training --num_classes 64 --n_infeat 64 $c $mf >>"${OUTPUT_DIR}/$m.$mf.$c.log" 2>&1
            done
        done
    done
done

for d in ${Datasets[@]}
do
    for c_idx in {0..1}
    do
        if [ $c_idx -eq 1 ]
        then
            continue
        fi
        echo "python -m python.RGCN.RGCNSingleLayerSeparateCOO -d $d --num_layers 1  --full_graph_training --num_classes 64 --n_infeat 64 $c"
        echo "python -m python.RGCN.RGCNSingleLayerSeparateCOO -d $d --num_layers 1  --full_graph_training --num_classes 64 --n_infeat 64 $c" >>"${OUTPUT_DIR}/RGCNSingleLayer.$c.log"
        python -m python.RGCN.RGCNSingleLayerSeparateCOO -d $d --num_layers 1  --full_graph_training --num_classes 64 --n_infeat 64 $c >>"${OUTPUT_DIR}/RGCNSingleLayer.$c.log" 2>&1
    done
done