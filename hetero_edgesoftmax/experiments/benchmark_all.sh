# source this file to execute
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
                # skip if c == 0
                # if [ $c_idx -eq 1 ]
                # then
                #     continue
                # fi
                c=${CompactFlag[$c_idx]}
                mf=${MulFlag[$mf_idx]}
                # print command to the log file
                echo "python -m python.$m -d $d --num_layers 1  --full_graph_training --num_classes 64 --n_infeat 64 $c $mf" >>"$m.$mf.$c.log"
                python -m python.$m -d $d --num_layers 1  --full_graph_training --num_classes 64 --n_infeat 64 $c $mf >>"$m.$mf.$c.log" 2>&1
            done
            python -m python.RGCN.RGCNSingleLayerSeparateCOO -d $d --num_layers 1  --full_graph_training --num_classes 64 --n_infeat 64 $c >>"RGCNSingleLayer.$c.log" 2>&1
        done
    done
done
