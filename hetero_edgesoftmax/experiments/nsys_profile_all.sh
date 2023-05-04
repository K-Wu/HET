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
                c=${CompactFlag[$c_idx]}
                mf=${MulFlag[$mf_idx]}
                # skip if c and mf both equal 0
                #if [ $c_idx -eq 0 ] && [ $mf_idx -eq 0 ]
                #then
                #    continue
                #fi
                 nsys profile --force-overwrite true -o "$m.$d.$mf.$c" python -m python.$m -d $d --num_layers 1  --full_graph_training --num_classes 64 --n_infeat 64 $c $mf -e 1 --no_warm_up
            done
        done
    done
done

for d in ${Datasets[@]}
do
    for c_idx in {0..1}
    do
        c=${CompactFlag[$c_idx]}
        nsys profile --force-overwrite true -o "RGCNSingleLayer.$d.$c" python -m python.RGCN.RGCNSingleLayer -d $d $c --num_layers 1  --full_graph_training --num_classes 64 --n_infeat 64 -e 1 --no_warm_up
    done
done