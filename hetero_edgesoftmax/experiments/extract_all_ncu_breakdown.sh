declare -a MODELS=("HGT.train" "RGAT.train")
# ("HGT.train" "RGAT.train" "RGCN.RGCNSingleLayer")
declare -a CompactFlag=("--compact_as_of_node_flag" "")
declare -a MulFlag=("--multiply_among_weights_first_flag" "")
declare -a Datasets=("aifb" "mutag" "bgs" "biokg" "am" "wikikg2" "mag" "fb15k")
# --print-details=header 
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
                nv-nsight-cu-cli --csv --section regex:Speed --import "$m.$d.$mf.$c.ncu-rep" >"$m.$d.$mf.$c.log" 2>&1
            done
        done
    done
done

for d in ${Datasets[@]}
do
    nv-nsight-cu-cli --csv --section regex:Speed --import "RGCNSingleLayer.$d.ncu-rep" >"RGCNSingleLayer.$d.log" 2>&1
done