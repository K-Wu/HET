do_all_cases() {
  declare -a MODELS=("HGT" "RGAT")
  # declare -a MODELS=("RGAT")
  declare -a CompactFlag=("--compact_as_of_node_flag" "" "--compact_as_of_node_flag --compact_direct_indexing_flag")
  declare -a MulFlag=("--multiply_among_weights_first_flag" "")
  declare -a Datasets=("aifb" "mutag" "bgs" "am" "mag" "wikikg2" "fb15k" "biokg")
  # TODO: num_classes need to be out_feat_per_head * num_heads
  OUTPUT_DIR=$1
  mkdir -p ${OUTPUT_DIR}
  echo "OUTPUT_DIR: ${OUTPUT_DIR}"

  for m in ${MODELS[@]}; do
    for d in ${Datasets[@]}; do
      for c_idx in {0..2}; do
        for mf_idx in {0..1}; do
          #skip if c == 0
          # if [ $c_idx -eq 1 ]
          # then
          #     continue
          # fi
          c=${CompactFlag[$c_idx]}
          mf=${MulFlag[$mf_idx]}
          # print command to the log file
          echo "python -m python.$m.train -d $d --num_layers 1  --full_graph_training --num_classes 64 --n_infeat 64 --num_heads 1 $c $mf"
          eval $2 python -m python.$m.train -d $d --num_layers 1 --full_graph_training --num_classes 64 --n_infeat 64 --num_heads 1 $c $mf $3
        done
      done
    done
  done

  m="RGCN"
  mf=""

  for d in ${Datasets[@]}; do
    for c_idx in {0..2}; do
      # if [ $c_idx -eq 1 ]
      # then
      #     continue
      # fi
      c=${CompactFlag[$c_idx]}
      # two dots in the name because mul flag is not set
      echo "python -m python.RGCN.RGCNSingleLayerSeparateCOO -d $d --num_layers 1  --full_graph_training --num_classes 64 --n_infeat 64 $c"
      eval $2 python -m python.RGCN.RGCNSingleLayerSeparateCOO -d $d --num_layers 1 --full_graph_training --num_classes 64 --n_infeat 64 $c $3
    done
  done
}
# TODO: generalize in_feat, out_feat, num_heads
