do_all_cases() {
  declare -a MODELS=("RGAT" "HGT")
  # declare -a MODELS=("RGAT")
  declare -a CompactFlag=("--compact_as_of_node_flag" "" "--compact_as_of_node_flag --compact_direct_indexing_flag")
  declare -a MulFlag=("--multiply_among_weights_first_flag" "")
  declare -a Datasets=("aifb" "mutag" "bgs" "am" "mag" "wikikg2" "fb15k" "biokg")
  DimsX=( 32 64 128 )
  DimsY=( 32 64 128 )
  # TODO: num_classes need to be out_feat_per_head * num_heads
  OUTPUT_DIR=$1
  mkdir -p ${OUTPUT_DIR}
  echo "OUTPUT_DIR: ${OUTPUT_DIR}"

  for dimx in ${DimsX[@]}; do
    for dimy in ${DimsY[@]}; do
      for m in ${MODELS[@]}; do
        for d in ${Datasets[@]}; do
          for c_idx in ${!CompactFlag[@]}; do
            for mf_idx in ${!MulFlag[@]}; do
              #skip if c == 0
              # if [ $c_idx -eq 1 ]
              # then
              #     continue
              # fi
              c=${CompactFlag[$c_idx]}
              mf=${MulFlag[$mf_idx]}
              # print command to the log file
              echo "python -m python.$m.train -d $d --num_layers 1  --full_graph_training --num_classes $dimx  --n_infeat $dimy --num_heads 1 $c $mf"
              eval $2 python -m python.$m.train -d $d --num_layers 1 --full_graph_training --num_classes $dimx --n_infeat $dimy --num_heads 1 $c $mf $3
            done
          done
        done
      done
    done
  done

  m="RGCN"
  mf=""


  for dimx in ${DimsX[@]}; do
    for dimy in ${DimsY[@]}; do
      for d in ${Datasets[@]}; do
        for c_idx in ${!CompactFlag[@]}; do
          # if [ $c_idx -eq 1 ]
          # then
          #     continue
          # fi
          c=${CompactFlag[$c_idx]}
          # two dots in the name because mul flag is not set
          echo "python -m python.RGCN.RGCNSingleLayerSeparateCOO -d $d --num_layers 1  --full_graph_training --num_classes $dimx --n_infeat $dimy $c"
          eval $2 python -m python.RGCN.RGCNSingleLayerSeparateCOO -d $d --num_layers 1 --full_graph_training --num_classes $dimx --n_infeat $dimy $c $3
        done
      done
    done
  done
}
# TODO: generalize in_feat, out_feat, num_heads
