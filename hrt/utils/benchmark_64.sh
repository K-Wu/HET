# Find the script path
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

source "${DIR}/_assert_in_het_dev_root.sh"
assert_in_het_dev_root

source "${DIR}/_do_all_cases.sh"
DimsX=( 64 )
DimsY=( 64 )

do_all_cases "misc/artifacts/benchmark_64_$(date +%Y%m%d%H%M)" \
  "" \
  ">>\"\${OUTPUT_DIR}/\$m.\$d.\$mf.\${c//[[:blank:]]/}.\$dimx.\$dimy.1.result.log\" 2>&1"
