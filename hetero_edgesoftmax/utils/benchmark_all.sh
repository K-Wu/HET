# Find the script path
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${DIR}/_do_all_cases.sh"
do_all_cases "misc/artifacts/benchmark_all_$(date +%Y%m%d%H%M)" \
  "" \
  ">>\"\${OUTPUT_DIR}/\$m.64.64.1.\$d.\$mf.\${c//[[:blank:]]/}.result.log\" 2>&1"
