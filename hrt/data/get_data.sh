DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# change directory to DIR and then recover after finishes
pushd "${DIR}" >/dev/null 2>&1
git clone --recursive https://tony_wukun_worker:PxXRTULrXSd7w4vAXygq@bitbucket.org/tonywuk/myhyb_dataset.git MyHybData
git clone --recursive https://tony_wukun_worker:PxXRTULrXSd7w4vAXygq@bitbucket.org/tonywuk/mywikikg2_dataset MyWikiKG2
git clone --recursive https://K-Wu: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@github.com/K-Wu/_hetero_edgesoftmax_data_ogbn_mag ogbn_mag
popd