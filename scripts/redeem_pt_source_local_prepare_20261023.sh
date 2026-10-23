#!/usr/bin/env bash
# Laptop, not Codon. This script does not terminate the interactive shell.
HOME_DOWNLOADS="$HOME/Downloads"
PKG="$HOME_DOWNLOADS/REDEEM_PT_MULTISHEET_FIX_20261023"
SOURCE="$HOME_DOWNLOADS/ProteomeTools_NatMethods2017_Supplementary_Table1.xlsx"
CAT="$HOME_DOWNLOADS/redeem_pt_synthesis_catalog_3sheets_20261023"
OLD="$HOME_DOWNLOADS/redeem_external_dev_validation_20261022"
NEW="$HOME_DOWNLOADS/redeem_pt_2017_new_validation_20261022"
COVERAGE="$HOME_DOWNLOADS/redeem_pt_2017_new_validation_coverage_20261023.json"
BRIDGE="$PKG/redeem_external_dev_benchmark_20261022.py"
SCRIPT="$PKG/redeem_pt_source_library_20261023.py"
PY="$HOME/.venvs/redeem-external-dev-uv-20261022/bin/python"
MODE="${1:-all}"

if [ ! -f "$SCRIPT" ] || [ ! -s "$BRIDGE" ]; then
  echo 'REDEEM_PT_SOURCE=BLOCKED package source missing'
elif [ ! -x "$PY" ]; then
  echo 'REDEEM_PT_SOURCE=BLOCKED expected uv-managed Python missing; consult runbook'
elif ! "$PY" "$SCRIPT" self-test; then
  echo 'REDEEM_PT_SOURCE=BLOCKED local source tests failed'
else
  case "$MODE" in
    inspect)
      if [ ! -s "$SOURCE" ]; then
        echo "REDEEM_PT_SOURCE=BLOCKED download Supplementary Table 1 XLSX as $SOURCE"
      else
        "$PY" "$SCRIPT" inspect --source "$SOURCE"
      fi
      ;;
    build)
      if [ ! -s "$SOURCE" ]; then
        echo "REDEEM_PT_SOURCE=BLOCKED download Supplementary Table 1 XLSX as $SOURCE"
      elif [ -e "$CAT" ]; then
        echo "REDEEM_PT_SOURCE=BLOCKED preserve existing catalog: $CAT"
      else
        "$PY" "$SCRIPT" build --source "$SOURCE" --output "$CAT" --sheet-set pt2017-synthesis-v1 --expected-source-sha256 6d5a1134bee97644cc3197f4a02e1eb9448cbcd1674f616a46bc385106ffbf17
      fi
      ;;
    cohort)
      if [ -e "$NEW" ]; then
        echo "REDEEM_PT_NEW_DEV=BLOCKED preserve existing output: $NEW"
      elif [ ! -s "$OLD/truth.tsv" ] || [ ! -s "$OLD/requests.tsv" ]; then
        echo 'REDEEM_PT_NEW_DEV=BLOCKED frozen original DEV512 cohort missing'
      else
        "$PY" "$SCRIPT" prepare-new-dev --old-cohort "$OLD" --bridge "$BRIDGE" --output "$NEW"
      fi
      ;;
    coverage)
      if [ -e "$COVERAGE" ]; then
        echo "REDEEM_PT_NEW_COVERAGE=BLOCKED preserve existing report: $COVERAGE"
      elif [ ! -s "$CAT/catalog.yaml" ] || [ ! -s "$NEW/truth.tsv" ]; then
        echo 'REDEEM_PT_NEW_COVERAGE=BLOCKED catalog or new DEV absent'
      else
        "$PY" "$SCRIPT" coverage --catalog-dir "$CAT" --new-cohort "$NEW" --output "$COVERAGE"
      fi
      ;;
    *) echo 'USAGE: bash redeem_pt_source_local_prepare_20261022.sh inspect|build|cohort|coverage' ;;
  esac
fi
