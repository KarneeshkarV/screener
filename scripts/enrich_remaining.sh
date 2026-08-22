#!/usr/bin/env zsh
set -e
cd /home/karneeshkar/Desktop/projects/trading/screener-pit-mid-small
LOG=findings/pit_midsmall/enrich.log
echo "=== leftover waiter start $(date -Is) ===" >> $LOG
while pgrep -f "run_pit_midsmall_study.py -u n5" >/dev/null 2>&1; do sleep 30; done
while pgrep -f "run_pit_midsmall_study.py --enrich" >/dev/null 2>&1; do sleep 30; done
set -a; source .env; set +a
export SCREENER_PRICE_PROVIDER=yfinance SCREENER_AGENT=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
echo "=== leftover n50/n500 enrich $(date -Is) ===" >> $LOG
for u in n50 n500; do
  echo "--- leftover $u $(date -Is) ---" >> $LOG
  xargs -P 4 -I {} nice -n 15 uv run python scripts/run_pit_midsmall_study.py --enrich -u $u -s {} < /tmp/pit_strats.txt >> $LOG 2>&1
done
uv run python scripts/run_pit_midsmall_study.py --summary-only >> $LOG 2>&1
uv run python scripts/build_pit_report.py >> $LOG 2>&1
echo "=== all ledgers done $(date -Is) ===" >> $LOG
