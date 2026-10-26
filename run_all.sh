#!/bin/bash
# Both tasks use the new query design (Lt x L1 x L2 table, 30 random endpoints per cell, independent per mode).
# Start:  caffeinate -i nohup bash run_all.sh > run_all.log 2>&1 &      Watch:  tail -f run_all.log
cd "$(dirname "$0")"
mkdir -p logs
run() {
    name=$1; shift
    echo "[$(date '+%m-%d %H:%M')] start $name"
    if python "$@" > "logs/$name.log" 2>&1; then
        echo "[$(date '+%m-%d %H:%M')] done  $name -> $(grep -h 'saved' "logs/$name.log" | tail -1)"
    else
        echo "[$(date '+%m-%d %H:%M')] FAILED $name (see logs/$name.log)"
    fi
}
# Task 1: reconstruction error vs number of query modes k, tree vs Tucker-ALS on the same boxes.
#         All four data sets write into one folder, so results/errk/fig_error_vs_k.png shows all of them.
run errk_Traffic    experiments2.py --task error_k --datasets Traffic                --out results/errk --threads 4
run errk_USStock    experiments2.py --task error_k --datasets USStock                --out results/errk --threads 4
run errk_KRStock    experiments2.py --task error_k --datasets KRStock    --T 2048    --out results/errk --threads 4
run errk_AirQuality experiments2.py --task error_k --datasets AirQuality --T 2048    --out results/errk --threads 4
# Task 2: query-time complexity + baselines (TUCKET + slicing, Tucker-ALS) on the same queries.
#         Traffic already done: results/grid-20261020-162526
run grid_USStock    experiments2.py --datasets USStock                --threads 4
run grid_KRStock    experiments2.py --datasets KRStock    --T 2048    --threads 4
run grid_AirQuality experiments2.py --datasets AirQuality --T 2048    --threads 4
echo "[$(date '+%m-%d %H:%M')] ALL DONE"
