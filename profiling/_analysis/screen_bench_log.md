# Screen CLI bench log

Timings are mean wall seconds over N timed runs after one warmup (see `profiling/scripts/bench_screen.py`).

## baseline

- git: `5d578f6adb87acb99165de77ec5eb55fc1f00ae4`
- when: 2026-08-10T06:20:24.666935+00:00
- turso_configured: True; pandas=3.0.5; numpy=2.4.6
- runs: 3

| metric | mean_s | samples | delta_vs_prev |
| --- | ---: | --- | ---: |
| M1_cli_warm_csv | 1.080 | 1.080, 1.070, 1.090 | - |
| M2_cli_warm_table | 1.357 | 1.360, 1.350, 1.360 | - |
| M3_cli_help | 0.627 | 0.620, 0.640, 0.620 | - |
| M4_workflow_warm_csv | 0.009 | 0.010, 0.009, 0.009 | - |
| M5_workflow_warm_full | 0.137 | 0.168, 0.126, 0.117 | - |
| M6_import_cli | 0.633 | 0.610, 0.620, 0.670 | - |
| M7_usage_pair | 0.415 | 0.412, 0.388, 0.447 | - |
| M8_cli_warm_csv_no_turso | 0.640 | 0.630, 0.660, 0.630 | - |
## after_c1

- git: `5d578f6adb87acb99165de77ec5eb55fc1f00ae4`
- when: 2026-08-10T06:40:12.781490+00:00
- turso_configured: True; pandas=3.0.5; numpy=2.4.6
- runs: 3

| metric | mean_s | samples | delta_vs_prev |
| --- | ---: | --- | ---: |
| M1_cli_warm_csv | 1.003 | 1.010, 1.000, 1.000 | -7.1% |
| M2_cli_warm_table | 1.530 | 1.570, 1.480, 1.540 | +12.8% |
| M3_cli_help | 0.630 | 0.640, 0.610, 0.640 | +0.5% |
| M4_workflow_warm_csv | 0.011 | 0.012, 0.010, 0.011 | +18.0% |
| M5_workflow_warm_full | 0.136 | 0.161, 0.128, 0.120 | -0.5% |
| M6_import_cli | 0.647 | 0.640, 0.630, 0.670 | +2.1% |
| M7_usage_pair | 0.232 | 0.203, 0.245, 0.247 | -44.2% |
| M8_cli_warm_csv_no_turso | 0.667 | 0.650, 0.680, 0.670 | +4.2% |
## after_c2

- git: `ee0b338a711672bff6b8a71b2e68399edf7a95fa`
- when: 2026-08-10T06:43:06.027858+00:00
- turso_configured: True; pandas=3.0.5; numpy=2.4.6
- runs: 3

| metric | mean_s | samples | delta_vs_prev |
| --- | ---: | --- | ---: |
| M1_cli_warm_csv | 0.780 | 0.780, 0.780, 0.780 | -22.3% |
| M2_cli_warm_table | 1.057 | 1.040, 1.060, 1.070 | -30.9% |
| M3_cli_help | 0.637 | 0.620, 0.630, 0.660 | +1.1% |
| M4_workflow_warm_csv | 0.009 | 0.009, 0.009, 0.009 | -18.3% |
| M5_workflow_warm_full | 0.140 | 0.168, 0.120, 0.133 | +3.0% |
| M6_import_cli | 0.637 | 0.640, 0.640, 0.630 | -1.5% |
| M7_usage_pair | 0.100 | 0.100, 0.100, 0.100 | -56.7% |
| M8_cli_warm_csv_no_turso | 0.657 | 0.650, 0.650, 0.670 | -1.5% |
## after_c3

- git: `09feea22261b6df7a29428027856e802cd70279c`
- when: 2026-08-10T06:47:42.097446+00:00
- turso_configured: True; pandas=3.0.5; numpy=2.4.6
- runs: 3

| metric | mean_s | samples | delta_vs_prev |
| --- | ---: | --- | ---: |
| M1_cli_warm_csv | 0.620 | 0.630, 0.610, 0.620 | -20.5% |
| M2_cli_warm_table | 0.893 | 0.890, 0.900, 0.890 | -15.5% |
| M3_cli_help | 0.143 | 0.150, 0.140, 0.140 | -77.5% |
| M4_workflow_warm_csv | 0.009 | 0.009, 0.009, 0.009 | +0.6% |
| M5_workflow_warm_full | 0.130 | 0.149, 0.120, 0.119 | -7.7% |
| M6_import_cli | 0.140 | 0.140, 0.140, 0.140 | -78.0% |
| M7_usage_pair | 0.100 | 0.100, 0.100, 0.100 | -0.1% |
| M8_cli_warm_csv_no_turso | 0.510 | 0.520, 0.510, 0.500 | -22.3% |
## after_c4

- git: `34352f77e35af3e1f862c872b303a6e50783992d`
- when: 2026-08-10T06:48:52.143028+00:00
- turso_configured: True; pandas=3.0.5; numpy=2.4.6
- runs: 3

| metric | mean_s | samples | delta_vs_prev |
| --- | ---: | --- | ---: |
| M1_cli_warm_csv | 0.623 | 0.650, 0.610, 0.610 | +0.5% |
| M2_cli_warm_table | 0.703 | 0.710, 0.710, 0.690 | -21.3% |
| M3_cli_help | 0.143 | 0.140, 0.150, 0.140 | +0.0% |
| M4_workflow_warm_csv | 0.009 | 0.009, 0.009, 0.008 | -4.9% |
| M5_workflow_warm_full | 0.049 | 0.051, 0.045, 0.052 | -62.0% |
| M6_import_cli | 0.140 | 0.140, 0.140, 0.140 | +0.0% |
| M7_usage_pair | 0.100 | 0.100, 0.100, 0.100 | -0.1% |
| M8_cli_warm_csv_no_turso | 0.513 | 0.510, 0.520, 0.510 | +0.7% |
## after_c5

- git: `327d6aee16dedb1b79927f694f453d727ed9ea58`
- when: 2026-08-10T06:50:15.393240+00:00
- turso_configured: True; pandas=3.0.5; numpy=2.4.6
- runs: 3

| metric | mean_s | samples | delta_vs_prev |
| --- | ---: | --- | ---: |
| M1_cli_warm_csv | 0.533 | 0.540, 0.530, 0.530 | -14.4% |
| M2_cli_warm_table | 0.610 | 0.610, 0.620, 0.600 | -13.3% |
| M3_cli_help | 0.140 | 0.140, 0.140, 0.140 | -2.3% |
| M4_workflow_warm_csv | 0.009 | 0.009, 0.009, 0.009 | +2.8% |
| M5_workflow_warm_full | 0.049 | 0.046, 0.045, 0.054 | -1.3% |
| M6_import_cli | 0.147 | 0.140, 0.150, 0.150 | +4.8% |
| M7_usage_pair | 0.100 | 0.100, 0.100, 0.100 | +0.1% |
| M8_cli_warm_csv_no_turso | 0.420 | 0.420, 0.420, 0.420 | -18.2% |
## final

- git: `1bc1700378e796ad4de0043f61f6de4fa9714b28`
- when: 2026-08-10T06:50:57.865256+00:00
- turso_configured: True; pandas=3.0.5; numpy=2.4.6
- runs: 3

| metric | mean_s | samples | delta_vs_prev |
| --- | ---: | --- | ---: |
| M1_cli_warm_csv | 0.550 | 0.550, 0.550, 0.550 | -49.1% |
| M2_cli_warm_table | 0.623 | 0.610, 0.620, 0.640 | -54.1% |
| M3_cli_help | 0.140 | 0.140, 0.140, 0.140 | -77.7% |
| M4_workflow_warm_csv | 0.009 | 0.009, 0.009, 0.009 | -3.1% |
| M5_workflow_warm_full | 0.044 | 0.045, 0.044, 0.043 | -67.8% |
| M6_import_cli | 0.143 | 0.150, 0.140, 0.140 | -77.4% |
| M7_usage_pair | 0.087 | 0.100, 0.100, 0.061 | -79.0% |
| M8_cli_warm_csv_no_turso | 0.433 | 0.440, 0.430, 0.430 | -32.3% |
## final

- git: `c9339edd193aaa72d11299c6c7299e18a5f320bf`
- when: 2026-08-10T06:52:28.944160+00:00
- turso_configured: True; pandas=3.0.5; numpy=2.4.6
- runs: 3

| metric | mean_s | samples | delta_vs_prev |
| --- | ---: | --- | ---: |
| M1_cli_warm_csv | 0.507 | 0.550, 0.480, 0.490 | -53.1% |
| M2_cli_warm_table | 0.573 | 0.580, 0.570, 0.570 | -57.7% |
| M3_cli_help | 0.140 | 0.140, 0.140, 0.140 | -77.7% |
| M4_workflow_warm_csv | 0.009 | 0.009, 0.009, 0.009 | -3.4% |
| M5_workflow_warm_full | 0.045 | 0.043, 0.048, 0.045 | -66.9% |
| M6_import_cli | 0.140 | 0.140, 0.140, 0.140 | -77.9% |
| M7_usage_pair | 0.050 | 0.050, 0.050, 0.050 | -87.9% |
| M8_cli_warm_csv_no_turso | 0.430 | 0.440, 0.420, 0.430 | -32.8% |

## Summary (baseline → final)

Machine: Turso via `.env`, pandas 3.0.5, numpy 2.4.6, thread limits = 1.
Means of 3 timed runs after warmup. See also `profiling/screen_perf_pr_results.md`.

| metric | baseline | final | improvement |
| --- | ---: | ---: | ---: |
| M1_cli_warm_csv | 1.080 | 0.507 | 53.1% |
| M2_cli_warm_table | 1.357 | 0.573 | 57.8% |
| M3_cli_help | 0.627 | 0.140 | 77.7% |
| M4_workflow_warm_csv | 0.009 | 0.009 | ~0% |
| M5_workflow_warm_full | 0.137 | 0.045 | 67.2% |
| M6_import_cli | 0.633 | 0.140 | 77.9% |
| M7_usage_pair | 0.415 | 0.050 | 87.9% |
| M8_cli_warm_csv_no_turso | 0.640 | 0.430 | 32.8% |

