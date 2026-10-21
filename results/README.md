# results/

Committed snapshot of the git-ignored `outputs/` folder, with every model checkpoint (`*.pt`) left out.
These are the saved files behind the numbers in `reports/`; see the main `README.md` §6 for what is where.

- Do not edit these files by hand. Rerun the script that writes the file (it writes to `outputs/`),
  then refresh this folder with `python scripts/snapshot_results.py`.
- The one checkpoint Task 3 needs, the Task 2 default encoder, is committed in `models/task2_default/`.
