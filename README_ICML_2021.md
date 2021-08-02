<!-- Our title -->
<div align="center">
  <h3>(Re)run experiments of GRAB</h3>
</div>

<!-- Short description -->
> How to run the experiments in
> * [Parametric Graph for Unimodal Ranking Bandit]() (Camille-Sovaneary Gauthier, Romaric Gaudel, Elisa Fromont, and Boammani Aser Lompo,  2021) presented at the 38$^{th}$ International Conference on Machine Learning ([ICML'21](https://icml.cc/Conferences/2021)). 


<!-- Draw horizontal rule -->
<hr>

<!-- Table of content -->

| Section | Description |
|-|-|
| [Requirements](#requirements) | How to get the same version as in the original paper |
| [Run all the experiments and generate the curves](#run-all-the-experiments-and-generate-the-curves) |  |
| [Run selected experiments](#run-selected-experiments) |  |
| [Generate curves](#generate-curves) |  |





## Requirements

To use the same version as in the original paper, checkout Git's tag `ICML_2021`.

## Run all the experiments and generate the curves

All the experiments may be ran altogether by executing `run_experiments.py`, with `bandits_to_rank`, our version of `pyclick`, and `data` in the path. Feel free to change in the head of the file

* the total number of iterations,
* the number of runs per query,
* the full path to the directory to store the curves.

> WARNING: long run on the horizon


## Run selected experiments

The experiments are run thanks to `exp.py` script. It serves both

* to run experiments, which generates one log_file per run,
* to merge log files.

A typical usages would be
```bash
python3 exp.py --play 5 10000 --small --GRAB 10000 0 FULL_PATH_TO_RESULTS_DIR
python3 exp.py --merge 10000 --small --GRAB 10000 0 FULL_PATH_TO_RESULTS_DIR
``` 
with `bandits_to_rank`, our version of `pyclick`, and `data` in the path. See `python3 exp.py -h` for a detailed list of parameters.

Note that the runs may be run on different nodes before merging the results, like so:
```bash
python3 exp.py --play 1 -s 0 10000 --small --GRAB 10000 0 FULL_PATH_TO_RESULTS_DIR
python3 exp.py --play 1 -s 1 10000 --small --GRAB 10000 0 FULL_PATH_TO_RESULTS_DIR
...
python3 exp.py --play 1 -s 4 10000 --small --GRAB 10000 0 FULL_PATH_TO_RESULTS_DIR
python3 exp.py --merge 10000 --small --GRAB 10000 0 FULL_PATH_TO_RESULTS_DIR
``` 

Finally, for Yandex and KDD, the query number has to be chosen when running experiments, but the merge could be done on the whole queries at once.
```bash
python3 exp.py --play 5 10000 --Yandex 0 --GRAB 10000 0 FULL_PATH_TO_RESULTS_DIR
python3 exp.py --play 5 10000 --Yandex 1 --GRAB 10000 0 FULL_PATH_TO_RESULTS_DIR
...
python3 exp.py --play 5 10000 --Yandex 7 --GRAB 10000 0  FULL_PATH_TO_RESULTS_DIR
python3 exp.py --merge 10000 --Yandex_all --GRAB 10000 0  FULL_PATH_TO_RESULTS_DIR
```

See `python3 exp.py -h` for a detailed list of parameters.


## Generate curves
The log file obtained after merging is a gziped json file. It is self contained, but you may also load it in a `Referee` object to get some helper functions. Here is a minimal example:

```Python
from bandits_to_rank.referee import Referee
import json
import gzip
import matplotlib.pyplot as plt

# --- load data ---
referee = Referee (None, -1, all_time_record=True)
file_name = 'PATH_TO_RESULTS_DIR/purely_simulated__small__shuffled_kappa__Bandit_GRAB_10000_T_5_gamma__games_10000_nb_trials_1000_record_length_5_games.gz'
with gzip.GzipFile(file_name, 'r') as fin:
    json_bytes = fin.read()
    json_str = json_bytes.decode('utf-8')
    referee.record_results = json.loads(json_str)  

# --- plot curve ---
trials = referee.get_recorded_trials()
mu, _, _ = referee.get_regret_expected()
plt.plot(trials, mu)
plt.show()
``` 

