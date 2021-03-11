# ranking_bandits
Bandit algorithms choosing a ranked-list of $L$ items among $K$ at each iteration

> This contains the code used for the research arcticle [Bandit Algorithm for Both Unknown BestPosition and Best Item Display on Web Pages]() (Camille-Sovaneary Gauthier, Romaric Gaudel, and Elisa Fromont, 2021) presented at the 19$^{th}$ Symposium on Intelligent Data Analysis ([IDA'21](https://ida2021.org/)).

# Requirements

* Python >= 3.6
* libraries (see `requirements.txt` build with Python 3.8.7)
    * numpy
    * scipy
    * (for PMED) tensorflow 
    * (for `epx.py` and `run_experiments.py`) docopt
    * (for `run_experiments.py`) matplotlib

# Run all the experiments and generate the curves

All the experiments may be ran altogether by executing `run_experiments.py`, with `bandits_to_rank`, our version of `pyclick`, and `data` in the path. Feel free to change in the head of the file

* the total number of iterations,
* the number of runs per query,
* the path to the directory to store the curves.

> WARNING: long run on the horizon


# Run selected experiments

The experiments are run thanks to `exp.py` script. It serves both

* to run experiments, which generates one log_file per run,
* to merge log files.

A typical usages would be
```bash
python3 exp.py --play 5 10000 --small --PB-MHB 1 --TGRW 1000. PATH_TO_RESULTS_DIR
python3 exp.py --merge 10000 --small --PB-MHB 1 --TGRW 1000. PATH_TO_RESULTS_DIR
``` 
with `bandits_to_rank`, our version of `pyclick`, and `data` in the path. See `python3 exp.py -h` for a detailed list of parameters.

Note that the runs may be run on different nodes before merging the results, like so:
```bash
python3 exp.py --play 1 -s 0 10000 --small --PB-MHB 1 --TGRW 1000. PATH_TO_RESULTS_DIR
python3 exp.py --play 1 -s 1 10000 --small --PB-MHB 1 --TGRW 1000. PATH_TO_RESULTS_DIR
...
python3 exp.py --play 1 -s 4 10000 --small --PB-MHB 1 --TGRW 1000. PATH_TO_RESULTS_DIR
python3 exp.py --merge 10000 --small --PB-MHB 1 --TGRW 1000. PATH_TO_RESULTS_DIR
``` 

Finally, for Yandex and KDD, the query number has to be chosen when running experiments, but the merge could be done on the whole queries at once.
```bash
python3 exp.py --play 5 10000 --KDD 0 --PB-MHB 1 --TGRW 1000. PATH_TO_RESULTS_DIR
python3 exp.py --play 5 10000 --KDD 1 --PB-MHB 1 --TGRW 1000. PATH_TO_RESULTS_DIR
...
python3 exp.py --play 5 10000 --KDD 7 --PB-MHB 1 --TGRW 1000. PATH_TO_RESULTS_DIR
python3 exp.py --merge 10000 --KDD_all --PB-MHB 1 --TGRW 1000. PATH_TO_RESULTS_DIR
``` 

See `python3 exp.py -h` for a detailed list of parameters.


# Generate curves
The log file obtained after merging is a gziped json file. It is self contained, but you may also load it in a `Referee` object to get some helper functions. Here is a minimal example:

```Python
from bandits_to_rank.referee import Referee
import json
import gzip
import matplotlib.pyplot as plt

# --- load data ---
referee = Referee (None, -1, all_time_record=True)
file_name = 'PATH_TO_RESULTS_DIR/KDD_all__Bandit_PB-MHB_warm-up_start_1_step_TGRW_1000.0_c_vari_sigma_proposal__games_10000_nb_trials_1000_record_length_160_games.gz'
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

