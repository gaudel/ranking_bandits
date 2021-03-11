#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""""""

nb_iter = 10**3                        # total number of iterations
preliminary_nb_iter = 10**3           # total number of iterations for parameter-selection of PB-MHB
nb_runs = 2                             # number of runs per query
path_to_logs = "dev_null/logs"          # path to the directory to store the logs
path_to_curves = "dev_null/figures"     # path to the directory to store the curves
env_names = ["xxsmall", "big", "KDD", "Yandex"]
                                        # name of tried settings

from exp import line_args_to_params, play, merge_records
from bandits_to_rank.referee import Referee
import json
import gzip
import os
import matplotlib.pyplot as plt

def run_alg(args1, args2, no_checkpoints=True):
    for env_name in env_names:
        if env_name == 'Yandex':
            for i in range(10):
                args = f'--play {nb_runs} {args1} --{env_name} {i} {args2}{" --nb_checkpoints 0" if no_checkpoints else ""}'
                print(args)
                play(line_args_to_params(args), verbose=False)
            merge_records(line_args_to_params(f'--merge {args1} --{env_name}_all {args2}'))
        elif env_name == 'KDD':
            for i in range(8):
                args = f'--play {nb_runs} {args1} --{env_name} {i} {args2}{" --nb_checkpoints 0" if no_checkpoints else ""}'
                print(args)
                play(line_args_to_params(args), verbose=False)
            merge_records(line_args_to_params(f'--merge {args1} --{env_name}_all {args2}'))
        else:
            args = f'--play {nb_runs} {args1} --{env_name} {args2}{" --nb_checkpoints 0" if no_checkpoints else ""}'
            print(args)
            play(line_args_to_params(args), verbose=False)
            merge_records(line_args_to_params(f'--merge {args1} --{env_name} {args2}'))


def retrieve_data_from_zip(file_name):
    if os.path.isfile(file_name):
        with gzip.GzipFile(file_name, 'r') as fin:
            json_bytes = fin.read()
        json_str = json_bytes.decode('utf-8')  # 2. string (i.e. JSON)
        data = json.loads(json_str)
        referee_ob = Referee(None, -1, all_time_record=True)
        referee_ob.record_results = data
    else:
        print(f'!!! unknown file: {file_name}')
        referee_ob = None
    return referee_ob


def myplot(ref, label, color, linestyle):
    try:
        trials = ref.get_recorded_trials()
        mu, d_10, d_90 = ref.get_regret_expected()
        plt.plot(trials, mu, color=color, linestyle=linestyle, label=label)
        X_val, Y_val, Yerror = ref.barerror_value(type_errorbar='standart_error')
        neg_Yerror = [mu[i]-Yerror[i] for i in range(len(Yerror))]
        pos_Yerror = [mu[i]+Yerror[i] for i in range(len(Yerror))]

        plt.fill_between(X_val, neg_Yerror, pos_Yerror, color=color, alpha=0.3, linestyle=linestyle, label='')
    except:
        plt.plot([], [], color=color, linestyle=linestyle, label=label)


# =====================================
# =====================================
# run algorithms
# =====================================
# =====================================

# --- play PB-MHB (preliminary) ---
for m, c, random_start in [(1, 10**-1, False), (1, 10**0, False), (1, 10**1, False), (1, 10**2, False),
                           (1, 10**3, False), (10, 10**0, False), (10, 10**3, False), (1, 10**3, True)]:
    args1 = f'{preliminary_nb_iter}'
    args2 = f'--PB-MHB {m} --TGRW {c} --vari_sigma {path_to_logs} {" --random_start" if random_start else ""}'
    run_alg(args1, args2)

# --- play PB-MHB ---
for m, c, random_start in [(1, 10**3, False)]:
    args1 = f'{nb_iter}'
    args2 = f'--PB-MHB {m} --TGRW {c} --vari_sigma {path_to_logs} {" --random_start" if random_start else ""}'
    run_alg(args1, args2)

# --- play epsilon-greedy ---
for c in [10**0, 10**1, 10**2, 10**3, 10**4, 10**5, 10**6]:
    args1 = f'{nb_iter}'
    args2 = f'--eGreedy {c} 1 {path_to_logs}'
    run_alg(args1, args2, no_checkpoints=True)  # epsilon-greedy does not support checkpoints for the time being


# --- play TopRank ---
args1 = f'{nb_iter}'
args2 = f'--TopRank {nb_iter} --horizon_time_known --sorted {path_to_logs}'
run_alg(args1, args2)

# --- play PMED ---
args1 = f'{nb_iter}'
args2 = f'--PMED 1 10 0 {path_to_logs}'
run_alg(args1, args2)

# =====================================
# =====================================
# plot Fig 1.a curves
# =====================================
# =====================================

for env_name in env_names:
    if env_name == 'Yandex':
        env_name = f'{env_name}_all'
        nb_games = 10 * nb_runs
    elif env_name == 'KDD':
        env_name = f'{env_name}_all'
        nb_games = 8 * nb_runs
    else:
        env_name = f'purely_simulated__{env_name}'
        nb_games = nb_runs
    prefix = f'{path_to_logs}/{env_name}__'
    suffix = f'__games_{preliminary_nb_iter}_nb_trials_1000_record_length_{nb_games}_games.gz'

    # --- load data ---
    refs = {}
    for m, c in [(1, 10.**-1), (1, 10.**0), (1, 10.**1), (1, 10.**2), (1, 10.**3)]:
        refs[f'PB-MHB, m={m}, c={c}'] = retrieve_data_from_zip(f'{prefix}Bandit_PB-MHB_warm-up_start_{m}_step_TGRW_{c}_c_vari_sigma_proposal{suffix}')

    # --- plot curve ---
    plt.figure(figsize=(8, 8))

    for m, c, color in [(1, 10.**-1, "orange"), (1, 10.**0, 'blue'), (1, 10.**1, 'green'), (1, 10.**2, 'purple'), (1, 10.**3, 'red')]:
        key = f'PB-MHB, m={m}, c={c}'
        myplot(ref=refs[key], label=key, color=color, linestyle='-')

    plt.xlabel('Time-stamp')
    plt.ylabel('Cumulative Expected Regret')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(f'{path_to_curves}/{env_name}_impact_of_c.pdf', bbox_inches='tight', pad_inches=0)

# =====================================
# =====================================
# plot Fig 1.b curves
# =====================================
# =====================================

for env_name in env_names:
    if env_name == 'Yandex':
        env_name = f'{env_name}_all'
        nb_games = 10 * nb_runs
    elif env_name == 'KDD':
        env_name = f'{env_name}_all'
        nb_games = 8 * nb_runs
    else:
        env_name = f'purely_simulated__{env_name}'
        nb_games = nb_runs
    prefix = f'{path_to_logs}/{env_name}__'
    suffix = f'__games_{preliminary_nb_iter}_nb_trials_1000_record_length_{nb_games}_games.gz'

    # --- load data ---
    refs = {}
    for m, c, random_start in [(1, 10.**0, False), (10, 10.**0, False), (1, 10.**3, False), (10, 10.**3, False), (1, 10.**3, True)]:
        refs[f'PB-MHB, m={m}, c={c}{", rand. start" if random_start else ""}'] = retrieve_data_from_zip(f'{prefix}Bandit_PB-MHB_{"random" if random_start else "warm-up"}_start_{m}_step_TGRW_{c}_c_vari_sigma_proposal{suffix}')

    # --- plot curve ---
    plt.figure(figsize=(8, 8))

    for m, c, random_start, color, linestyle in [(1, 10.**0, False, 'blue', '-'), (10, 10.**0, False, 'blue', '--'),
                                                 (1, 10.**3, False, 'red', '-'), (10, 10.**3, False, 'red', '--'),
                                                 (1, 10.**3, True, 'red', ':')]:
        key = f'PB-MHB, m={m}, c={c}{", rand. start" if random_start else ""}'
        myplot(ref=refs[key], label=key, color=color, linestyle=linestyle)

    plt.xlabel('Time-stamp')
    plt.ylabel('Cumulative Expected Regret')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(f'{path_to_curves}/{env_name}_impact_of_m_and_random-start.pdf', bbox_inches='tight', pad_inches=0)

# =====================================
# =====================================
# plot Fig 2. curves
# =====================================
# =====================================

for env_name in env_names:
    if env_name == 'Yandex':
        env_name = f'{env_name}_all'
        nb_games = 10 * nb_runs
    elif env_name == 'KDD':
        env_name = f'{env_name}_all'
        nb_games = 8 * nb_runs
    else:
        env_name = f'purely_simulated__{env_name}'
        nb_games = nb_runs
    prefix = f'{path_to_logs}/{env_name}__'
    suffix = f'__games_{nb_iter}_nb_trials_1000_record_length_{nb_games}_games.gz'

    # --- load data ---
    refs = {}
    refs[f'PB-MHB'] = retrieve_data_from_zip(f'{prefix}Bandit_PB-MHB_warm-up_start_1_step_TGRW_1000.0_c_vari_sigma_proposal{suffix}')
    refs['eGreedy'] = retrieve_data_from_zip(f'{prefix}Bandit_EGreedy_SVD_10000.0_c_1_update{suffix}')
    refs['TopRank'] = retrieve_data_from_zip(f'{prefix}extended_kappas__sorted_kappa__Bandit_TopRank_{float(nb_iter)}_delta_TimeHorizonKnown_{suffix}')
    refs[f'PMED'] = retrieve_data_from_zip(f'{prefix}sorted_kappa__Bandit_PMED_1.0_alpha_1_gap_MLE_0_gap_q{suffix}')

    # --- plot curve ---
    plt.figure(figsize=(8, 8))
    
    myplot(ref=refs['eGreedy'], label='$\epsilon_n$-greedy, c=$10^4$', color='orange', linestyle='-')
    myplot(ref=refs['PB-MHB'], label='PB_MHB, c=$10^3$, m=1', color='red', linestyle='-')
    myplot(ref=refs['TopRank'], label='TopRank', color='green', linestyle='--')
    myplot(ref=refs['PMED'], label='PMED', color='purple', linestyle='--')

    plt.xlabel('Time-stamp')
    plt.ylabel('Cumulative Expected Regret')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(f'{path_to_curves}/{env_name}.pdf', bbox_inches='tight', pad_inches=0)