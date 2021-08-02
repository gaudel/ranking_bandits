#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""""""

nb_iter = 10**3                        # total number of iterations
nb_runs = 2                             # number of runs per query
path_to_logs = "dev_null/logs"          # path to the directory to store the logs
path_to_curves = "dev_null/figures"     # path to the directory to store the curves
env_names = ["xxsmall", "big", "Yandex"]
                                        # name of tried settings

from exp import line_args_to_params, play, merge_records
from bandits_to_rank.referee import Referee
import json
import gzip
import os
import matplotlib.pyplot as plt

def run_alg(args1, args2):
    for env_name in env_names:
        if env_name == 'Yandex':
            for i in range(10):
                args = f'--play {nb_runs} {args1} --{env_name} {i} {args2}'
                print(args)
                play(line_args_to_params(args), verbose=False)
            merge_records(line_args_to_params(f'--merge {args1} --{env_name}_all {args2}'))
        else:
            args = f'--play {nb_runs} {args1} --{env_name} {args2}'
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


# --- play GRAB ---
args1 = f'{nb_iter} --shuffle_kappa'
args2 = f'--GRAB {nb_iter} 10 {path_to_logs}'
run_alg(args1, args2)

# --- play S-GRAB ---
args1 = f'{nb_iter} --shuffle_kappa'
args2 = f'--SGRAB {nb_iter} 35 {path_to_logs}'
run_alg(args1, args2)

# --- play KL-CombUCB1 ---
args1 = f'{nb_iter} --shuffle_kappa'
args2 = f'--KL-COMB {nb_iter} {path_to_logs}'
run_alg(args1, args2)

# --- play PB-MHB ---
args1 = f'{nb_iter} --shuffle_kappa_except_first'
args2 = f'--PB-MHB 1 --TGRW 1000 --vari_sigma {path_to_logs}'
run_alg(args1, args2)

# --- play epsilon-greedy ---
for c in [10**3, 10**4, 10**5]:
    args1 = f'{nb_iter} --shuffle_kappa'
    args2 = f'--eGreedy {c} 1 {path_to_logs}'
    run_alg(args1, args2)

# --- play TopRank ---
args1 = f'{nb_iter} --order_kappa'
args2 = f'--TopRank {nb_iter} --horizon_time_known {path_to_logs}'
run_alg(args1, args2)

# --- play PMED ---
args1 = f'{nb_iter} --order_kappa'
args2 = f'--PMED 1 10 0 {path_to_logs}'
run_alg(args1, args2)


# =====================================
# =====================================
# plot curves
# =====================================
# =====================================

for env_name in env_names:
    if env_name == 'Yandex':
        env_name =  f'{env_name}_all'
        nb_games = 10 * nb_runs
    else:
        env_name = f'purely_simulated__{env_name}'
        nb_games = nb_runs
    prefix = f'{path_to_logs}/{env_name}__'
    suffix = f'__games_{nb_iter}_nb_trials_1000_record_length_{nb_games}_games.gz'

    # --- load data ---
    refs = {}
    refs['GRAB'] = retrieve_data_from_zip(f'{prefix}shuffled_kappa__Bandit_GRAB_{nb_iter}_T_10_gamma{suffix}')
    refs['S-GRAB'] = retrieve_data_from_zip(f'{prefix}shuffled_kappa__Bandit_SGRAB_{nb_iter}_T_35_gamma{suffix}')
    refs['KL-CombUCB1'] = retrieve_data_from_zip(f'{prefix}shuffled_kappa__Bandit_KL-COMB_{nb_iter}_horizon{suffix}')

    refs[f'PB-MHB'] = retrieve_data_from_zip(f'{prefix}Bandit_PB-MHB_warm-up_start_1_step_TGRW_1000.0_c_vari_sigma_proposal{suffix}')
    if env_name == 'Yandex_all':
        refs['eGreedy'] = retrieve_data_from_zip(f'{prefix}shuffled_kappa__Bandit_EGreedy_SVD_10000.0_c_1_update{suffix}')
    elif env_name == 'purely_simulated__big':
        refs['eGreedy'] = retrieve_data_from_zip(f'{prefix}shuffled_kappa__Bandit_EGreedy_SVD_1000.0_c_1_update{suffix}')
    elif env_name == 'purely_simulated__xxsmall':
        refs['eGreedy'] = retrieve_data_from_zip(f'{prefix}shuffled_kappa__Bandit_EGreedy_SVD_100000.0_c_1_update{suffix}')

    refs['TopRank'] = retrieve_data_from_zip(f'{prefix}sorted_kappa__Bandit_TopRank_{float(nb_iter)}_delta_TimeHorizonKnown_{suffix}')
    refs[f'PMED'] = retrieve_data_from_zip(f'{prefix}sorted_kappa__Bandit_PMED_1.0_alpha_1_gap_MLE_0_gap_q{suffix}')


    # --- plot curve ---
    plt.figure(figsize=(8, 8))
    
    myplot(ref=refs['GRAB'], label='GRAB', color='red', linestyle='-')
    myplot(ref=refs['S-GRAB'], label='S-GRAB', color='pink', linestyle='-')
    myplot(ref=refs['KL-CombUCB1'], label='KL-CombUCB1', color='purple', linestyle='-')

    myplot(ref=refs['PB-MHB'], label='PB_MHB, c=$10^3$, m=1', color='grey', linestyle='-')
    if env_name == 'Yandex_all':
        myplot(ref=refs['eGreedy'], label='$\epsilon_n$-greedy, c=$10^4$', color='orange', linestyle='-')
    elif env_name == 'purely_simulated__big':
        myplot(ref=refs['eGreedy'], label='$\epsilon_n$-greedy, c=$10^3$', color='orange', linestyle='-')
    elif env_name == 'purely_simulated__xxsmall':
        myplot(ref=refs['eGreedy'], label='$\epsilon_n$-greedy, c=$10^5$', color='orange', linestyle='-')

    myplot(ref=refs['TopRank'], label='TopRank', color='brown', linestyle='--')
    myplot(ref=refs['PMED'], label='PMED', color='green', linestyle='--')

    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Expected Regret')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.savefig(f'{path_to_curves}/{env_name}.pdf', bbox_inches='tight', pad_inches=0)