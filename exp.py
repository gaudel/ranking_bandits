#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""

Manage Epsilon-greedy experiments.

Usage:
  exp.py --play <nb_game> [-s <start_game>] <nb_trials>  [-r <record_len>]
            ( | [--order_kappa] | [--shuffle_kappa] | [--shuffle_kappa_except_first] | [--increasing_kappa] | [--increasing_kappa_except_first])
            (--KDD <query>
            |--Yandex <query>|--Yandex_equi <query> <K>
            | --std | --small | --big | --xsmall | --xxsmall | --test
            |--Yandex_CM <query> <nb_position> <nb_item>
            |--std_CM | --small_CM | --big_CM | --xsmall_CM | --xxsmall_CM | --test_CM)
            (--eGreedy <c> <maj> [--noSVD]
                | --PBM-TS [--oracle|--greedyMLE|--greedySVD]
                | --PBM-PIE <epsilon> [--oracle|--greedyMLE|--greedySVD]
                | --PBM-UCB <epsilon> [--oracle|--greedyMLE|--greedySVD]
                | --BC-MPTS [--oracle|--greedyMLE|--greedySVD]
                | --PB-MHB <nb_steps> (--TGRW <c> [--vari_sigma]|--LGRW <c> [--vari_sigma]|--RR <c> <str_proposal_possible> [--vari_sigma]|--MaxPos|--PseudoView ) [--random_start]
                | --PMED <alpha> <gap_MLE> <gap_q>
                | --TopRank <T> [--horizon_time_known] [--doubling_trick] [--oracle]
                | --SGRAB <T>  <gamma> [--force]
                | --GRAB <T> <gamma> [--force]
                | --Comb-UCB1
                | --KL-COMB <T>
            )
            (<output_path> [--force] [--nb_checkpoints <nb_checkpoints>])
  exp.py --merge <nb_trials>  [-r <record_len>]
            ( | [--order_kappa] | [--shuffle_kappa] | [--shuffle_kappa_except_first] | [--increasing_kappa] | [--increasing_kappa_except_first])
            (--KDD_all | --KDD <query>
            | --Yandex_all| --Yandex <query> | --Yandex_equi_all <K> | --Yandex_equi <query> <K>
            | --std | --small | --big | --xsmall | --xxsmall | --test
            | --Yandex_CM_all <nb_position> <nb_item> | --Yandex_CM <query> <nb_position> <nb_item>
            |--std_CM | --small_CM | --big_CM | --xsmall_CM | --xxsmall_CM | --test_CM)
            (--eGreedy <c> <maj> [--noSVD]
                | --PBM-TS [--oracle|--greedyMLE|--greedySVD]
                | --PBM-PIE <epsilon> [--oracle|--greedyMLE|--greedySVD]
                | --PBM-UCB <epsilon> [--oracle|--greedyMLE|--greedySVD]
                | --BC-MPTS [--oracle|--greedyMLE|--greedySVD]
                | --PB-MHB <nb_steps> (--TGRW <c> [--vari_sigma]|--LGRW <c> [--vari_sigma]|--RR <c>  <str_proposal_possible> [--vari_sigma]|--MaxPos|--PseudoView ) [--random_start]
                | --PMED <alpha> <gap_MLE> <gap_q>
                | --TopRank <T> [--horizon_time_known] [--doubling_trick] [--oracle]
                | --SGRAB <T>  <gamma> [--force]
                | --GRAB <T> <gamma> [--force]
                | --Comb-UCB1
                | --KL-COMB <T>
            )
            (<input_path> [<output_path>])
  exp.py (-h | --help)

Options:
  -h --help         Show this screen
  -r <record_len>   Number of recorded trials [default: 1000]
  -s <start_game>   Run games from <start_game> to <start_game> + <nb_games> - 1 [default: 0]
  <output_path>     Where to put the merged file [default: <input_path>]
                    ! WARNING ! has to be relative wrt. $SCRATCHDIR or absolute
  -KDD_all          Round-robin on KDD queries
  --nb_checkpoints <nb_checkpoints>     [default: 0]
  --known_horizon <T>   [default: -1]
"""


from param import Parameters, record_zip
from bandits_to_rank.environment import PositionsRanking

import os
from glob import glob
import json
import gzip
from docopt import docopt
import pickle
import time
from shutil import move

# --- Useful ---
def retrieve_data_from_zip(file_name):
    with gzip.GzipFile(file_name, 'r') as fin:
        json_bytes = fin.read()

    json_str = json_bytes.decode('utf-8')            # 2. string (i.e. JSON)
    return json.loads(json_str)


# --- Functions ---
def args_to_params(args):
    #### Init parameters
    params = Parameters()

    #### Init environment
    if args['--KDD']:
        params.set_env_KDD(int(args['<query>']))
    elif args['--KDD_all']:
        params.set_env_KDD_all()
    elif args['--Yandex']:
        params.set_env_Yandex(int(args['<query>']))
    elif args['--Yandex_all']:
        params.set_env_Yandex_all()
    elif args['--Yandex_equi']:
        params.set_env_Yandex_equi(query=int(args['<query>']),K=int(args['<K>']))
    elif args['--Yandex_equi_all']:
        params.set_env_Yandex_equi_all(K=int(args['<K>']))
    elif args['--test']:
        params.set_env_test()
    elif args['--std']:
        params.set_env_std()
    elif args['--small']:
        params.set_env_small()
    elif args['--xsmall']:
        params.set_env_extra_small()
    elif args['--xxsmall']:
        params.set_env_xx_small()
    elif args['--big']:
        params.set_env_big()
    elif args['--Yandex_CM']:
        params.set_env_Yandex_CM(query=int(args['<query>']))
    elif args['--Yandex_CM_all']:
        params.set_env_Yandex_CM_all()
    elif args['--test_CM']:
        params.set_env_test_CM()
    elif args['--std_CM']:
        params.set_env_std_CM()
    elif args['--small_CM']:
        params.set_env_small_CM()
    elif args['--xsmall_CM']:
        params.set_env_extra_small_CM()
    elif args['--xxsmall_CM']:
        params.set_env_xx_small_CM()
    elif args['--big_CM']:
        params.set_env_big_CM()
    else:
        raise ValueError("unknown environment")

    #### Init environment shuffling
    if args['--order_kappa']:
        params.set_positions_ranking(PositionsRanking.DECREASING)
    elif args['--shuffle_kappa']:
        params.set_positions_ranking(PositionsRanking.SHUFFLE)
    elif args['--shuffle_kappa_except_first']:
        params.set_positions_ranking(PositionsRanking.SHUFFLE_EXCEPT_FIRST)
    elif args['--increasing_kappa']:
        params.set_positions_ranking(PositionsRanking.INCREASING)
    elif args['--increasing_kappa_except_first']:
        params.set_positions_ranking(PositionsRanking.INCREASING_EXCEPT_FIRST)
    else:  # default
        params.set_positions_ranking(PositionsRanking.SHUFFLE)

    #### Init player
    if args['--eGreedy']:
        params.set_player_eGreedy(float(args['<c>']), int(args['<maj>']), args['--noSVD'])
    elif args['--PBM-TS']:
        if args['--oracle']:
            type = "oracle"
        elif args['--greedyMLE']:
            type = "greedyMLE"
        elif args['--greedySVD']:
            type = "greedySVD"
        else:
            type = "greedySVD"
        params.set_player_PBM_TS(type=type)
    elif args['--PBM-PIE']:
        if args['--oracle']:
            type = "oracle"
        elif args['--greedyMLE']:
            type = "greedyMLE"
        elif args['--greedySVD']:
            type = "greedySVD"
        else:
            type = "greedySVD"
        params.set_player_PBM_PIE(float(epsilon=args['<epsilon>']), T=int(args['<nb_trials>']),type = type)
    elif args['--PBM-UCB']:
        if args['--oracle']:
            type = "oracle"
        elif args['--greedyMLE']:
            type = "greedyMLE"
        elif args['--greedySVD']:
            type = "greedySVD"
        else:
            type = "greedySVD"
        params.set_player_PBM_UCB(epsilon=float(args['<epsilon>']), type=type)
    elif args['--BC-MPTS']:
        if args['--oracle']:
            type = "oracle"
        elif args['--greedyMLE']:
            type = "greedyMLE"
        elif args['--greedySVD']:
            type = "greedySVD"
        else:
            type = "greedySVD"
        params.set_player_BC_MPTS(type)
    elif args['--PB-MHB']:  # --PB-MHB <nb_steps> <c> [--random_start]
        if args['--TGRW']:
            params.set_proposal_TGRW(float(args['<c>']), args['--vari_sigma'])
        elif args['--LGRW']:
            params.set_proposal_LGRW(float(args['<c>']), args['--vari_sigma'])
        elif args['--RR']:
            params.set_proposal_RR(float(args['<c>']),args['<str_proposal_possible>'], args['--vari_sigma'])
        elif args['--MaxPos']:
            params.set_proposal_MaxPos()
        elif args['--PseudoView']:
            params.set_proposal_PseudoView()
        params.set_player_PB_MHB(int(args['<nb_steps>']),  args['--random_start'])
    elif args['--PMED']:  # --BubbleRank <delta> [--sorted] [--oracle]
        params.set_player_PMED(float(args['<alpha>']),int(args['<gap_MLE>']),int(args['<gap_q>']), run=args['--play'])
    elif args['--TopRank']:  # --TopRank [--sorted] [--oracle]
        params.set_player_TopRank(float(args['<T>']), horizon_time_known=args['--horizon_time_known'], doubling_trick=args['--doubling_trick'], oracle=args['--oracle'])
    elif args['--SGRAB']:
        params.set_player_SGRAB(int(args['<T>']), gamma=int(args['<gamma>']), forced_initiation=args['--force'])
    elif args['--GRAB']:
        params.set_player_GRAB(T=int(args['<T>']), gamma=int(args['<gamma>']), forced_initiation=args['--force'])
    elif args['--Comb-UCB1']:
        params.set_player_CombUCB1()
    elif args['--KL-COMB']:
        params.set_player_KL_COMB(horizon=int(args['<T>']))
    else:
        raise ValueError("unknown player")

    #### Set rules
    params.set_rules(int(args['<nb_trials>']), nb_records=int(args['-r']))

    #### Set experiment
    if args['<nb_game>'] is None:
        args['<nb_game>'] = -1
    params.set_exp(first_game=int(args['-s']), nb_games=int(args['<nb_game>']), nb_checkpoints=int(args['--nb_checkpoints']),
                   input_path=args['<input_path>'], output_path=args['<output_path>'],
                   force=args['--force']
                   )

    return params


def line_args_to_params(line_args):
    try:
        args = docopt(__doc__, argv=line_args.split())
        return args_to_params(args)
    except:
        print(f'Error bad arguments: {line_args}\nSplit as: {line_args.split()}\nWhile expecting:')
        raise


def play(params, dry_run=False, verbose=True):
    """
    # Parameters
        params
        dry_run : bool
            if True, do not play games, only count the number of games to be played

    # Returns
        nb_played_games : int
    """
    nb_trials_between_check_points = params.referee.nb_trials // (params.nb_checkpoints+1)
    nb_played_games = 0
    for id_g in range(params.first_game, params.end_game):
        if verbose:
            print('#### GAME '+str(id_g))
        base_output_file_name = f'{params.output_path}/{params.env_name}__{params.player_name}__{params.rules_name}_{id_g}_game_id'
        output_file_name = f'{base_output_file_name}.gz'
        if os.path.exists(output_file_name) and not params.force:
            if verbose:
                print('File', output_file_name, 'already exists. Keep it.')
        else:
            nb_played_games += 1
            if verbose and os.path.exists(output_file_name):
                print('File', output_file_name, 'already exists. Replace with a new one.')
            if not dry_run:
                # --- play one game, in multiple chunks ---

                if not params.force and (os.path.isfile(f'{base_output_file_name}.ckpt.pickle.gz') or os.path.isfile(f'{base_output_file_name}.ckpt.pickle.old')):
                    # load save game state ...
                    if verbose:
                        start_time = time.time()
                        print('--- loading game state ---')
                    if os.path.isfile(f'{base_output_file_name}.ckpt.pickle.gz'):
                        saved_params = pickle.load(gzip.GzipFile(f'{base_output_file_name}.ckpt.pickle.gz', 'rb'))
                    else:
                        saved_params = pickle.load(gzip.GzipFile(f'{base_output_file_name}.ckpt.pickle.old', 'rb'))
                    if verbose:
                        print(f'--- done in {time.time()-start_time} sec ---')
                    params.env = saved_params.env
                    params.player = saved_params.player
                    params.referee = saved_params.referee
                else:
                    # ... or init and play first trial
                    params.env.shuffle(positions_ranking=params.positions_ranking)
                    params.player.clean()
                    params.referee.clean_recorded_results()
                    params.referee.prepare_new_game()

                # play the remaining game
                while params.referee.running_t < params.referee.nb_trials-1:
                    params.referee.play_game(params.player, new_game=False, nb_trials_before_break=nb_trials_between_check_points, nb_relevant_positions=params.nb_relevant_positions)

                    # save game state
                    if params.referee.running_t < params.referee.nb_trials-1:
                        if verbose:
                            start_time = time.time()
                            print('--- saving game state ---')
                        pickle.dump(params, gzip.GzipFile(f'{base_output_file_name}.ckpt.pickle.tmp', 'wb'))
                        if verbose:
                            print(f'--- done in {time.time() - start_time} sec ---')
                        if os.path.exists(f'{base_output_file_name}.ckpt.pickle.gz'):
                            #os.remove(f'{base_output_file_name}.ckpt.pickle.old')
                            move(f'{base_output_file_name}.ckpt.pickle.gz', f'{base_output_file_name}.ckpt.pickle.old')
                            #os.rename(f'{base_output_file_name}.ckpt.pickle.gz', f'{base_output_file_name}.ckpt.pickle.old')
                        #os.remove(f'{base_output_file_name}.ckpt.pickle.gz')
                        move(f'{base_output_file_name}.ckpt.pickle.tmp', f'{base_output_file_name}.ckpt.pickle.gz')
                        #os.rename(f'{base_output_file_name}.ckpt.pickle.tmp', f'{base_output_file_name}.ckpt.pickle.gz')

                # save the results
                record_zip(output_file_name, params.referee.record_results)
                for ext in ['tmp', 'old', 'gz']:
                    file = f'{base_output_file_name}.ckpt.pickle.{ext}'
                    if os.path.isfile(file):
                        os.remove(file)

    return nb_played_games


def merge_records(params, verbose=False):
    # Merge
    params.referee.clean_recorded_results()
    nb_games = 0
    logs_env_name = params.logs_env_name
    for file_name in glob(f'{params.input_path}/{logs_env_name}__{params.player_name}__{params.rules_name}_*_game_id.gz'):

        if verbose:
            print('#### Read ' + os.path.basename(file_name))
        record = retrieve_data_from_zip(file_name)
        params.referee.add_record(record)
        nb_games += 1
        print(nb_games)

    # Save results
    if nb_games != 0:
        record_zip(f'{params.output_path}/{params.env_name}__{params.player_name}__{params.rules_name}_{nb_games}_games.gz', params.referee.record_results)


if __name__ == "__main__":
    import sys
    print(sys.argv)
    arguments = docopt(__doc__)
    print(arguments)
    params = args_to_params(arguments)

    if arguments['--play']:
        play(params)
    elif arguments['--merge']:
        merge_records(params)
