#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""

Manage Epsilon-greedy experiments.

Usage:
  exp.py --play <nb_game> [-s <start_game>] <nb_trials>  [-r <record_len>]
            (--KDD <query> |--Yandex <query> | --std | --small | --big | --xsmall | --xxsmall)
            (--eGreedy <c> <maj> [--noSVD]
                | --PBM-TS [--oracle]
                | --PBM-PIE <epsilon> [--oracle]
                | --PBM-UCB <epsilon> [--oracle]
                | --BC-MPTS [--oracle]
                | --PB-MHB <nb_steps> (--TGRW <c> [--vari_sigma]|--LGRW <c> [--vari_sigma]|--RR <c> <str_proposal_possible> [--vari_sigma]|--MaxPos|--PseudoView ) [--random_start]
                | --PMED <alpha> <gap_MLE> <gap_q>
                | --BubbleRank <delta> [--sorted] [--oracle]
                | --TopRank <T> [--horizon_time_known] [--doubling_trick] [--sorted] [--oracle]
            )
            (<output_path> [--force] [--nb_checkpoints <nb_checkpoints>])
  exp.py --merge <nb_trials>  [-r <record_len>]
            (--KDD_all | --KDD <query> | --Yandex_all | --Yandex <query> | --std | --small | --big | --xsmall | --xxsmall)
            (--eGreedy <c> <maj> [--noSVD]
                | --PBM-TS [--oracle]
                | --PBM-PIE <epsilon> [--oracle]
                | --PBM-UCB <epsilon> [--oracle]
                | --BC-MPTS [--oracle]
                | --PB-MHB <nb_steps> (--TGRW <c> [--vari_sigma]|--LGRW <c> [--vari_sigma]|--RR <c>  <str_proposal_possible> [--vari_sigma]|--MaxPos|--PseudoView ) [--random_start]
                | --PMED <alpha> <gap_MLE> <gap_q>
                | --BubbleRank <delta> [--sorted] [--oracle]
                | --TopRank <T> [--horizon_time_known] [--doubling_trick] [--sorted] [--oracle]
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
"""


from param import Parameters, record_zip

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
    else:
        raise ValueError("unknown environment")

    #### Init player
    if args['--eGreedy']:
        params.set_player_eGreedy(float(args['<c>']), int(args['<maj>']), args['--noSVD'])
    elif args['--PBM-TS']:
        params.set_player_PBM_TS(args['--oracle'])
    elif args['--PBM-PIE']:
        params.set_player_PBM_PIE(float(args['<epsilon>']), int(args['<nb_trials>']),args['--oracle'])
    elif args['--PBM-UCB']:
        params.set_player_PBM_UCB(float(args['<epsilon>']), args['--oracle'])
    elif args['--BC-MPTS']:
        params.set_player_BC_MPTS(args['--oracle'])
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
        params.set_player_PMED(float(args['<alpha>']),int(args['<gap_MLE>']),int(args['<gap_q>']))
    elif args['--BubbleRank']:  # --BubbleRank <delta> [--sorted] [--oracle]
        params.set_player_BubbleRank(float(args['<delta>']), sorted=args['--sorted'], oracle=args['--oracle'])
    elif args['--TopRank']:  # --TopRank [--sorted] [--oracle]
        params.set_player_TopRank(float(args['<T>']), horizon_time_known=args['--horizon_time_known'], doubling_trick=args['--doubling_trick'], sorted=args['--sorted'], oracle=args['--oracle'])
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
                    params.env.shuffle(fixed_kappa=not params.shuffle_kappa)
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
