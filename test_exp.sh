#!/usr/bin/env bash

TESTDIR=`pwd`/test
mkdir $TESTDIR
mkdir $TESTDIR/dev_null
mkdir $TESTDIR/dev_null_bis

set -x
# basic usage
python3 exp.py --play 2 100 -r 10 --small --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 exp.py --play 2 -s 2 100 -r 10 --small --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 exp.py --merge 100 -r 10 --small --eGreedy 0.1 10 $TESTDIR/dev_null || exit
python3 exp.py --merge 100 -r 10 --small --eGreedy 0.1 10 $TESTDIR/dev_null $TESTDIR/dev_null_bis || exit

# KDD subtleties
python3 exp.py --play 2 100 -r 10 --KDD 0 --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 exp.py --play 2 100 -r 10 --KDD 1 --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 exp.py --merge 100 -r 10 --KDD 0 --eGreedy 0.1 10 $TESTDIR/dev_null || exit
python3 exp.py --merge 100 -r 10 --KDD_all --eGreedy 0.1 10 $TESTDIR/dev_null || exit

# Other datasets
python3 exp.py --play 2 100 -r 10 --Yandex 0 --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 exp.py --play 2 100 -r 10 --Yandex 1 --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 exp.py --merge 100 -r 10 --Yandex 0 --eGreedy 0.1 10 $TESTDIR/dev_null || exit
python3 exp.py --merge 100 -r 10 --Yandex_all --eGreedy 0.1 10 $TESTDIR/dev_null || exit

python3 exp.py --play 2 100 -r 10 --std --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 exp.py --merge 100 -r 10 --std --eGreedy 0.1 10 $TESTDIR/dev_null || exit

python3 exp.py --play 2 100 -r 10 --big --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 exp.py --merge 100 -r 10 --big --eGreedy 0.1 10 $TESTDIR/dev_null || exit

python3 exp.py --play 2 100 -r 10 --xsmall --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 exp.py --merge 100 -r 10 --xsmall --eGreedy 0.1 10 $TESTDIR/dev_null || exit

python3 exp.py --play 2 100 -r 10 --xxsmall --eGreedy 0.1 10 $TESTDIR/dev_null --force || exit
python3 exp.py --merge 100 -r 10 --xxsmall --eGreedy 0.1 10 $TESTDIR/dev_null || exit


# Other algorithms
python3 exp.py --play 2 100 -r 10 --small --eGreedy 0.1 10 --noSVD $TESTDIR/dev_null --force --nb_checkpoints 0  || exit # todo: does not support checkpoints for the time being
python3 exp.py --merge 100 -r 10 --small --eGreedy 0.1 10 --noSVD $TESTDIR/dev_null || exit

python3 exp.py --play 2 100 -r 10 --small --PBM-TS $TESTDIR/dev_null --force || exit
python3 exp.py --merge 100 -r 10 --small --PBM-TS $TESTDIR/dev_null || exit

python3 exp.py --play 2 100 -r 10 --small --PBM-TS --oracle $TESTDIR/dev_null --force || exit
python3 exp.py --merge 100 -r 10 --small --PBM-TS --oracle $TESTDIR/dev_null || exit

python3 exp.py --play 2 100 -r 10 --small --PBM-PIE 0.01 $TESTDIR/dev_null --force || exit
python3 exp.py --merge 100 -r 10 --small --PBM-PIE 0.01 $TESTDIR/dev_null || exit

python3 exp.py --play 2 100 -r 10 --small --PBM-PIE 0.01 --oracle $TESTDIR/dev_null --force || exit
python3 exp.py --merge 100 -r 10 --small --PBM-PIE 0.01 --oracle $TESTDIR/dev_null || exit

python3 exp.py --play 2 100 -r 10 --small --PBM-UCB 0.01 $TESTDIR/dev_null --force || exit
python3 exp.py --merge 100 -r 10 --small --PBM-UCB 0.01 $TESTDIR/dev_null || exit

python3 exp.py --play 2 100 -r 10 --small --PBM-UCB 0.01 --oracle $TESTDIR/dev_null --force || exit
python3 exp.py --merge 100 -r 10 --small --PBM-UCB 0.01 --oracle $TESTDIR/dev_null || exit

python3 exp.py --play 2 100 -r 10 --small --BC-MPTS $TESTDIR/dev_null --force || exit
python3 exp.py --merge 100 -r 10 --small --BC-MPTS $TESTDIR/dev_null || exit

python3 exp.py --play 2 100 -r 10 --small --BC-MPTS --oracle $TESTDIR/dev_null --force || exit
python3 exp.py --merge 100 -r 10 --small --BC-MPTS --oracle $TESTDIR/dev_null || exit

python3 exp.py --play 2 100 -r 10 --small --PB-MHB 2 --TGRW 0.1 $TESTDIR/dev_null --force || exit
python3 exp.py --merge 100 -r 10 --small --PB-MHB 2 --TGRW 0.1 $TESTDIR/dev_null || exit

python3 exp.py --play 2 100 -r 10 --small --PB-MHB 2 --TGRW 0.1 --vari_sigma $TESTDIR/dev_null --force || exit
python3 exp.py --merge 100 -r 10 --small --PB-MHB 2 --TGRW 0.1 --vari_sigma $TESTDIR/dev_null || exit

python3 exp.py --play 2 100 -r 10 --small --PB-MHB 2 --TGRW 0.1 --random_start $TESTDIR/dev_null --force || exit
python3 exp.py --merge 100 -r 10 --small --PB-MHB 2 --TGRW 0.1 --random_start $TESTDIR/dev_null || exit

python3 exp.py --play 2 100 -r 10 --small --PB-MHB 2 --TGRW 0.1 --vari_sigma --random_start $TESTDIR/dev_null --force || exit
python3 exp.py --merge 100 -r 10 --small --PB-MHB 2 --TGRW 0.1 --vari_sigma --random_start $TESTDIR/dev_null || exit

python3 exp.py --play 2 100 -r 10 --small --TopRank 100 --horizon_time_known $TESTDIR/dev_null --force || exit
python3 exp.py --merge 100 -r 10 --small --TopRank 100 --horizon_time_known $TESTDIR/dev_null || exit

python3 exp.py --play 2 100 -r 10 --small --PMED 1. 10 40 $TESTDIR/dev_null --force || exit
python3 exp.py --merge 100 -r 10 --small --PMED 1. 10 40 $TESTDIR/dev_null || exit


# --force
python3 exp.py --play 6 100 -r 10 --small --eGreedy 0.1 10 $TESTDIR/dev_null || exit
python3 exp.py --merge 100 -r 10 --small --eGreedy 0.1 10 $TESTDIR/dev_null || exit

# checkpoints
python3 exp.py --play 1 100 -r 10 --small --eGreedy 1 10 $TESTDIR/dev_null --force  || exit
python3 exp.py --play 1 100 -r 10 --small --eGreedy 1 10 $TESTDIR/dev_null_bis --nb_checkpoints 5 --force  || exit

