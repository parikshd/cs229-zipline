import argparse

from p05b_lwr import main as p05b

parser = argparse.ArgumentParser()
parser.add_argument('p_num', nargs='?', type=int, default=0,
                    help='Problem number to run, 0 for all problems.')
args = parser.parse_args()

if args.p_num == 0 or args.p_num == 5:
    p05b(tau=250,
         train_path='output/flights_pass_1_na_0.csv',
         eval_path='testinput/flights_new_till_03dec.csv')
