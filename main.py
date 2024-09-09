import yaml
import json
import argparse
from easydict import EasyDict

from rl.rl import RL

def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch implementation of MID')
    parser.add_argument('--exp_name', default='test')
    parser.add_argument('--dataset', default='zara1')
    parser.add_argument('--pre', default='', type=str)
    parser.add_argument('--eval_at', default=90, type=int)
    parser.add_argument('--process_data', default=False, type=bool)

    parser.add_argument('--reward_name', default='average_baseline_linear_10_negative_0.1')
    parser.add_argument('--reset', default='random')
    parser.add_argument('--ceof_ref', default=1, type=float)
    parser.add_argument('--lr', default=5e-6, type=float)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--clip_coef', default=0.1, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--optimizer', default='Adam')

    parser.add_argument('--worker', default=128, type=int)
    parser.add_argument('--sample', default=20, type=int)
    parser.add_argument('--discount_sigma', default=1, type=float)
    parser.add_argument('--discount_ref', default=1, type=float)

    return parser.parse_args()


def main():
    args = parse_args()

    with open('configs/baseline.yaml') as f:
       config = yaml.safe_load(f)
    config = EasyDict(config)
    config.eval_mode = True
    config.exp_name = 'baseline'
    config.dataset = args.dataset
    config.pre = args.pre
    config.eval_at = args.eval_at

    # if config.dataset == 'eth':
    #     config.eval_at = 62
    # elif config.dataset == 'hotel':
    #     config.eval_at = 73
    # elif config.dataset == 'univ':
    #     config.eval_at = 68
    # elif config.dataset == 'zara1':
    #     config.eval_at = 88
    # elif config.dataset == 'zara2':
    #     config.eval_at = 80
    # elif config.dataset == 'sdd':
    #     config.eval_at = 80

    if config.dataset == 'univ' or 'zara2':
        config.reward_name = 'average_baseline_linear_5_negative_0.1'

    agent = RL(config)
    if args.process_data == True:
        agent.get_data('eval')
        agent.get_data('train')
        return
    else:
        with open('rl/config.json', 'r') as f:#读取config
            rl_config = json.loads(f.read())
        for k, v in vars(args).items():
            rl_config[k] = v
        rl_config = EasyDict(rl_config)
        
        agent.train(rl_config)


if __name__ == '__main__':
    main()
