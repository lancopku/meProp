'''
Train MLPs for MNIST using meProp
'''
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
from argparse import ArgumentParser

import torch

from data import get_mnist
from util import TestGroup


def get_args():
    # a simple use example (not unified)
    parser = ArgumentParser()
    parser.add_argument(
        '--n_epoch', type=int, default=20, help='number of training epochs')
    parser.add_argument(
        '--d_hidden', type=int, default=512, help='dimension of hidden layers')
    parser.add_argument(
        '--n_layer',
        type=int,
        default=3,
        help='number of layers, including the output layer')
    parser.add_argument(
        '--d_minibatch', type=int, default=32, help='size of minibatches')
    parser.add_argument(
        '--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument(
        '--k',
        type=int,
        default=30,
        help='k in meProp (if invalid, e.g. 0, do not use meProp)')
    parser.add_argument(
        '--unified',
        dest='unified',
        action='store_true',
        help='use unified meProp')
    parser.add_argument(
        '--no-unified',
        dest='unified',
        action='store_false',
        help='do not use unified meProp')
    parser.add_argument(
        '--random_seed', type=int, default=12976, help='random seed')
    parser.set_defaults(unified=False)
    return parser.parse_args()


def get_args_unified():
    # a simple use example (unified)
    parser = ArgumentParser()
    parser.add_argument(
        '--n_epoch', type=int, default=20, help='number of training epochs')
    parser.add_argument(
        '--d_hidden', type=int, default=500, help='dimension of hidden layers')
    parser.add_argument(
        '--n_layer',
        type=int,
        default=3,
        help='number of layers, including the output layer')
    parser.add_argument(
        '--d_minibatch', type=int, default=50, help='size of minibatches')
    parser.add_argument(
        '--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument(
        '--k',
        type=int,
        default=30,
        help='k in meProp (if invalid, e.g. 0, do not use meProp)')
    parser.add_argument(
        '--unified',
        dest='unified',
        action='store_true',
        help='use unified meProp')
    parser.add_argument(
        '--no-unified',
        dest='unified',
        action='store_false',
        help='do not use unified meProp')
    parser.add_argument(
        '--random_seed', type=int, default=12976, help='random seed')
    parser.set_defaults(unified=True)
    return parser.parse_args()


def main():
    args = get_args()
    trn, dev, tst = get_mnist()

    group = TestGroup(
        args,
        trn,
        args.d_minibatch,
        args.d_hidden,
        args.n_layer,
        args.dropout,
        args.unified,
        dev,
        tst,
        file=sys.stdout)

    # results may be different at each run
    group.run(0, args.n_epoch)
    # mbsize: 32, hidden size: 512, layer: 3, dropout: 0.1, k: 0
    # 0：dev set: Average loss: 0.1202, Accuracy: 4826/5000 (96.52%)
    #     test set: Average loss: 0.1272, Accuracy: 9616/10000 (96.16%)
    # 1：dev set: Average loss: 0.1047, Accuracy: 4832/5000 (96.64%)
    #     test set: Average loss: 0.1131, Accuracy: 9658/10000 (96.58%)
    # 2：dev set: Average loss: 0.0786, Accuracy: 4889/5000 (97.78%)
    #     test set: Average loss: 0.0834, Accuracy: 9749/10000 (97.49%)
    # 3：dev set: Average loss: 0.0967, Accuracy: 4875/5000 (97.50%)
    # 4：dev set: Average loss: 0.0734, Accuracy: 4907/5000 (98.14%)
    #     test set: Average loss: 0.0818, Accuracy: 9790/10000 (97.90%)
    # 5：dev set: Average loss: 0.0848, Accuracy: 4894/5000 (97.88%)
    # 6：dev set: Average loss: 0.0750, Accuracy: 4904/5000 (98.08%)
    # 7：dev set: Average loss: 0.0882, Accuracy: 4897/5000 (97.94%)
    # 8：dev set: Average loss: 0.0978, Accuracy: 4896/5000 (97.92%)
    # 9：dev set: Average loss: 0.0828, Accuracy: 4910/5000 (98.20%)
    #     test set: Average loss: 0.0973, Accuracy: 9797/10000 (97.97%)
    # 10：dev set: Average loss: 0.1004, Accuracy: 4901/5000 (98.02%)
    # 11：dev set: Average loss: 0.0813, Accuracy: 4914/5000 (98.28%)
    #     test set: Average loss: 0.0917, Accuracy: 9795/10000 (97.95%)
    # 12：dev set: Average loss: 0.0880, Accuracy: 4912/5000 (98.24%)
    # 13：dev set: Average loss: 0.1106, Accuracy: 4910/5000 (98.20%)
    # 14：dev set: Average loss: 0.0981, Accuracy: 4921/5000 (98.42%)
    #     test set: Average loss: 0.1065, Accuracy: 9824/10000 (98.24%)
    # 15：dev set: Average loss: 0.1044, Accuracy: 4914/5000 (98.28%)
    # 16：dev set: Average loss: 0.1235, Accuracy: 4904/5000 (98.08%)
    # 17：dev set: Average loss: 0.1202, Accuracy: 4910/5000 (98.20%)
    # 18：dev set: Average loss: 0.1021, Accuracy: 4926/5000 (98.52%)
    #     test set: Average loss: 0.1167, Accuracy: 9800/10000 (98.00%)
    # 19：dev set: Average loss: 0.1490, Accuracy: 4910/5000 (98.20%)
    # $98.52|98.00 at 18
    group.run(args.k, args.n_epoch)
    # mbsize: 32, hidden size: 512, layer: 3, dropout: 0.1, k: 30
    # 0：dev set: Average loss: 0.3283, Accuracy: 4754/5000 (95.08%)
    #     test set: Average loss: 0.3611, Accuracy: 9480/10000 (94.80%)
    # 1：dev set: Average loss: 0.1762, Accuracy: 4804/5000 (96.08%)
    #     test set: Average loss: 0.1837, Accuracy: 9592/10000 (95.92%)
    # 2：dev set: Average loss: 0.1192, Accuracy: 4871/5000 (97.42%)
    #     test set: Average loss: 0.1231, Accuracy: 9715/10000 (97.15%)
    # 3：dev set: Average loss: 0.0966, Accuracy: 4875/5000 (97.50%)
    #     test set: Average loss: 0.0978, Accuracy: 9745/10000 (97.45%)
    # 4：dev set: Average loss: 0.1027, Accuracy: 4880/5000 (97.60%)
    #     test set: Average loss: 0.0846, Accuracy: 9774/10000 (97.74%)
    # 5：dev set: Average loss: 0.1052, Accuracy: 4865/5000 (97.30%)
    # 6：dev set: Average loss: 0.0888, Accuracy: 4889/5000 (97.78%)
    #     test set: Average loss: 0.0936, Accuracy: 9773/10000 (97.73%)
    # 7：dev set: Average loss: 0.0840, Accuracy: 4894/5000 (97.88%)
    #     test set: Average loss: 0.0890, Accuracy: 9797/10000 (97.97%)
    # 8：dev set: Average loss: 0.0845, Accuracy: 4894/5000 (97.88%)
    # 9：dev set: Average loss: 0.0896, Accuracy: 4893/5000 (97.86%)
    # 10：dev set: Average loss: 0.0929, Accuracy: 4902/5000 (98.04%)
    #     test set: Average loss: 0.1005, Accuracy: 9785/10000 (97.85%)
    # 11：dev set: Average loss: 0.0903, Accuracy: 4906/5000 (98.12%)
    #     test set: Average loss: 0.0950, Accuracy: 9792/10000 (97.92%)
    # 12：dev set: Average loss: 0.1019, Accuracy: 4906/5000 (98.12%)
    # 13：dev set: Average loss: 0.0870, Accuracy: 4919/5000 (98.38%)
    #     test set: Average loss: 0.0894, Accuracy: 9823/10000 (98.23%)
    # 14：dev set: Average loss: 0.1003, Accuracy: 4909/5000 (98.18%)
    # 15：dev set: Average loss: 0.1162, Accuracy: 4910/5000 (98.20%)
    # 16：dev set: Average loss: 0.1031, Accuracy: 4920/5000 (98.40%)
    #     test set: Average loss: 0.1001, Accuracy: 9828/10000 (98.28%)
    # 17：dev set: Average loss: 0.0994, Accuracy: 4913/5000 (98.26%)
    # 18：dev set: Average loss: 0.0969, Accuracy: 4905/5000 (98.10%)
    # 19：dev set: Average loss: 0.1095, Accuracy: 4920/5000 (98.40%)
    # $98.40|98.28 at 16


def main_unified():
    args = get_args_unified()
    trn, dev, tst = get_mnist()

    # change the sys.stdout to a file object to write the results to the file
    group = TestGroup(
        args,
        trn,
        args.d_minibatch,
        args.d_hidden,
        args.n_layer,
        args.dropout,
        args.unified,
        dev,
        tst,
        file=sys.stdout)

    # results may be different at each run
    group.run(0)
    # mbsize: 50, hidden size: 500, layer: 3, dropout: 0.1, k: 0
    # 0：dev set: Average loss: 0.1043, Accuracy: 4843/5000 (96.86%)
    #     test set: Average loss: 0.1163, Accuracy: 9655/10000 (96.55%)
    # 1：dev set: Average loss: 0.0789, Accuracy: 4892/5000 (97.84%)
    #     test set: Average loss: 0.0792, Accuracy: 9766/10000 (97.66%)
    # 2：dev set: Average loss: 0.0818, Accuracy: 4875/5000 (97.50%)
    # 3：dev set: Average loss: 0.0823, Accuracy: 4880/5000 (97.60%)
    # 4：dev set: Average loss: 0.0869, Accuracy: 4888/5000 (97.76%)
    # 5：dev set: Average loss: 0.0810, Accuracy: 4904/5000 (98.08%)
    #     test set: Average loss: 0.0711, Accuracy: 9807/10000 (98.07%)
    # 6：dev set: Average loss: 0.0752, Accuracy: 4903/5000 (98.06%)
    # 7：dev set: Average loss: 0.0805, Accuracy: 4907/5000 (98.14%)
    #     test set: Average loss: 0.0833, Accuracy: 9799/10000 (97.99%)
    # 8：dev set: Average loss: 0.1105, Accuracy: 4876/5000 (97.52%)
    # 9：dev set: Average loss: 0.0913, Accuracy: 4901/5000 (98.02%)
    # 10：dev set: Average loss: 0.0800, Accuracy: 4915/5000 (98.30%)
    #     test set: Average loss: 0.0832, Accuracy: 9830/10000 (98.30%)
    # 11：dev set: Average loss: 0.0909, Accuracy: 4913/5000 (98.26%)
    # 12：dev set: Average loss: 0.0908, Accuracy: 4907/5000 (98.14%)
    # 13：dev set: Average loss: 0.0753, Accuracy: 4920/5000 (98.40%)
    #     test set: Average loss: 0.0902, Accuracy: 9803/10000 (98.03%)
    # 14：dev set: Average loss: 0.0947, Accuracy: 4918/5000 (98.36%)
    # 15：dev set: Average loss: 0.0800, Accuracy: 4918/5000 (98.36%)
    # 16：dev set: Average loss: 0.0822, Accuracy: 4915/5000 (98.30%)
    # 17：dev set: Average loss: 0.1059, Accuracy: 4916/5000 (98.32%)
    # 18：dev set: Average loss: 0.1128, Accuracy: 4914/5000 (98.28%)
    # 19：dev set: Average loss: 0.0936, Accuracy: 4924/5000 (98.48%)
    #     test set: Average loss: 0.1214, Accuracy: 9794/10000 (97.94%)
    # $98.48|97.94 at 19
    group.run()
    # mbsize: 50, hidden size: 500, layer: 3, dropout: 0.1, k: 30
    # 0：dev set: Average loss: 0.1749, Accuracy: 4743/5000 (94.86%)
    #     test set: Average loss: 0.1867, Accuracy: 9466/10000 (94.66%)
    # 1：dev set: Average loss: 0.1180, Accuracy: 4828/5000 (96.56%)
    #     test set: Average loss: 0.1219, Accuracy: 9638/10000 (96.38%)
    # 2：dev set: Average loss: 0.1190, Accuracy: 4818/5000 (96.36%)
    # 3：dev set: Average loss: 0.0901, Accuracy: 4868/5000 (97.36%)
    #     test set: Average loss: 0.0984, Accuracy: 9697/10000 (96.97%)
    # 4：dev set: Average loss: 0.0834, Accuracy: 4876/5000 (97.52%)
    #     test set: Average loss: 0.0883, Accuracy: 9736/10000 (97.36%)
    # 5：dev set: Average loss: 0.0849, Accuracy: 4866/5000 (97.32%)
    # 6：dev set: Average loss: 0.0775, Accuracy: 4891/5000 (97.82%)
    #     test set: Average loss: 0.0893, Accuracy: 9732/10000 (97.32%)
    # 7：dev set: Average loss: 0.0767, Accuracy: 4893/5000 (97.86%)
    #     test set: Average loss: 0.0777, Accuracy: 9770/10000 (97.70%)
    # 8：dev set: Average loss: 0.0788, Accuracy: 4881/5000 (97.62%)
    # 9：dev set: Average loss: 0.0816, Accuracy: 4894/5000 (97.88%)
    #     test set: Average loss: 0.0840, Accuracy: 9756/10000 (97.56%)
    # 10：dev set: Average loss: 0.0820, Accuracy: 4884/5000 (97.68%)
    # 11：dev set: Average loss: 0.0776, Accuracy: 4887/5000 (97.74%)
    # 12：dev set: Average loss: 0.0779, Accuracy: 4896/5000 (97.92%)
    #     test set: Average loss: 0.0740, Accuracy: 9791/10000 (97.91%)
    # 13：dev set: Average loss: 0.0811, Accuracy: 4891/5000 (97.82%)
    # 14：dev set: Average loss: 0.0817, Accuracy: 4904/5000 (98.08%)
    #     test set: Average loss: 0.0831, Accuracy: 9783/10000 (97.83%)
    # 15：dev set: Average loss: 0.0828, Accuracy: 4895/5000 (97.90%)
    # 16：dev set: Average loss: 0.0836, Accuracy: 4897/5000 (97.94%)
    # 17：dev set: Average loss: 0.0774, Accuracy: 4904/5000 (98.08%)
    # 18：dev set: Average loss: 0.0783, Accuracy: 4914/5000 (98.28%)
    #     test set: Average loss: 0.0707, Accuracy: 9804/10000 (98.04%)
    # 19：dev set: Average loss: 0.0873, Accuracy: 4901/5000 (98.02%)
    # $98.28|98.04 at 18


if __name__ == '__main__':
    # this runs meprop
    main()
    # uncomment to run unified meprop
    # main_unified()
