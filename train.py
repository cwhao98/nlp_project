import os
import torch
import random
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from preprocess import DataLoader
from textcnn import TextCNN


if __name__ == '__main__':
    # parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    parser.add_argument('--batchsize', help='batch size', default=64, type=int)
    parser.add_argument('--num_epoch', help='number of epochs', default=100, type=int)
    parser.add_argument('--lr', help='optimizer learning rate', default=1e-4, type=float)
    parser.add_argument('--log_dir', help='log dir', default='./snap/base')
    parser.add_argument('--model_type', default='BaseCNN')
    parser.add_argument('--gpu', help='use gpu', default=False, action='store_true')
    parser.add_argument('--num_emotion', help='number of emotion type', default=2, type=int)
    parser.add_argument('--emb_size', help='word embedding size', default=100, type=int)
    parser.add_argument('--seed', help='random seed', default=237, type=int)

    # args and init
    args = parser.parse_args()

    # set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # preprocess
    dataset = DataLoader()
    train_data, val_data = dataset.train_data, dataset.val_data
    args.vocab_size = dataset.num_word

    if args.model_type == 'BaseCNN':
        model = TextCNN(args=args)
    else:
        pass

    model.run(train_data, val_data)















