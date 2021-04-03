#!/usr/bin/env python3
import sys
import os
import argparse
import random
import torch.cuda
import torch.distributed
import torch.multiprocessing

from retinanet import infer
from retinanet.model import Model
from retinanet.config_defaults import _C as cfg


def parse(args):
    parser = argparse.ArgumentParser(description='RetinaNet Detection Utility.')
    parser.add_argument('--master', metavar='address:port', type=str, help='Adress and port of the master worker', default='127.0.0.1:29500')

    subparsers = parser.add_subparsers(help='sub-command', dest='command')
    subparsers.required = True

    devcount = max(1, torch.cuda.device_count())

    parser_infer = subparsers.add_parser('infer', help='run inference')
    parser_infer.add_argument('--config_file', type=str, help='path to config file', default='../configs/MAL_X-101-FPN_e2e.yaml')
    parser_infer.add_argument('--images', metavar='path', type=str, help='path to images', default='.')
    parser_infer.add_argument('--annotations', metavar='annotations', type=str,
                              help='evaluate using provided annotations')
    parser_infer.add_argument('--output', metavar='file', type=str, help='save detections to specified JSON file',
                              default='detections.json')
    parser_infer.add_argument('--batch', metavar='size', type=int, help='batch size', default=4 * devcount)
    parser_infer.add_argument('--resize', metavar='scale', type=int, help='resize to given size', default=800)
    parser_infer.add_argument('--max-size', metavar='max', type=int, help='maximum resizing size', default=1333)
    parser_infer.add_argument('--with-dali', help='use dali for data loading', action='store_true')
    parser_infer.add_argument('--full-precision', help='inference in full precision', action='store_true')

    return parser.parse_args(args)

def load_model(args, verbose=False):
    if not os.path.isfile(args.config_file):
        raise RuntimeError('Config file {} does not exist!'.format(args.config_file))
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    if verbose:
        print('Loading model from {}...'.format(os.path.basename(cfg.MODEL.WEIGHT)))
    model = Model.load(cfg)
    if verbose:
        print(model)

    state = cfg.MODEL.WEIGHT
    return model, state

def worker(rank, args, world, model, state):
    'Per-device distributed worker'

    if torch.cuda.is_available():
        os.environ.update({
            'MASTER_PORT': args.master.split(':')[-1],
            'MASTER_ADDR': ':'.join(args.master.split(':')[:-1]),
            'WORLD_SIZE':  str(world),
            'RANK':        str(rank),
            'CUDA_DEVICE': str(rank)
        })

        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

        if args.batch % world != 0:
            raise RuntimeError('Batch size should be a multiple of the number of GPUs')

    if args.command == 'infer':
        infer.infer(model, args.images, args.output, args.resize, args.max_size, args.batch,
            annotations=args.annotations, mixed_precision=not args.full_precision,
            is_master=(rank == 0), world=world, use_dali=args.with_dali, verbose=True)

def main(args=None):
    'Entry point for the retinanet command'

    args = parse(args or sys.argv[1:])

    model, state = load_model(args, verbose=True)

    world = torch.cuda.device_count()

    rank = 0
    worker(rank, args, world, model, state)

    """
    if model: model.share_memory()

    torch.multiprocessing.spawn(worker, args=(args, world, model, state), nprocs=world)
    """

if __name__ == '__main__':
    main()
