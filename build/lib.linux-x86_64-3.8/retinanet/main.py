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
from retinanet._C import Engine #junl

def parse(args):
    parser = argparse.ArgumentParser(description='RetinaNet Detection Utility.')
    parser.add_argument('--master', metavar='address:port', type=str, help='Adress and port of the master worker', default='127.0.0.1:29500')

    subparsers = parser.add_subparsers(help='sub-command', dest='command')
    subparsers.required = True

    devcount = max(1, torch.cuda.device_count())

    parser_infer = subparsers.add_parser('infer', help='run inference')
    parser_infer.add_argument('--config_file', type=str, help='path to config file', default='configs/MAL_R-50-FPN_e2e.yaml')
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

  
    parser_export = subparsers.add_parser('export', help='export a model into a TensorRT engine')
    parser_export.add_argument('--config_file', type=str, help='path to config file', default='../configs/MAL_X-101-FPN_e2e.yaml')
    parser_export.add_argument('model', type=str, help='path to model')
    parser_export.add_argument('export', type=str, help='path to exported output')
    parser_export.add_argument('--size', metavar='height width', type=int, nargs='+', help='input size (square) or sizes (h w) to use when generating TensorRT engine', default=[1280])
    parser_export.add_argument('--batch', metavar='size', type=int, help='max batch size to use for TensorRT engine', default=2)
    parser_export.add_argument('--full-precision', help='export in full instead of half precision', action='store_true')
    parser_export.add_argument('--int8', help='calibrate model and export in int8 precision', action='store_true')
    parser_export.add_argument('--calibration-batches', metavar='size', type=int, help='number of batches to use for int8 calibration', default=10)
    parser_export.add_argument('--calibration-images', metavar='path', type=str, help='path to calibration images to use for int8 calibration', default="")
    parser_export.add_argument('--calibration-table', metavar='path', type=str, help='path of existing calibration table to load from, or name of new calibration table', default="")
    parser_export.add_argument('--verbose', help='enable verbose logging', action='store_true')

    return parser.parse_args(args)

def load_model(args, verbose=False):
    if not os.path.isfile(args.config_file):
        raise RuntimeError('Config file {} does not exist!'.format(args.config_file))
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    model = None
    state = {}
    _, ext = os.path.splitext(cfg.MODEL.WEIGHT)
    print('####ext:',ext)

    if ext == '.pth' or ext == '.torch':
       if verbose:
          print('Loading model from {}...'.format(os.path.basename(cfg.MODEL.WEIGHT)))
          #print('***********Cfg:',cfg)
          #print('#####*cfg.MODEL.WEIGHT:',cfg.MODEL.WEIGHT)

       model = Model.load(cfg)
       if verbose:
        print(model)
    elif ext in ['.engine', '.plan']:
        model = None

    else:
        raise RuntimeError('Invalid model format "{}"!'.format(args.ext))    

    #state = cfg.MODEL.WEIGHT
    state['path'] = cfg.MODEL.WEIGHT
    return model, state

def worker(rank, args, world, model, state):
    'Per-device distributed worker'
    print('####rank:',rank)
    #import pdb;pdb.set_trace()
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
        if model is None:  #junl
            if rank == 0: print('Loading CUDA engine from {}...'.format(os.path.basename(cfg.MODEL.WEIGHT)))
            print('cfg.MODEL.WEIGHT',cfg.MODEL.WEIGHT)
            model = Engine.load(cfg.MODEL.WEIGHT)
        #print('  resize:',args.resize)
        print('max_size:',args.max_size)

        infer.infer(model, args.images, args.output, args.resize, args.max_size, args.batch,
            annotations=args.annotations, mixed_precision=not args.full_precision,
            is_master=(rank == 0), world=world, use_dali=args.with_dali, verbose=True)

    elif args.command == 'export':
        #import pdb;pdb.set_trace()
        onnx_only = args.export.split('.')[-1] == 'onnx'
        input_size = args.size * 2 if len(args.size) == 1 else args.size

        calibration_files = []
        if args.int8:
            # Get list of images to use for calibration
            if os.path.isdir(args.calibration_images):
                import glob
                file_extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
                for ex in file_extensions:
                    calibration_files += glob.glob("{}/*{}".format(args.calibration_images, ex), recursive=True)
                # Only need enough images for specified num of calibration batches
                if len(calibration_files) >= args.calibration_batches * args.batch:
                    calibration_files = calibration_files[:(args.calibration_batches * args.batch)]
                else:
                    print('Only found enough images for {} batches. Continuing anyway...'.format(len(calibration_files) // args.batch))

                random.shuffle(calibration_files)

        precision = "FP32"
        if args.int8:
            precision = "INT8"
        elif not args.full_precision:
            precision = "FP16"

        exported = model.export(input_size, args.batch, precision, calibration_files, args.calibration_table, args.verbose, onnx_only=onnx_only)
        if onnx_only:
            with open(args.export, 'wb') as out:
                out.write(exported)
        else:
            exported.save(args.export)


def main(args=None):
    'Entry point for the retinanet command'

    args = parse(args or sys.argv[1:])
    print('args:',args)

    model, state = load_model(args, verbose=True)

    world = torch.cuda.device_count()
    print('  world:',world)

    rank = 0
    if args.command == 'export' or world <= 1:
        world = 1
    worker(rank, args, world, model, state)

    """    
    if model: model.share_memory()

    torch.multiprocessing.spawn(worker, args=(args, world, model, state), nprocs=world)
    """
    

if __name__ == '__main__':
    main()
