import torch
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str)
parser.add_argument('--replace', type=str)
parser.add_argument('--dst', type=str)
parser.add_argument('--key', type=str, default='')
parser.add_argument('--tgt_key', type=str)
parser.add_argument('--no_del', action='store_true')
parser.add_argument('--display_only', action='store_true')
args = parser.parse_args()

if args.tgt_key is None:
    args.tgt_key = args.key

sdsrc = torch.load(args.src)
if args.display_only:
    print([k for k in sdsrc.keys() if k.startswith(args.key)])
    sys.exit(0)
sdrep = torch.load(args.replace)

args.key = args.key.split(',')
args.tgt_key = args.tgt_key.split(',')

for k in list(sdsrc.keys()):
    for key in args.key:
        if k.startswith(key) and not args.no_del: del sdsrc[k]
for k in list(sdrep.keys()):
    for idx, key in enumerate(args.key):
        if k.startswith(key):
            new_k = args.tgt_key[idx] + k[len(key):]
            sdsrc[new_k] = sdrep[k]

torch.save(sdsrc, args.dst)