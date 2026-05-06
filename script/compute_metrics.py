import argparse
import sys

sys.path.append('.')

from utils.eval_helper import compute_score


def parse_args():
    parser = argparse.ArgumentParser(description='Compute LION metrics from saved samples')
    parser.add_argument('--samples', required=True, help='Path to samples_*.pt')
    parser.add_argument('--ref', required=True, help='Path to ref_val_*.pt')
    parser.add_argument('--batch_size_test', type=int, default=5,
                        help='Batch size for metric computation')
    parser.add_argument('--device', default='cuda', help='Device string (e.g., cuda, cuda:0, cpu)')
    parser.add_argument('--norm_box', action='store_true',
                        help='Enable box normalization before metrics')
    parser.add_argument('--dataset', default='', help='Optional dataset label for report')
    parser.add_argument('--hash', default='', help='Optional run hash/name for report')
    parser.add_argument('--step', default='eval', help='Optional step label for report')
    parser.add_argument('--epoch', default='eval', help='Optional epoch label for report')
    return parser.parse_args()


def main():
    args = parse_args()
    compute_score(
        args.samples,
        ref_name=args.ref,
        batch_size_test=args.batch_size_test,
        device_str=args.device,
        norm_box=args.norm_box,
        dataset=args.dataset,
        hash=args.hash,
        step=args.step,
        epoch=args.epoch,
    )


if __name__ == '__main__':
    main()
