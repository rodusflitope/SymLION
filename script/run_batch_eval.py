import argparse
import glob
import os
import re
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run evaluation (sample generation) and metric computation for experiments in batch."
    )
    parser.add_argument(
        "--exp_root",
        default="../exp",
        help="Path to exp root folder (default: ../exp)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Specific date folder under exp_root to process (e.g. 0514, 0515)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Filter by specific dataset (e.g. airplane, chair, car)",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="epoch_399_iters_*.pt",
        help="Pattern or filename of checkpoint to evaluate (default: epoch_399_iters_*.pt)",
    )
    parser.add_argument(
        "--batch_size_test",
        type=int,
        default=5,
        help="Batch size for metric computation (default: 5)",
    )
    parser.add_argument(
        "--device", default="cuda", help="Device for metric computation (default: cuda)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip eval generation step if sample files already exist",
    )
    return parser.parse_args()


def extract_step_from_ckpt(ckpt_path):
    filename = os.path.basename(ckpt_path)
    match = re.search(r"iters_(\d+)", filename)
    if match:
        return match.group(1)
    match = re.search(r"epoch_(\d+)", filename)
    if match:
        return match.group(1)
    return "eval"


def extract_hash_from_folder(folder_name, dataset):
    pattern = r"^[a-z0-9]+_train_(.+)"
    match = re.match(pattern, folder_name)
    if match:
        extracted = match.group(1)
        if dataset and f"_{dataset}" in extracted:
            extracted = extracted.split(f"_{dataset}")[0]
        return extracted
    return folder_name


def find_checkpoints(exp_root, date=None, dataset=None, ckpt_pattern="epoch_399_iters_*.pt"):
    search_path = os.path.join(
        exp_root,
        date if date else "*",
        dataset if dataset else "*",
        "*",
        "checkpoints",
        ckpt_pattern,
    )
    matched = glob.glob(search_path)

    if not matched:
        alt_search_path = os.path.join(
            exp_root,
            date if date else "*",
            dataset if dataset else "*",
            "*",
            "checkpoints",
            "*.pt",
        )
        matched = glob.glob(alt_search_path)

    matched.sort()
    return matched


def main():
    args = parse_args()
    checkpoints = find_checkpoints(
        args.exp_root, args.date, args.dataset, args.checkpoint_name
    )

    if not checkpoints:
        print(f"No checkpoints found matching criteria in {args.exp_root}")
        return

    print(f"Found {len(checkpoints)} checkpoint(s) to evaluate:")
    for ckpt in checkpoints:
        print(f"  - {ckpt}")

    for idx, ckpt_path in enumerate(checkpoints, 1):
        print(f"\n=======================================================")
        print(f"  Processing [{idx}/{len(checkpoints)}]: {ckpt_path}")
        print(f"=======================================================")

        abs_ckpt = os.path.abspath(ckpt_path)
        ckpt_dir = os.path.dirname(abs_ckpt)
        exp_dir = os.path.dirname(ckpt_dir)
        dataset_dir = os.path.dirname(exp_dir)

        dataset = os.path.basename(dataset_dir)
        exp_folder = os.path.basename(exp_dir)
        step = extract_step_from_ckpt(ckpt_path)
        hash_label = extract_hash_from_folder(exp_folder, dataset)

        ref_file = f"./datasets/test_data/ref_val_{dataset}.pt"

        eval_dir = os.path.join(exp_dir, "eval")
        sample_pattern = os.path.join(eval_dir, f"samples_{step}*.pt")
        existing_samples = glob.glob(sample_pattern)

        if args.skip_eval and existing_samples:
            print(f"Found existing samples at {existing_samples[0]}, skipping eval generation.")
            sample_file = existing_samples[0]
        else:
            eval_cmd = ["bash", "./script/eval.sh", ckpt_path]
            print(f"\nStep 1: Generating samples with command:\n    {' '.join(eval_cmd)}")
            if not args.dry_run:
                res = subprocess.run(eval_cmd)
                if res.returncode != 0:
                    print(f"Error running eval.sh for {ckpt_path}, skipping metrics.")
                    continue

            samples_found = glob.glob(sample_pattern)
            if not samples_found:
                samples_found = glob.glob(os.path.join(eval_dir, "samples_*.pt"))

            if not samples_found:
                print(f"Could not find generated samples in {eval_dir}")
                continue
            sample_file = samples_found[0]

        metrics_cmd = [
            sys.executable,
            "script/compute_metrics.py",
            "--samples",
            sample_file,
            "--ref",
            ref_file,
            "--batch_size_test",
            str(args.batch_size_test),
            "--device",
            args.device,
            "--dataset",
            dataset,
            "--hash",
            hash_label,
            "--step",
            str(step),
            "--epoch",
            "eval",
        ]

        print(f"\n[>] Step 2: Computing metrics with command:\n    {' '.join(metrics_cmd)}")
        if not args.dry_run:
            res = subprocess.run(metrics_cmd)
            if res.returncode != 0:
                print(f"Error running compute_metrics.py for {sample_file}")
            else:
                print(f"Successfully completed evaluation & metrics for {exp_folder}")


if __name__ == "__main__":
    main()
