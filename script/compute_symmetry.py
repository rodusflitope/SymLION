import argparse
import os
import sys
import csv

import numpy as np
import torch

sys.path.append('.')

def chamfer_distance_batch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dists_xy = torch.cdist(x, y, p=2.0).pow(2)  # [B, N, M]
    min_xy = dists_xy.min(dim=2)[0].mean(dim=1)  # [B]
    min_yx = dists_xy.min(dim=1)[0].mean(dim=1)  # [B]
    return min_xy + min_yx


def reflection_symmetry_cd(
    pc: torch.Tensor,
    axis: int = 0,
    batch_size: int = 64,
) -> torch.Tensor:
    B = pc.shape[0]
 
    centroid = pc.mean(dim=1, keepdim=True)
    pc_c = pc - centroid

    pc_r = pc_c.clone()
    pc_r[:, :, axis] = -pc_r[:, :, axis]

    all_cd = []
    for i in range(0, B, batch_size):
        j = min(i + batch_size, B)
        cd = chamfer_distance_batch(pc_c[i:j], pc_r[i:j])
        all_cd.append(cd)
    return torch.cat(all_cd, dim=0)

def load_and_denormalize(samples_path: str, ref_path: str):

    ref_data = torch.load(ref_path, map_location='cpu')
    ref_pcs = ref_data['ref'][:, :, :3]
    m_pcs = ref_data['mean']
    s_pcs = ref_data['std']

    gen_pcs = torch.load(samples_path, map_location='cpu')
    if isinstance(gen_pcs, dict):
        gen_pcs = gen_pcs['ref']

    if gen_pcs.shape[1] > ref_pcs.shape[1]:
        xperm = np.random.permutation(np.arange(gen_pcs.shape[1]))[
            :ref_pcs.shape[1]]
        gen_pcs = gen_pcs[:, xperm]

    N_ref = ref_pcs.shape[0]
    m_pcs = m_pcs[:N_ref]
    s_pcs = s_pcs[:N_ref]
    ref_pcs = ref_pcs[:N_ref]
    gen_pcs = gen_pcs[:N_ref]

    if gen_pcs.shape[2] == 6:
        gen_pcs = gen_pcs[:, :, :3]
    if ref_pcs.shape[2] == 6:
        ref_pcs = ref_pcs[:, :, :3]

    ref_pcs = ref_pcs * s_pcs + m_pcs
    gen_pcs = gen_pcs * s_pcs + m_pcs

    return gen_pcs, ref_pcs


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute Reflection Symmetry CD on LION samples')
    parser.add_argument('--samples', required=True,
                        help='Path to generated samples .pt file')
    parser.add_argument('--ref', required=True,
                        help='Path to reference ref_val_*.pt file')
    parser.add_argument('--axis', type=int, default=0,
                        help='Reflection axis: 0=X, 1=Y, 2=Z (default: 0)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Sub-batch size for CD computation')
    parser.add_argument('--device', default='cuda',
                        help='Device (cuda, cpu)')
    parser.add_argument('--dataset', default='',
                        help='Dataset label for CSV output')
    parser.add_argument('--hash', default='',
                        help='Run hash/name for CSV output')
    parser.add_argument('--output', default='results/symmetry_out.csv',
                        help='Path to output CSV file')
    return parser.parse_args()


def main():
    args = parse_args()
    axis_names = {0: 'X', 1: 'Y', 2: 'Z'}
    axis_name = axis_names.get(args.axis, str(args.axis))

    print(f'[Symmetry] Loading samples: {args.samples}')
    print(f'[Symmetry] Loading reference: {args.ref}')
    print(f'[Symmetry] Reflection axis: {axis_name} (index {args.axis})')

    gen_pcs, ref_pcs = load_and_denormalize(args.samples, args.ref)
    print(f'[Symmetry] gen shape: {gen_pcs.shape}, ref shape: {ref_pcs.shape}')

    device = torch.device(args.device)
    gen_pcs = gen_pcs.to(device).float()
    ref_pcs = ref_pcs.to(device).float()

    print('[Symmetry] Computing Sym-CD for generated samples...')
    with torch.no_grad():
        sym_cd_gen = reflection_symmetry_cd(
            gen_pcs, axis=args.axis, batch_size=args.batch_size)

    print('[Symmetry] Computing Sym-CD for reference (GT)...')
    with torch.no_grad():
        sym_cd_ref = reflection_symmetry_cd(
            ref_pcs, axis=args.axis, batch_size=args.batch_size)

    mean_gen = sym_cd_gen.mean().item()
    std_gen = sym_cd_gen.std().item()
    mean_ref = sym_cd_ref.mean().item()
    std_ref = sym_cd_ref.std().item()
    ratio = mean_gen / max(mean_ref, 1e-10)

    print('\n' + '=' * 60)
    print(f'  Reflection Symmetry CD (axis={axis_name})')
    print('=' * 60)
    print(f'  Generated:  mean={mean_gen:.6f}  std={std_gen:.6f}')
    print(f'  Reference:  mean={mean_ref:.6f}  std={std_ref:.6f}')
    print(f'  Ratio (gen/ref, 1.0=ideal): {ratio:.4f}')
    print('=' * 60)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    file_exists = os.path.exists(args.output)
    with open(args.output, 'a', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        if not file_exists:
            writer.writerow([
                'Dataset', 'Model', 'Axis',
                'Sym-CD Gen (mean)', 'Sym-CD Gen (std)',
                'Sym-CD Ref (mean)', 'Sym-CD Ref (std)',
                'Sym-CD Ratio',
            ])
        writer.writerow([
            args.dataset or 'unknown',
            args.hash or 'unknown',
            axis_name,
            f'{mean_gen:.6f}',
            f'{std_gen:.6f}',
            f'{mean_ref:.6f}',
            f'{std_ref:.6f}',
            f'{ratio:.4f}',
        ])
    print(f'[Symmetry] Results appended to {args.output}')


if __name__ == '__main__':
    main()
