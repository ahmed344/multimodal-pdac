"""Capped real-data training/export and full logical spatial GPU batch check.

Run from repository root: python -m dann.verification.real_smoke
Artifacts remain under each combination's verification/smoke run directory.
"""
from pathlib import Path
import argparse
import copy
import json
import time
import gc
import torch
import numpy as np

from dann.config import load_config, apply_smoke_overrides, resolve_execution, seed_everything
from dann.data_loader import create_data_bundle
from dann.model import AdversarialLatentFusion
from dann.losses import ZILNLoss
from dann.train import train_model, move_batch_to_device
from dann.spatial import run_inference


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--mode', choices=['benchmark', 'mlp', 'cnn'], required=True)
    parser.add_argument("--export-inference", action="store_true", help="Also export independent inference input.")
    args = parser.parse_args()
    torch.set_num_threads(4)
    c = load_config(Path('dann/config.yaml'))
    c['results']['run_name'] = 'verification'
    c['training'].update(smoke_train_samples=128, smoke_validation_samples=64, smoke_test_samples=64)
    c['analysis'].update(figure_dpi=100, density_mc_samples=100, activity_max_nonzeros=5000000)
    if args.mode == 'mlp':
        c['model']['aggregation']['type'] = 'mlp'
        c['model']['heads']['biology']['type'] = 'mlp'
    c = apply_smoke_overrides(c)
    root = Path(c['results']['run_dir'])
    root.mkdir(parents=True, exist_ok=True)
    if args.mode == 'benchmark':
        seed_everything(c['training']['seed'])
        c['data']['max_train_samples'] = None
        c['training']['num_workers'] = 4
        c['training']['prefetch_factor'] = 2
        c['training']['core_size'] = 32
        c['training']['batch_size'] = 8
        c['training']['validation_batch_size'] = 8
        c['training']['tiles_per_batch'] = 8
        start = time.perf_counter()
        bundle = create_data_bundle(c)
        cpu = next(iter(bundle.loaders['train']))
        load_seconds = time.perf_counter() - start
        model = AdversarialLatentFusion.from_config(c, len(bundle.metadata.batch_names), 4).cuda()
        batch = move_batch_to_device(cpu, torch.device('cuda'))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        out = model(batch, grl_strength=.25)
        loss = ZILNLoss.from_config(c).cuda()(out['pi_logits'], out['mu'], out['sigma'],
                   batch['targets'], batch['target_valid_mask']).total
        loss = 5*loss + torch.nn.functional.cross_entropy(out['batch_logits'], batch['batches'])
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        result = {'gpu': torch.cuda.get_device_name(), 'seconds': time.perf_counter()-start,
                  'data_load_seconds': load_seconds,
                  'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                  'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30,
                  'tiles': len(cpu['occupancy']), 'tile_shape': list(cpu['occupancy'].shape),
                  'occupied_context': len(cpu['peak_counts']), 'active_peaks': len(cpu['peak_indices']),
                  'supervised_core_rows': len(cpu['targets']), 'loss': float(loss.detach()),
                  'peak_budget': c['model']['spectral_peak_budget'], 'checkpointing': True, 'workers': c['training']['num_workers'],
                  'prefetch_factor': c['training']['prefetch_factor']}
        (root/'gpu_benchmark.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        for d in bundle.datasets.values(): d.close()
    else:
        start = time.perf_counter()
        checkpoint = train_model(c)
        torch.cuda.empty_cache()
        gc.collect()
        if args.export_inference:
            from dann.config import inference_input
            run_inference(inference_input(c), checkpoint, Path(c['spatial']['output']),
                          latent_output_path=Path(c['spatial']['latents']), max_rows=128,
                          num_workers_override=0, overwrite=True)
        (root/'smoke_runtime.json').write_text(json.dumps({'seconds':time.perf_counter()-start,
            'checkpoint': str(checkpoint), 'train_labels':128, 'validation_labels':64, 'test_labels':64,
            'export_rows':128 if args.export_inference else 0}, indent=2))
        print(f'Finished {args.mode}: {root}', flush=True)


if __name__ == '__main__':
    main()
