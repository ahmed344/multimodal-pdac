"""Frozen, restartable H&E-disabled alignment sweep and architecture transfer.

Prepare with --previous PRIOR_SWEEP --output NEW --prepare; execute with --run.
Only training and validation stages are permitted. Scientific decisions are
immutable, and completed stages verify their artifacts before being reused.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd
import torch
import yaml

from dann.analysis_execution import digest_file, fingerprint
from dann.batch_alignment import AlignmentEvaluationError, NAMES
from dann.checkpoints import checkpoint_config, validate_checkpoint
from dann.config import apply_smoke_overrides, load_config, read_config, resolve_execution
from dann.data_loader import create_data_bundle
from dann.model import AdversarialLatentFusion
from dann.spatial import _file_identity
from dann.train import train_model
from dann.training_artifacts import atomic_json
from dann.verification import alignment_sweep as sweep
from dann.verification.alignment_refinement import no_exports
from dann.verification.alignment_study_metrics import LIMITS, SEED, eligibility, improves
from dann.verification.dictionary_comparison import close, immutable_json, parameter_hash
from dann.verification.pooling_diagnostics import json_clean

NHSIC = (0., .01, .03, .1, .3, 1.)
ANOVA = (.003, .01, .03, .1)
SOFT_NN = (.1, .3, 1., 3.)
STAGES = ('smoke', 'train', 'evaluate', 'full_umap')
PHASES = ('nhsic', 'anova', 'soft_nn', 'architectures')
HISTORICAL = ('control', 'reference', 'combined')


def decision(root: Path, name: str, value: dict) -> None:
    """Normalize JSON types before comparing immutable decisions on restart."""
    sweep.decision(root,name,json_clean(value))


def make_config(source: dict, root: Path, arm: str, weights: tuple, architecture: str = 'mlp') -> dict:
    """Retain scientific defaults while selecting an architecture and constant weights."""
    c = copy.deepcopy(source)
    c['model']['aggregation']['type'] = architecture
    c['model']['heads']['biology']['type'] = architecture
    c['model']['heads'].setdefault('he', {})['enabled'] = False
    c['model']['heads']['discriminator']['enabled'] = False
    c['loss']['he_weight'] = 0.
    c['loss'].setdefault('batch_alignment', {})
    return no_exports(sweep.make_config(c, root, arm, weights))


def arm_name(architecture: str, weights: tuple) -> str:
    """Canonical names deduplicate identical scientific configurations."""
    return architecture + '_' + '_'.join(n + f'{w:g}'.replace('.', 'p') for n,w in zip(NAMES, weights))


def register_arm(root: Path, weights: tuple, architecture: str = 'mlp') -> str:
    """Register exactly one immutable configuration for each distinct arm."""
    arm = arm_name(architecture, weights)
    source = yaml.safe_load((root/'source_config.yaml').read_text())
    c = make_config(source, root, arm, weights, architecture)
    path = root/'configs'/f'{arm}.yaml'
    content = yaml.safe_dump(c, sort_keys=False)
    if path.exists() and path.read_text() != content:
        raise ValueError(f'Arm configuration changed: {arm}')
    if not path.exists():
        path.write_text(content)
    record = dict(arm=arm, architecture=architecture, weights=list(weights), config=str(path),
                  config_sha256=digest_file(path), directory=c['results']['run_dir'])
    decision(root, 'arm_' + arm, record)
    target = root/'arms'/f'{arm}.json'
    if not target.exists():
        immutable_json(target, record)
    elif json.loads(target.read_text()) != record:
        raise ValueError('Arm identity changed.')
    return arm


def config_for(root: Path, arm: str) -> dict:
    """Read registered configurations without applying new defaults or overrides."""
    c = load_config(root/'configs'/f'{arm}.yaml')
    assert_disabled(c)
    return c


def assert_disabled(c: dict) -> None:
    """Fail closed if H&E or any downstream workflow becomes enabled."""
    if c['model']['heads'].get('he', {}).get('enabled', False) or c['loss']['he_weight'] != 0:
        raise ValueError('H&E must be disabled with zero weight.')
    if any(c['post_training'].values()):
        raise ValueError('All post-training workflow switches must be disabled.')


@contextmanager
def training_validation_only():
    """Runtime guards cover normal, resumed and smoke execution."""
    def forbidden(*args, **kwargs):
        raise RuntimeError('Forbidden tissue inference, spatial export or post-training workflow.')
    with ExitStack() as stack:
        for name in ('dann.workflow.run_post_training_workflow', 'dann.spatial.run_inference',
                     'dann.spatial._run_inference', 'dann.spatial_heatmaps.run_spatial_heatmaps'):
            stack.enter_context(patch(name, forbidden))
        yield


def prepare(root: Path, previous: Path, source: Path) -> None:
    """Freeze the old protocol, historical evidence and current scientific settings."""
    previous, source = previous.resolve(), source.resolve()
    old = sweep.manifest_for(previous)
    c = load_config(source)
    prior = yaml.safe_load((previous/'source_config.yaml').read_text())
    # Compare the experiment settings after removing only the intended changes.
    expected = make_config(prior, root, 'identity', (0.,0.,0.))
    actual = make_config(c, root, 'identity', (0.,0.,0.))
    for key in ('model', 'data', 'loss', 'training'):
        if actual[key] != expected[key]:
            raise ValueError(f'Root {key} settings differ from the historical study beyond H&E/alignment.')
    if (c['training']['seed'] != SEED or c['training']['epochs'] != 300 or
            c['training']['early_stopping_metric'] != 'supervised_objective'):
        raise ValueError('Study requires the original seed, horizon and supervised stopping.')
    historical = {a:sweep.records_for(previous,a) for a in HISTORICAL}
    anchor = json.loads((previous/'decisions/recommendation.json').read_text())['selected']
    if not anchor or anchor['arm'] != 'combined':
        raise ValueError('Expected the existing combined recommendation.')
    dependencies = {}
    def depend(path):
        dependencies[str(path)] = digest_file(path)
    for path in [previous/'manifest.json', previous/'decisions/recommendation.json']:
        depend(path)
    for arm, records in historical.items():
        if not records:
            raise ValueError(f'Historical evidence missing: {arm}')
        depend(previous/'diagnostics'/f'{arm}_checkpoints.json')
        for record in records:
            path = Path(record['checkpoint'])
            if digest_file(path) != record['identity']['checkpoint_sha256']:
                raise ValueError('Historical checkpoint mismatch.')
            sweep.validate_identity(torch.load(path,map_location='cpu',weights_only=False),old)
            depend(path)
            out = previous/'diagnostics'/record['id']
            if not sweep.reusable(out/'completion.json', record['identity']):
                raise ValueError('Historical evaluation is incomplete.')
            for p in out.rglob('*'):
                if p.is_file():
                    depend(p)
        directory = checkpoint_directory(Path(records[0]['checkpoint']))
        for p in directory.glob('*'):
            if p.is_file():
                depend(p)
    root.mkdir(parents=True, exist_ok=False)
    for name in ('configs','arms','stages','logs','diagnostics','decisions','source','verification'):
        (root/name).mkdir()
    frozen = ['splits.npz','probe_train_rows.npy','panel_rows.npy','samples.npz','validation_rows.csv']
    for name in frozen:
        shutil.copyfile(previous/name, root/name)
        depend(previous/name)
    (root/'root_config_original.yaml').write_text(yaml.safe_dump(read_config(source), sort_keys=False))
    c = make_config(c,root,'source',(0.,0.,0.))
    c['analysis']['exceedance_thresholds'] = old['cd8_thresholds']
    (root/'source_config.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
    frozen += ['source_config.yaml','root_config_original.yaml']
    repository = Path(__file__).resolve().parents[2]
    sources = {}
    for path in sorted((repository/'dann').rglob('*.py')) + [source]:
        relative = Path('dann/config.yaml') if path == source else path.relative_to(repository)
        dest = root/'source'/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        if path == source:
            shutil.copyfile(root/'root_config_original.yaml', dest)
        else:
            shutil.copyfile(path,dest)
        sources[str(relative)] = digest_file(dest)
    m = copy.deepcopy(old)
    m.update(version=3, created_utc=datetime.now(timezone.utc).isoformat(), previous=str(previous),
        sources=sources, files={n:digest_file(root/n) for n in frozen}, dependencies=dependencies,
        historical=historical, anchor=anchor, root_config=str(source), root_config_sha256=digest_file(source),
        # Tissue is a historical dependency only; every new stage reads lesion context.
        inputs={'path':_file_identity(Path(c['data']['path']))},
        no_he=dict(nhsic=NHSIC, anova=ANOVA, soft_nn=SOFT_NN, architectures=['cnn','gatv2'],
            inference=False, spatial_exports=False, arrays='memory only',
            eligibility='both new supervised-selected zero-alignment control and original nHSIC 0.1',
            selection='lowest eligible latent excess; ties within 0.01 use UMAP, CD8 loss, stable ID',
            additions='eligible and strictly improve both mixing measures over preceding winner',
            sampling='exact row coverage across architectures; order and boundaries matched within architecture'))
    immutable_json(root/'manifest.json',json_clean(m))
    for weight in NHSIC:
        register_arm(root,(weight,0.,0.))
    # Register spatial controls early so all three smokes precede full fitting.
    for architecture in ('cnn','gatv2'):
        register_arm(root,(0.,0.,0.),architecture)
    print(f'Prepared H&E-disabled study: {root}',flush=True)


def active_source(root: Path) -> Path:
    """Validate an append-only recovery chain without changing the base snapshot."""
    source = root/'source'
    parent = 'base'
    for path in sorted((root/'recoveries').glob('*/provenance.json')):
        record = json.loads(path.read_text())
        if (record['parent'] != parent or record['manifest_sha256'] != digest_file(root/'manifest.json')):
            raise ValueError('Recovery provenance chain changed.')
        source = path.parent/'source'
        for relative,sha in record['sources'].items():
            if digest_file(source/relative) != sha:
                raise ValueError(f'Recovery source changed: {relative}')
        for relative,sha in record.get('evidence',{}).items():
            if digest_file(path.parent/relative) != sha:
                raise ValueError(f'Recovery evidence changed: {relative}')
        parent = digest_file(path)
    return source


def freeze_recovery(root: Path, reason: str) -> None:
    """Append a reviewable runner/report repair; preserve all prior evidence.

    Training, extraction, configuration construction and selection functions
    must retain identical syntax trees, apart from docstrings. Repairs to those require a separately
    designed scientific recovery, rather than this reporting-only mechanism.
    """
    import ast
    import difflib
    manifest_for(root)
    previous = active_source(root)
    repository = Path(__file__).resolve().parents[2]
    relative = Path('dann/verification/alignment_no_he.py')
    old,new = (previous/relative).read_text(),(repository/relative).read_text()
    def functions(content):
        result = {}
        for node in ast.parse(content).body:
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                if ast.get_docstring(node) is not None:
                    node.body = node.body[1:]
                result[node.name] = ast.dump(node,include_attributes=False)
        return result
    before,after = functions(old),functions(new)
    def module_settings(content):
        return [ast.dump(node,include_attributes=False) for node in ast.parse(content).body
                if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))]
    if module_settings(old) != module_settings(new):
        raise ValueError('Reporting recovery may not change module settings or imports.')
    allowed = {'active_source','freeze_recovery','execution_provenance','manifest_for',
               'run_subprocess','main','report','verify_suite','execute_stage'}
    if any(before.get(name) != after.get(name) for name in (before.keys()|after.keys())-allowed):
        raise ValueError('Reporting recovery may not change scientific functions.')
    with (root/'runner.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Cannot activate a recovery while the runner is active.') from error
        if active_source(root) != previous:
            raise ValueError('Recovery source changed during validation.')
        revisions = root/'recoveries'
        revisions.mkdir(exist_ok=True)
        revision = revisions/f'{len(list(revisions.glob("*/provenance.json")))+1:04d}'
        revision.mkdir(exist_ok=False)
        shutil.copytree(previous,revision/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache'))
        changed = {}
        for relative in (Path('dann/verification/alignment_no_he.py'),Path('dann/tests/test_alignment_no_he.py')):
            original,updated = previous/relative,repository/relative
            changed[str(relative)] = dict(before=digest_file(original),after=digest_file(updated))
            shutil.copyfile(updated,revision/'source'/relative)
        patch_text = ''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
                             fromfile=str(previous/'dann/verification/alignment_no_he.py'),
                             tofile=str(revision/'source/dann/verification/alignment_no_he.py')))
        (revision/'changes.patch').write_text(patch_text)
        sources = {str(p.relative_to(revision/'source')):digest_file(p)
                   for p in (revision/'source').rglob('*') if p.is_file()}
        previous_provenance = previous.parent/'provenance.json'
        parent = digest_file(previous_provenance) if previous_provenance.exists() else 'base'
        immutable_json(revision/'provenance.json',dict(reason=reason,parent=parent,
            created_utc=datetime.now(timezone.utc).isoformat(),manifest_sha256=digest_file(root/'manifest.json'),
            scientific_functions_unchanged=True,scientific_configurations_unchanged=True,
            changed=changed,sources=sources,evidence={'changes.patch':digest_file(revision/'changes.patch')}))
        print(f'Frozen recovery snapshot: {revision}',flush=True)


def execution_provenance(root: Path) -> dict:
    """Identify the exact immutable source revision used for new stage execution."""
    source = Path(__file__).resolve().parents[2]
    path = source.parent/'provenance.json'
    return dict(source=str(source),revision_sha256=digest_file(path) if path.exists() else None)


def manifest_for(root: Path) -> dict:
    """Verify sources, frozen inputs/configs, and unchanged historical artifacts."""
    m = sweep.manifest_for(root)
    active_source(root)
    for path,digest in m['dependencies'].items():
        if digest_file(Path(path)) != digest:
            raise ValueError(f'Historical dependency changed: {path}')
    return m


def checkpoint_directory(path: Path) -> Path:
    """Resolve the model directory for an archive or named checkpoint."""
    return path.parent.parent if path.parent.name == 'archives' else path.parent


def checkpoint_paths(root: Path, arm: str) -> list[Path]:
    """Every distinct completed archive/best/latest; failure snapshots are excluded."""
    directory = Path(config_for(root,arm)['training']['output_dir'])
    result, seen = [], {}
    for path in [directory/'best.pt',directory/'latest.pt',*sorted((directory/'archives').glob('epoch_*.pt'))]:
        if not path.exists():
            continue
        ck = torch.load(path,map_location='cpu',weights_only=False)
        checkpoint_config(ck)
        epoch = ck['epoch']+1
        state = fingerprint({k:fingerprint(v.numpy()) for k,v in ck['model_state'].items()})
        if epoch in seen and seen[epoch] != state:
            raise ValueError('Completed checkpoints disagree at the same epoch.')
        if epoch not in seen:
            result.append(path)
        seen[epoch] = state
    return sorted(result,key=lambda p:torch.load(p,map_location='cpu',weights_only=False)['epoch'])


def validate_scientific_config(saved: dict, requested: dict) -> None:
    """Reject silent optimizer, schedule, sampling or objective changes on resume."""
    for key in ('model','data','loss','training'):
        left,right = copy.deepcopy(saved[key]),copy.deepcopy(requested[key])
        if key == 'training':
            left.pop('resume_checkpoint',None)
            right.pop('resume_checkpoint',None)
        if left != right:
            raise ValueError(f'Checkpoint scientific configuration mismatch: {key}')


def training_observer(root: Path, arm: str, m: dict):
    """Match fresh initialization within each architecture and all frozen splits."""
    architecture = config_for(root,arm)['model']['aggregation']['type']
    def observe(event, model, bundle, config, record):
        if event not in ('parameters_initialized','heads_initialized'):
            return
        for key,rows in bundle.split_indices.items():
            if fingerprint(rows) != m['splits'][key]['sha256']:
                raise ValueError('Training split identity changed.')
        value = dict(parameter_sha256=parameter_hash(model), record=record)
        decision(root, architecture+'_'+event, value)
        atomic_json(root/'diagnostics'/f'{arm}_{event}.json',value)
    return observe


def validate_boundary(root: Path, arm: str, m: dict) -> dict:
    """Validate stopping, archives, exact coverage and within-architecture sampling."""
    c = config_for(root,arm)
    directory = Path(c['training']['output_dir'])
    latest = torch.load(directory/'latest.pt',map_location='cpu',weights_only=False)
    epoch = latest['epoch']+1
    if not (epoch == c['training']['epochs'] or
            latest['early_stopping_state']['stale_epochs'] >= c['training']['early_stopping_patience']):
        raise ValueError('Training has not reached its configured stopping boundary.')
    paths = checkpoint_paths(root,arm)
    for path in paths:
        saved = torch.load(path,map_location='cpu',weights_only=False)
        validate_checkpoint(saved,c)
        sweep.validate_identity(saved,m)
        validate_scientific_config(saved['config'],c)
        if not saved.get('training_rng_state') or saved['training_progress']['update_in_epoch']:
            raise ValueError('Checkpoint lacks exact completed-epoch restart state.')
        if not all(torch.isfinite(v).all() for v in saved['model_state'].values()):
            raise FloatingPointError('Nonfinite checkpoint weights.')
    archives = {int(p.stem.split('_')[-1]) for p in (directory/'archives').glob('epoch_*.pt')}
    if archives != set(range(5,epoch+1,5)):
        raise ValueError('Missing or unexpected five-epoch archives.')
    rows = np.sort(latest['split_indices']['train']).astype('<i8')
    coverage = hashlib.sha256(rows.tobytes()).hexdigest()
    architecture = c['model']['aggregation']['type']
    controls = [p for p in (root/'arms').glob('*.json') if json.loads(p.read_text())['architecture'] == architecture]
    sampling_files = []
    for e in range(epoch):
        path = directory/'sampling'/f'epoch_{e:04d}_summary.json'
        sample = json.loads(path.read_text())
        if (not sample['complete'] or sample['processed_rows'] != len(rows) or
                sample['unique_rows'] != len(rows) or sample['row_coverage_sha256'] != coverage):
            raise ValueError('Training core row coverage differs from the frozen split.')
        for control in controls:
            other = Path(json.loads(control.read_text())['directory'])/'model/sampling'/path.name
            if other.exists():
                old = json.loads(other.read_text())
                if any(old[k] != sample[k] for k in ('row_order_sha256','boundaries_sha256')):
                    raise ValueError('Within-architecture sampling changed.')
        sampling_files.append(str(path))
    best = torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
    return dict(stopped_epoch=epoch,supervised_selected_epoch=best['epoch']+1,
        files=list(dict.fromkeys([str(directory/'best.pt'),str(directory/'latest.pt'),*map(str,paths),
                                  str(directory/'history.csv'),*sampling_files])))


def bounded_smoke(root: Path, c: dict) -> dict:
    """Bounded real lesion training and ordered extraction without H&E reads."""
    c = copy.deepcopy(c)
    # Each architecture owns its fixture so no completed smoke evidence is overwritten.
    fixture = root/'verification'/c['model']['aggregation']['type']
    fixture.mkdir(exist_ok=True)
    sweep.smoke_inputs(fixture,c,include_he=False,lesion_only=True,spatial_regions=True)
    c['data']['path'] = c['data']['context_path'] = str(fixture/'smoke_data/path.h5ad')
    c = no_exports(apply_smoke_overrides(c))
    c['training'].update(archive_checkpoint_every=1,resume_checkpoint=None)
    c['analysis']['figure_dpi'] = 50
    c['performance']['async_artifacts'] = False
    assert_disabled(c)
    path = train_model(c)
    ck = torch.load(path,map_location='cpu',weights_only=False)
    validate_checkpoint(ck,c)
    history = pd.read_csv(path.parent/'history.csv')
    if any('he_mse' in k or 'he_r2' in k for k in history):
        raise ValueError('Disabled H&E produced performance metrics.')
    for name in ('train_total_loss','validation_total_loss','validation_supervised_objective'):
        if not np.isfinite(history[name]).all():
            raise FloatingPointError(f'Nonfinite smoke {name}.')
    bundle = create_data_bundle(c,ck['split_indices'],ck['data_identity'],requested_splits=('validation',))
    try:
        model = AdversarialLatentFusion.from_config(c,len(ck['batch_names']),4).to(c['training']['device']).eval()
        model.load_state_dict(ck['model_state'])
        rows = ck['split_indices']['validation'][::-1].copy()
        data, he = sweep.extract_rows(model,c,bundle,rows,he=True)
        np.testing.assert_array_equal(data['rows'],rows)
        if he != dict(available=False,reason='head_disabled') or bundle.he_reader is not None:
            raise ValueError('Smoke accessed H&E targets.')
    finally:
        close(bundle)
    archive = path.parent/'archives/epoch_0001.pt'
    if not archive.exists() or not ck.get('training_rng_state'):
        raise ValueError('Smoke lacks restartable archives.')
    return dict(files=[str(path),str(archive),str(path.parent/'history.csv'),str(fixture/'smoke_data/provenance.json')],
                validation_rows=len(rows),inference=False,spatial_exports=False,he=he)


def reference_valid(reference: dict | None) -> bool:
    """Require complete, finite, noncollapsed evidence before using a reference."""
    if reference is None:
        return False
    try:
        return bool(eligibility(reference,reference)['eligible'])
    except (KeyError,TypeError,ValueError):
        return False


def dual_eligibility(candidate: dict, control: dict, reference: dict) -> dict:
    """Apply every original biological/CD8 tolerance against both references."""
    checks = {name:eligibility(candidate,r) if reference_valid(r) else
              dict(eligible=False,checks=[dict(metric='reference_valid',passed=False)],
                   reason='Reference cannot establish valid comparisons.')
              for name,r in [('control',control),('reference',reference)]}
    return dict(eligible=all(v['eligible'] for v in checks.values()),references=checks)


def choose(candidates: list[dict], control: dict, reference: dict, *, base: dict | None = None) -> dict | None:
    """Initial selection needs eligibility; additions also improve both mixing scores."""
    values = [r for r in candidates if dual_eligibility(r,control,reference)['eligible']
              and (base is None or improves(r,base))]
    if not values:
        return None
    minimum = min(r['mixing']['latent']['excess'] for r in values)
    tied = [r for r in values if r['mixing']['latent']['excess'] <= minimum+LIMITS['latent_tie']+1e-12]
    return min(tied,key=lambda r:(r['mixing']['umap']['excess'],r['distributions']['cd8']['loss'],r['id']))


def full_umap(root: Path, selected: dict, m: dict, c: dict) -> dict:
    """Save full-validation coordinates, colors and images, with predictions in memory."""
    path = Path(selected['checkpoint'])
    ck = torch.load(path,map_location='cpu',weights_only=False)
    model = AdversarialLatentFusion.from_config(checkpoint_config(ck),len(m['batch_names']),4).cuda().eval()
    model.load_state_dict(ck['model_state'])
    bundle = create_data_bundle(c,ck['split_indices'],ck['data_identity'],requested_splits=('validation',))
    out = root/'diagnostics'/selected['id']/'full_validation'
    out.mkdir(exist_ok=True)
    try:
        data,_ = sweep.extract_rows(model,c,bundle,ck['split_indices']['validation'])
        coords = sweep.fit_umap(data['latent'])
        if not np.isfinite(coords).all():
            raise FloatingPointError('Nonfinite full-validation UMAP.')
        np.savez_compressed(out/'umap.npz',rows=data['rows'],umap=coords)
        sweep.plot_umap(coords,data['rows'],data,bundle.metadata,m,out/'umap.png',selected['id']+' full validation')
        atomic_json(out/'provenance.json',dict(checkpoint_sha256=digest_file(path),rows=len(data['rows']),
            rows_sha256=fingerprint(data['rows']),settings=m['protocol']['umap']))
    finally:
        close(bundle)
    return dict(files=[str(p) for p in out.iterdir() if p.is_file()])


def execute_stage(root: Path, arm: str, stage: str) -> None:
    """Execute or reuse a stage; scientific failures are terminal exclusions."""
    if stage not in STAGES:
        raise ValueError('Unknown stage.')
    m, c = manifest_for(root), config_for(root,arm)
    identity = dict(manifest_sha256=digest_file(root/'manifest.json'),arm=arm,stage=stage,config=fingerprint(c))
    if stage == 'evaluate':
        paths = checkpoint_paths(root,arm)
        if not paths:
            raise ValueError('No completed checkpoints to evaluate.')
        identity['checkpoints'] = {str(p):digest_file(p) for p in paths}
    if stage == 'full_umap':
        selected = json.loads((root/'decisions/selected_epochs.json').read_text())[arm]
        identity['checkpoint'] = digest_file(Path(selected['checkpoint']))
    stamp = root/'stages'/f'{arm}-{stage}.json'
    if sweep.reusable(stamp,identity):
        print(f'Reusing {arm} {stage}',flush=True)
        return
    start = time.monotonic()
    atomic_json(stamp,dict(status='running',identity=identity))
    try:
        with training_validation_only():
            if stage == 'smoke':
                result = bounded_smoke(root,c)
            elif stage == 'train':
                latest = Path(c['training']['output_dir'])/'latest.pt'
                if latest.exists():
                    saved = torch.load(latest,map_location='cpu',weights_only=False)
                    validate_checkpoint(saved,c)
                    sweep.validate_identity(saved,m)
                    validate_scientific_config(saved['config'],c)
                    if not saved.get('training_rng_state'):
                        raise ValueError('Resume scientific configuration or RNG state mismatch.')
                    c['training']['resume_checkpoint'] = str(latest)
                train_model(c,observer=training_observer(root,arm,m))
                result = validate_boundary(root,arm,m)
            elif stage == 'evaluate':
                saved = torch.load(paths[0],map_location='cpu',weights_only=False)
                bundle = create_data_bundle(c,saved['split_indices'],saved['data_identity'],requested_splits=('validation',))
                best = torch.load(Path(c['training']['output_dir'])/'best.pt',map_location='cpu',weights_only=False)['epoch']+1
                try:
                    records = []
                    for path in paths:
                        record = sweep.evaluate_checkpoint(root,arm,path,m,c,bundle)
                        record['supervised_selected'] = record['epoch'] == best
                        records.append(record)
                finally:
                    close(bundle)
                dest = root/'diagnostics'/f'{arm}_checkpoints.json'
                atomic_json(dest,records)
                result = dict(files=[str(dest),*[str(root/'diagnostics'/r['id']/'completion.json') for r in records]])
            else:
                result = full_umap(root,selected,m,c)
        result['execution'] = execution_provenance(root)
        sweep.complete(stamp,identity,result,start)
    except (FloatingPointError,AlignmentEvaluationError) as error:
        if stage not in ('train','evaluate'):
            raise
        directory = Path(c['training']['output_dir'])
        exclusion = root/'diagnostics'/f'{arm}_exclusion.json'
        atomic_json(exclusion,dict(arm=arm,stage=stage,reason=str(error),selectable=False,
            traceback=traceback.format_exc(),failure_snapshots=[str(p) for p in (directory/'failures').rglob('snapshot.pt')]))
        empty = root/'diagnostics'/f'{arm}_checkpoints.json'
        atomic_json(empty,[])
        sweep.complete(stamp,identity,dict(excluded=True,files=[str(exclusion),str(empty)],execution=execution_provenance(root)),start)
    except BaseException as error:
        atomic_json(stamp,dict(status='failed',identity=identity,error=repr(error),traceback=traceback.format_exc()))
        raise


def run_subprocess(root: Path, arm: str, stage: str) -> None:
    """One GPU subprocess at a time, importing only frozen sources."""
    print(f'Starting {arm} {stage}',flush=True)
    with (root/'logs'/f'{arm}-{stage}.log').open('a') as log:
        subprocess.run([sys.executable,'-m','dann.verification.alignment_no_he','--output',str(root),
            '--arm',arm,'--stage',stage],cwd=active_source(root),check=True,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
                     NUMBA_NUM_THREADS='4',MPLBACKEND='Agg',PYTHONUNBUFFERED='1'))


def verify_suite(root: Path) -> None:
    """Require a successful frozen DANN synthetic suite before real fitting."""
    identity = dict(manifest_sha256=digest_file(root/'manifest.json'),stage='synthetic')
    stamp = root/'stages/synthetic.json'
    if sweep.reusable(stamp,identity):
        return
    start = time.monotonic()
    log = root/'logs/synthetic.log'
    with log.open('w') as handle:
        subprocess.run([sys.executable,'-m','pytest','-q','dann/tests'],cwd=active_source(root),check=True,
            stdout=handle,stderr=subprocess.STDOUT,env=dict(os.environ,PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
                OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',MPLBACKEND='Agg'))
    sweep.complete(stamp,identity,dict(files=[str(log)]),start)


def records_for(root: Path, arm: str) -> list[dict]:
    """Read completed checkpoint metrics, excluding terminally failed arms."""
    if (root/'diagnostics'/f'{arm}_exclusion.json').exists():
        return []
    return sweep.records_for(root,arm)


def finish(root: Path, m: dict, control: dict | None, reference: dict, winner: dict | None, reason: str) -> None:
    """Freeze distinct supervised/mixing selections and render every completed arm."""
    selected, supervised, mixing, unavailable = {}, {}, {}, {}
    candidates = []
    for path in sorted((root/'arms').glob('*.json')):
        arm = path.stem
        records = records_for(root,arm)
        if not records:
            continue
        candidates.extend(records)
        best = next(r for r in records if r['supervised_selected'])
        choice = choose(records,control,reference) if control else None
        supervised[arm] = best
        mixing[arm] = choice
        display = choice or best
        if display['spectrum']['noncollapsed']:
            selected[arm] = display
        else:
            unavailable[arm] = display.get('invalid_reason','Numerically collapsed latent representation.')
    replacement = choose(candidates,control,reference,base=m['anchor']) if winner else None
    decision(root,'selected_epochs',selected)
    decision(root,'supervised_epochs',supervised)
    decision(root,'mixing_epochs',mixing)
    if unavailable:
        decision(root,'unavailable_umaps',unavailable)
    decision(root,'recommendation',dict(selected=winner,replacement=replacement,
        retained_recommendation=replacement or m['anchor'],
        replaced_combined=replacement is not None,reason=reason,architecture_transfer=winner is not None))
    for arm in selected:
        run_subprocess(root,arm,'full_umap')
    report(root,finished=True)
    audit(root)


def run_all(root: Path, *, through: str = 'architectures') -> None:
    """Run sequential MLP gates then transfer frozen weights to spatial models."""
    m = manifest_for(root)
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_suite(root)
        for architecture in ('mlp','cnn','gatv2'):
            run_subprocess(root,arm_name(architecture,(0.,0.,0.)),'smoke')
        reference = next(r for r in m['historical']['reference'] if r['supervised_selected'])
        def fit(weights,architecture='mlp'):
            arm = register_arm(root,weights,architecture)
            run_subprocess(root,arm,'train')
            if not (root/'diagnostics'/f'{arm}_exclusion.json').exists():
                run_subprocess(root,arm,'evaluate')
            report(root)
            return records_for(root,arm)
        controls = fit((0.,0.,0.))
        control = next((r for r in controls if r['supervised_selected']),None)
        if not reference_valid(control) or not reference_valid(reference):
            finish(root,m,control,reference,None,'Stopped: control/reference cannot establish valid comparisons.')
            return
        candidates = list(controls)
        for weight in NHSIC[1:]:
            candidates.extend(fit((weight,0.,0.)))
        winner = choose(candidates,control,reference)
        decision(root,'nhsic',dict(selected=winner,control=control))
        if winner is None:
            finish(root,m,control,reference,None,'Stopped: no MLP nHSIC candidate passes both biological/CD8 references.')
            return
        if through == 'nhsic':
            report(root)
            return
        weights = tuple(json.loads((root/'arms'/f'{winner["arm"]}.json').read_text())['weights'])
        for phase,index,grid in [('anova',1,ANOVA),('soft_nn',2,SOFT_NN)]:
            candidates = []
            for weight in grid:
                proposed = list(weights); proposed[index] = weight
                candidates.extend(fit(tuple(proposed)))
            preceding = winner
            winner = choose(candidates,control,reference,base=preceding) or preceding
            weights = tuple(json.loads((root/'arms'/f'{winner["arm"]}.json').read_text())['weights'])
            decision(root,phase,dict(selected=winner,preceding=preceding,weights=weights,adopted=winner['id'] != preceding['id']))
            if through == phase:
                report(root)
                return
        decision(root,'frozen_mlp',dict(selected=winner,weights=weights))
        for architecture in ('cnn','gatv2'):
            own_control = fit((0.,0.,0.),architecture)
            aligned = own_control if weights == (0.,0.,0.) else fit(weights,architecture)
            baseline = next((r for r in own_control if r['supervised_selected']),None)
            valid_baseline = reference_valid(baseline)
            choice = choose(aligned,control,reference) if valid_baseline else None
            comparison = dict(control=baseline,selected=choice,
                improves_own_control=bool(choice and improves(choice,baseline)),weights=weights)
            if not valid_baseline:
                comparison.update(comparison_available=False,
                    unavailable_reason='Architecture control cannot establish valid comparisons.')
            decision(root,architecture,comparison)
        finish(root,m,control,reference,winner,'Completed MLP alignment selection and frozen-weight architecture comparison.')


def report(root: Path, *, finished: bool = False) -> None:
    """Write linked metrics, matched epochs and component-only historical comparisons."""
    m = manifest_for(root)
    previous = Path(m['previous'])
    records = [r for p in sorted((root/'arms').glob('*.json')) for r in records_for(root,p.stem)]
    reference = next(r for r in m['historical']['reference'] if r['supervised_selected'])
    control = next((r for r in records if r['arm'] == arm_name('mlp',(0.,0.,0.)) and r['supervised_selected']),None)
    selected_path = root/'decisions/selected_epochs.json'
    selected = json.loads(selected_path.read_text()) if selected_path.exists() else {}
    mixing_path = root/'decisions/mixing_epochs.json'
    mixing = json.loads(mixing_path.read_text()) if mixing_path.exists() else {}
    tables = {n:[] for n in ('checkpoint_comparison','validation_targets','biological_probes','eligibility_limits',
        'training_history','mixing_samples','mixing_per_slide','stratified_probes','variance_spectra','gradient_alignment')}
    panels = []
    all_records = records + [r for values in m['historical'].values() for r in values]
    for r in all_records:
        historical = r['arm'] in HISTORICAL
        out = (previous if historical else root)/'diagnostics'/r['id']
        check = dual_eligibility(r,control,reference) if control else None
        primary = dict(id=r['id'],arm=r['arm'],epoch=r['epoch'],historical=historical,
            supervised_selected=r['supervised_selected'],
            mixing_selected=(mixing.get(r['arm']) or {}).get('id')==r['id'],
            umap_selected=selected.get(r['arm'],{}).get('id')==r['id'],
            eligible=check['eligible'] if check else None,he_available=historical,
            cd8_loss=r['distributions'].get('cd8',{}).get('loss'),
            **{f'{s}_{k}':v for s,values in r['mixing'].items() for k,v in values.items()})
        for space in ('raw','normalized'):
            spec = r['spectrum'].get(space,{})
            primary[space+'_participation_ratio'] = spec.get('participation_ratio')
            primary[space+'_leading_fraction'] = spec.get('leading_fraction')
            if primary[space+'_leading_fraction'] is None and spec.get('eigenvalues'):
                primary[space+'_leading_fraction'] = spec['eigenvalues'][0]/sum(spec['eigenvalues'])
            eig = spec.get('eigenvalues',[])
            tables['variance_spectra'].extend(dict(id=r['id'],space=space,component=i+1,eigenvalue=v,
                fraction=v/sum(eig) if sum(eig)>0 else None) for i,v in enumerate(eig))
        tables['checkpoint_comparison'].append(primary)
        if check:
            tables['eligibility_limits'].extend(dict(id=r['id'],reference_name=n,**v)
                for n,result in check['references'].items() for v in result['checks'])
        history = next(iter(r['training_history']),{})
        for target,values in r['distributions'].items():
            tables['validation_targets'].append(dict(id=r['id'],target=target,loss=values['loss'],
                presence_brier=values['presence']['brier'],positive_pit_ks=values.get('positive_pit_ks'),
                **{k:v for k,v in history.items() if k.startswith('validation_') and target in k.lower()},
                **{f'coverage_{level}':v['coverage'] for level,v in values['intervals'].items()},
                **{f'exceedance_{t}_brier':v['brier'] for t,v in values.get('exceedance',{}).items()}))
        tables['biological_probes'].extend(dict(id=r['id'],probe=n,**v) for n,v in r['probes'].items())
        tables['stratified_probes'].extend(dict(id=r['id'],stratum=q,probe=n,**v)
            for q,values in r.get('stratified_probes',{}).items() for n,v in values.items())
        tables['gradient_alignment'].append(dict(id=r['id'],**{k:v for k,v in history.items() if 'gradient' in k or 'alignment' in k}))
        for name in ('mixing_samples','mixing_per_slide'):
            if (out/f'{name}.csv').exists():
                tables[name].extend(dict(id=r['id'],**v) for v in pd.read_csv(out/f'{name}.csv').to_dict('records'))
        if selected.get(r['arm'],{}).get('id') == r['id']:
            links = []
            for suffix in ('metrics.json','panel.png','full_validation/umap.png','full_validation/umap.npz'):
                p = out/suffix
                if p.exists():
                    links.append(f'<a href="{os.path.relpath(p,root)}">{suffix}</a>')
            panels.append(f'<h3>{html.escape(r["id"])}</h3><p>{" · ".join(links)}</p>')
    for arm in sorted({r['arm'] for r in all_records}):
        r = next(r for r in all_records if r['arm']==arm)
        history = pd.read_csv(checkpoint_directory(Path(r['checkpoint']))/'history.csv')
        # H&E-disabled histories have no H&E performance columns. Direct historical
        # comparisons use biology/CD8 components, never total supervised loss.
        tables['training_history'].extend(dict(arm=arm,**v) for v in history.to_dict('records'))
    frames = {n:pd.DataFrame(values) for n,values in tables.items()}
    history = frames['training_history']
    comparisons = []
    for weight,old_arm in [(0.,'control'),(.1,'reference')]:
        new_arm = arm_name('mlp',(weight,0.,0.))
        if history.empty or new_arm not in set(history.arm):
            continue
        cols = [k for k in history if k=='epoch' or k.startswith('validation_') and
                ('biology' in k or 'cd8' in k.lower() or any(n in k.lower() for n in ('tumor','collagen','stroma')))]
        old_h = history[history.arm==old_arm][cols]
        new_h = history[history.arm==new_arm][cols]
        common = old_h.merge(new_h,on='epoch',suffixes=('_he_enabled','_he_disabled'))
        comparisons.extend(dict(weight=weight,comparison='common_epoch',**v) for v in common.to_dict('records'))
        old_best = next(r for r in m['historical'][old_arm] if r['supervised_selected'])
        new_best = next((r for r in records if r['arm']==new_arm and r['supervised_selected']),None)
        if new_best:
            row = dict(weight=weight,comparison='supervised_selected',epoch_he_enabled=old_best['epoch'],epoch_he_disabled=new_best['epoch'])
            for label,frame,record in [('he_enabled',old_h,old_best),('he_disabled',new_h,new_best)]:
                row.update({k+'_'+label:v for k,v in frame[frame.epoch==record['epoch']].iloc[0].items() if k!='epoch'})
            comparisons.append(row)
    frames['he_component_comparisons'] = pd.DataFrame(comparisons)
    architecture_comparisons = []
    frozen_path = root/'decisions/frozen_mlp.json'
    if frozen_path.exists():
        weights = tuple(json.loads(frozen_path.read_text())['weights'])
        for architecture in ('cnn','gatv2'):
            baseline = [r for r in records if r['arm']==arm_name(architecture,(0.,0.,0.))]
            aligned = [r for r in records if r['arm']==arm_name(architecture,weights)]
            pairs = [('common_epoch',a,b) for a in baseline for b in aligned if a['epoch']==b['epoch']]
            pairs += [('supervised_selected',a,b) for a in baseline for b in aligned
                      if a['supervised_selected'] and b['supervised_selected']]
            for kind,a,b in pairs:
                available = reference_valid(a) and reference_valid(b)
                comparison = dict(architecture=architecture,comparison=kind,
                    control_id=a['id'],aligned_id=b['id'],control_epoch=a['epoch'],aligned_epoch=b['epoch'],
                    comparison_available=available)
                if available:
                    comparison.update(
                        eligible_against_own_control=eligibility(b,a)['eligible'],
                        latent_excess_delta=b['mixing']['latent']['excess']-a['mixing']['latent']['excess'],
                        umap_excess_delta=b['mixing']['umap']['excess']-a['mixing']['umap']['excess'],
                        cd8_loss_delta=b['distributions']['cd8']['loss']-a['distributions']['cd8']['loss'])
                else:
                    comparison.update(eligible_against_own_control=None,latent_excess_delta=None,
                        umap_excess_delta=None,cd8_loss_delta=None,
                        unavailable_reason='invalid_control' if not reference_valid(a) else 'invalid_aligned_checkpoint')
                architecture_comparisons.append(comparison)
    frames['architecture_comparisons'] = pd.DataFrame(architecture_comparisons)
    for name,frame in frames.items():
        frame.to_csv(root/f'{name}.csv',index=False)
    if not history.empty:
        plot_curves(root,history,frames['checkpoint_comparison'])
    recommendation_path = root/'decisions/recommendation.json'
    recommendation = json.loads(recommendation_path.read_text()) if recommendation_path.exists() else None
    exclusions = [json.loads(p.read_text()) for p in sorted((root/'diagnostics').glob('*_exclusion.json'))]
    links = ' · '.join(f'<a href="{name}.csv">{name.replace("_"," ")}</a>' for name in frames)
    decision_links = ' · '.join(f'<a href="decisions/{p.name}">{p.stem}</a>' for p in sorted((root/'decisions').glob('*.json')) if not p.stem.startswith('arm_'))
    body = f'''<!doctype html><meta charset="utf-8"><title>H&E-disabled alignment study</title>
<style>body{{font:15px system-ui;margin:30px}}table{{border-collapse:collapse;font-size:12px}}td,th{{padding:5px;border:1px solid #ddd}}.scroll{{overflow:auto}}</style>
<h1>H&E-disabled alignment study</h1><p>Complete: {finished}. {html.escape(recommendation['reason'] if recommendation else 'Study in progress.')}</p>
<p>H&E performance is unavailable for new arms; its targets are never loaded. Early stopping uses biology only.
Historical H&E-enabled total supervised objectives include another term, so compare biology and CD8 components directly.
Training and validation use lesion context. Tissue inference, spatial exports and all post-training workflows are disabled.</p>
<p><a href="manifest.json">Frozen protocol</a> · {links}</p><p>{decision_links}</p>
<p>Eligibility applies the original tolerances against both the new supervised-selected zero control and original nHSIC 0.1.
Initial selection minimizes latent excess; ties within 0.01 use UMAP excess, CD8 loss and stable ID.
ANOVA and soft-NN additions must improve both mixing scores. Spatial models receive frozen MLP weights and fresh initialization.
See architecture decisions for comparisons to their own controls. Supervised and mixing selections remain separate.</p>
<div class="scroll">{frames['checkpoint_comparison'].to_html(index=False)}</div>
<p>Raw/normalized participation ratio and leading-component concentration are diagnostics; only the original numerical-collapse checks exclude representations.
Exact neighbors exclude self and subtract chance; five balanced samples and training stromal quartiles are fixed.
Single training seed; diagnostic draws are not independent training replications. Validation selection is not an unbiased test estimate.</p>
<h2>Numerical/scientific exclusions</h2><pre>{html.escape(json.dumps(exclusions,indent=2))}</pre>
<h2>Selected full-validation UMAPs</h2>{''.join(panels)}'''
    if (root/'comparison_curves.png').exists():
        body += '<h2>Biology, CD8 and mixing curves</h2><img src="comparison_curves.png" width="100%">'
    if finished:
        images = sorted(p for p in root.rglob('*.png') if 'source' not in p.relative_to(root).parts)
        with zipfile.ZipFile(root/'images.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
            for p in images:
                archive.write(p,p.relative_to(root))
        body += '<p><a href="images.zip">Download image ZIP</a></p>'
    (root/'report.html').write_text(body)
    atomic_json(root/'status.json',dict(complete=finished,recommendation=recommendation,exclusions=exclusions,
        updated_utc=datetime.now(timezone.utc).isoformat()))


def plot_curves(root: Path, history: pd.DataFrame, frame: pd.DataFrame) -> None:
    """Use comparable biology/CD8 components; omit disabled H&E curves."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,3,figsize=(19,10),layout='constrained')
    for ax,key in zip(axes.flat[:3],('validation_biology_loss','validation_cd8_loss','validation_biology_weighted_loss')):
        for arm,group in history.groupby('arm'):
            if key in group:
                ax.plot(group.epoch,group[key],label=arm)
        ax.set(title=key,xlabel='Epoch')
    for ax,key in zip(axes.flat[3:],('latent_excess','umap_excess','raw_leading_fraction')):
        for arm,group in frame.groupby('arm'):
            if key in group:
                ax.plot(group.epoch,group[key],'.-',label=arm)
        ax.set(title=key,xlabel='Epoch')
    axes.flat[0].legend(fontsize=6)
    fig.savefig(root/'comparison_curves.png',dpi=140)
    plt.close(fig)


def audit(root: Path) -> None:
    """Check historical hashes, checkpoint receipts, UMAP rows/colors and report links."""
    import re
    m = manifest_for(root)
    if digest_file(Path(m['root_config'])) != m['root_config_sha256']:
        raise ValueError('User root configuration changed during the study.')
    if read_config(Path(m['root_config'])) != read_config(root/'root_config_original.yaml'):
        raise ValueError('User configuration settings changed during the study.')
    receipts = []
    for p in (root/'stages').glob('*.json'):
        receipt = json.loads(p.read_text())
        if receipt.get('status') != 'complete' or not sweep.reusable(p,receipt['identity']):
            raise ValueError(f'Incomplete stage: {p}')
        receipts.append(str(p))
    selected = json.loads((root/'decisions/selected_epochs.json').read_text())
    expected_rows = np.load(root/'splits.npz')['validation']
    colors = None
    for arm,r in selected.items():
        validate_boundary(root,arm,m)
        out = root/'diagnostics'/r['id']/'full_validation'
        with np.load(out/'umap.npz') as data:
            np.testing.assert_array_equal(data['rows'],expected_rows)
            if not np.isfinite(data['umap']).all():
                raise ValueError('Nonfinite UMAP coordinates.')
        palette = json.loads((out/'umap.colors.json').read_text())
        historical_palette = json.loads((Path(m['previous'])/'diagnostics'/m['anchor']['id']/'panel.colors.json').read_text())
        if palette != historical_palette or colors is not None and colors != palette:
            raise ValueError('UMAP slide palette changed.')
        colors = palette
    for link in re.findall(r'(?:href|src)="([^"]+)"',(root/'report.html').read_text()):
        if not (root/link).exists():
            raise ValueError(f'Broken report link: {link}')
    forbidden = [str(p) for p in root.rglob('*') if p.is_file() and
                 (p.suffix in ('.parquet','.feather') or p.name in ('latents.npy','probe_predictions.npz','probe_parameters.npz'))]
    if forbidden:
        raise ValueError(f'Forbidden exported arrays: {forbidden}')
    atomic_json(root/'audit.json',dict(complete=True,historical_hashes=True,root_config_unchanged=True,
        checkpoint_receipts=receipts,umap_identities_and_colors=True,report_links=True,forbidden_exports=[]))


def main() -> None:
    """Dispatch preparation and execution through immutable study sources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--previous',type=Path)
    parser.add_argument('--config',type=Path,default=Path('dann/config.yaml'))
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--recover-reporting',metavar='REASON')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--report',action='store_true')
    parser.add_argument('--audit',action='store_true')
    parser.add_argument('--through',choices=PHASES,default='architectures')
    parser.add_argument('--arm')
    parser.add_argument('--stage',choices=STAGES)
    args = parser.parse_args()
    root = args.output.resolve()
    torch.set_num_threads(4)
    if args.prepare:
        if not args.previous:
            parser.error('--prepare requires --previous')
        prepare(root,args.previous,args.config)
    if args.recover_reporting:
        freeze_recovery(root,args.recover_reporting)
    if args.run or args.stage or args.report or args.audit:
        manifest_for(root)
        if Path(__file__).resolve() != active_source(root)/'dann/verification/alignment_no_he.py':
            command = [sys.executable,'-m','dann.verification.alignment_no_he','--output',str(root),'--through',args.through]
            for option in ('run','report','audit'):
                if getattr(args,option): command.append('--'+option)
            if args.stage:
                if not args.arm: parser.error('--stage requires --arm')
                command += ['--arm',args.arm,'--stage',args.stage]
            subprocess.run(command,cwd=active_source(root),check=True)
            return
    if args.stage:
        if not args.arm: parser.error('--stage requires --arm')
        execute_stage(root,args.arm,args.stage)
    if args.run: run_all(root,through=args.through)
    if args.report:
        status = root/'status.json'
        report(root,finished=json.loads(status.read_text())['complete'] if status.exists() else False)
    if args.audit: audit(root)


if __name__ == '__main__':
    main()
