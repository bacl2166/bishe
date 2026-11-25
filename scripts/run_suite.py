"""Run the matched three-domain Amazon experiment suite serially.

Each child uses this checkout as cwd so config data/output paths are stable even
when this script is launched from another directory. No download is performed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('musical', 'fashion', 'appliances')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=('1', '2', 'both'), default='both')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prepare-only', action='store_true', help='prepare experiment 1 only; requires --experiment 1')
    parser.add_argument('--config-dir', type=Path, default=Path('configs'),
                        help='config directory; relative paths are resolved against the repository root')
    args = parser.parse_args(argv)
    if args.prepare_only and args.experiment != '1':
        parser.error('--prepare-only requires --experiment 1; experiment 2 needs completed experiment 1 encodings')
    config_dir = args.config_dir if args.config_dir.is_absolute() else REPO_ROOT / args.config_dir
    experiments = (1, 2) if args.experiment == 'both' else (int(args.experiment),)
    for dataset in dict.fromkeys(args.datasets):
        for experiment in experiments:
            config_path = (config_dir / f'experiment{experiment}_{dataset}.json').resolve()
            try:
                config = json.loads(config_path.read_text(encoding='utf-8'))
                if experiment == 2:
                    source = config.get('source_run')
                    if not isinstance(source, str) or not source:
                        raise ValueError(f'{config_path}: experiment 2 requires source_run')
                    source_path = Path(source)
                    if not source_path.is_absolute():
                        source_path = REPO_ROOT / source_path
                    if not source_path.is_dir():
                        raise FileNotFoundError(f'{dataset}: missing experiment 2 source_run: {source_path}. '
                                                'Run experiment 1 for this dataset first.')
            except (OSError, ValueError) as error:
                parser.exit(1, f'run_suite: {error}\n')
            command = [sys.executable, '-m', f'exp{experiment}', '--config', str(config_path)]
            if args.resume:
                command.append('--resume')
            if args.prepare_only:
                command.append('--prepare-only')
            print(f'Running experiment {experiment}: {dataset} ({config_path.name})', flush=True)
            subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
