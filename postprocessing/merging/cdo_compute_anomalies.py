#!/usr/bin/env python3
"""
Python wrapper for computing anomaly fields relative to a base climatology
of raw model output.

For each requested variable/level, extracts the field from both the input
dataset and the climatology (treating both as raw multi-level model output),
time-averages the climatology, and saves the anomaly to the same directory
as the input dataset.

Usage:
    python compute_anomalies.py --dataset /path/to/data.nc \
        --clim /path/to/base_run/atmos_6_hourly_merged.nc \
        --vars vor:850 ps:surface
    or:
    python compute_anomalies.py --config config.yaml
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from cdo_utils import (
    PLEVELS,
    parse_level,
    parse_variable_spec,
    get_output_suffix,
    load_config,
    make_slurm_header,
    run_script,
)


class AnomalyCalculator:
    """
    Generates and optionally executes a CDO bash script that computes
    anomaly fields relative to a base climatology of raw model output.

    Both the input dataset and the climatology are treated as raw multi-level
    model output; level selection (sellevidx) is always applied to both.

    For each (variable, level) pair the generated script:
        1. Extracts the variable at the specified level from the dataset.
        2. Extracts the same variable at the same level from the climatology.
        3. Time-averages the climatology extract and broadcasts it to the
           dataset grid.
        4. Subtracts the climatology mean from the dataset field.
        5. Writes the anomaly NetCDF to the output directory.
    """

    def __init__(
        self,
        dataset_path: str,
        clim_path: str,
        variables: List[Tuple[str, Union[str, float, int]]],
        plevels: Optional[np.ndarray] = None,
        output_prefix: str = 'anomaly',
        output_dir: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        dataset_path : str
            Path to the input dataset NetCDF file.
        clim_path : str
            Path to the climatology NetCDF file (raw multi-level model output).
        variables : list of (str, level) tuples
            Variables to process, e.g. [('vor', 850), ('ps', 'surface')].
        plevels : np.ndarray, optional
            Model pressure levels in hPa. Defaults to cdo_utils.PLEVELS.
        output_prefix : str
            Prefix for output anomaly filenames. Default: 'anomaly'.
        output_dir : str, optional
            Directory for output files. Defaults to the dataset's parent
            directory.
        """
        self.dataset_path = Path(dataset_path)
        self.clim_path    = Path(clim_path)
        self.variables    = variables
        self.plevels      = plevels if plevels is not None else PLEVELS
        self.output_prefix = output_prefix
        self.output_dir   = (
            Path(output_dir) if output_dir else self.dataset_path.parent
        )

        # Validate paths eagerly so errors surface before script generation
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        if not self.clim_path.exists():
            raise FileNotFoundError(
                f"Climatology file not found: {self.clim_path}"
            )

        # Pre-compute level metadata for all requested variables
        self.parsed_vars = self._parse_all_variables()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_all_variables(self) -> List[Dict]:
        """Resolve level indices and output paths for all variables."""
        parsed = []
        for var_name, level in self.variables:
            level_idx, level_suffix = parse_level(level, self.plevels)
            var_suffix = get_output_suffix(var_name, level, self.plevels)
            parsed.append({
                'var_name':    var_name,
                'level_idx':   level_idx,
                'var_suffix':  var_suffix,
                'output_path': str(
                    self.output_dir / f"{self.output_prefix}_{var_suffix}.nc"
                ),
            })
        return parsed

    # ------------------------------------------------------------------
    # Script generation
    # ------------------------------------------------------------------

    def generate_bash_script(self) -> str:
        """Return a bash script string that performs all CDO operations."""

        lines = make_slurm_header(
            mem='16G',
            stdout='compute_anomaly.out',
            stderr='compute_anomaly.err',
            extra_directives=["-t 0-00:30"]
        )

        lines += [
            "# ---------------------------------------------------------------",
            "# Anomaly computation",
            f"# Dataset:     {self.dataset_path}",
            f"# Climatology: {self.clim_path}",
            f"# Output dir:  {self.output_dir}",
            "# ---------------------------------------------------------------",
            "",
            "# Unique temp directory - cleaned up automatically on exit",
            "TMPDIR=$(mktemp -d)",
            'trap "rm -rf $TMPDIR" EXIT',
            "",
        ]

        for v in self.parsed_vars:
            var_name  = v['var_name']
            level_idx = v['level_idx']
            vsuffix   = v['var_suffix']
            out       = v['output_path']
            
            if var_name == 'ps':
                lines += [
                    f"# ---- {var_name} at surface ----",
                    "",
                    f"# 1. Extract {var_name} from dataset",
                    f"cdo -selname,{var_name} \\",
                    f"    {self.dataset_path} \\",
                    f"    $TMPDIR/extracted_{vsuffix}.nc",
                    "",
                    "# 2. Anomaly = dataset field - climatology mean",
                    f"cdo -sub $TMPDIR/extracted_{vsuffix}.nc \\",
                    f"    {self.clim_path} \\",
                    f"    {out}",
                    "",
                    f'echo "Saved: {out}"',
                    "",
                ]
            else:
                lines += [
                    f"# ---- {var_name} at level index {level_idx} ----",
                    "",
                    f"# 1. Extract {var_name} at level {level_idx} from dataset",
                    f"cdo -sellevidx,{level_idx} -selname,{var_name} \\",
                    f"    {self.dataset_path} \\",
                    f"    $TMPDIR/extracted_{vsuffix}.nc",
                    "",
                    "# 2. Anomaly = dataset field - climatology mean",
                    f"cdo -sub $TMPDIR/extracted_{vsuffix}.nc \\",
                    f"    {self.clim_path} \\",
                    f"    {out}",
                    "",
                    f'echo "Saved: {out}"',
                    "",
                ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, submit_slurm: bool = True, dry_run: bool = False):
        """
        Execute or display the generated bash script.

        Parameters
        ----------
        submit_slurm : bool
            Submit via sbatch if True; run directly with bash if False.
        dry_run : bool
            Print the generated script and exit without running anything.
        """
        run_script(
            self.generate_bash_script(),
            submit_slurm=submit_slurm,
            dry_run=dry_run,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            'Compute CDO anomaly fields relative to a climatology '
            'of raw multi-level model output.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Command-line usage
  python %(prog)s --dataset /path/to/merged.nc \\
      --clim /path/to/base_run/merged.nc --vars vor:850 ps:surface

  # YAML config
  python %(prog)s --config config.yaml

  # Preview the generated script without running
  python %(prog)s --config config.yaml --dry-run

  # Run directly (bypass SLURM)
  python %(prog)s --config config.yaml --no-slurm
        """,
    )

    parser.add_argument('--config', type=str,
                        help='Path to YAML configuration file.')
    parser.add_argument('--dataset', type=str,
                        help='Path to the input dataset NetCDF file.')
    parser.add_argument('--clim', type=str,
                        help='Path to the climatology NetCDF file '
                             '(raw multi-level model output).')
    parser.add_argument('--vars', type=str, nargs='+',
                        help='Variables as "name:level" (e.g. vor:850 ps:surface).')
    parser.add_argument('--plevels-file', type=str,
                        help='Path to a .npy file containing the pressure '
                             'levels array.')
    parser.add_argument('--output-prefix', type=str, default='anomaly',
                        help='Prefix for output filenames (default: anomaly).')
    parser.add_argument('--output-dir', type=str,
                        help='Output directory (default: same as dataset).')
    parser.add_argument('--no-slurm', action='store_true',
                        help='Run directly instead of submitting to SLURM.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the generated script without executing.')

    args = parser.parse_args()

    # Optional external pressure levels
    plevels = None
    if args.plevels_file:
        plevels = np.load(args.plevels_file)
        print(f"Loaded {len(plevels)} pressure levels from {args.plevels_file}.")

    if args.config:
        cfg = load_config(args.config)
        dataset_path  = cfg['dataset']
        clim_path     = cfg['climatology']
        variables     = [parse_variable_spec(v) for v in cfg['variables']]
        output_prefix = cfg.get('output_prefix', 'anomaly')
        output_dir    = cfg.get('output_dir', None)
        if plevels is None and 'plevels' in cfg:
            plevels = np.array(cfg['plevels'])
    else:
        if not all([args.dataset, args.clim, args.vars]):
            parser.error(
                "Provide either --config or all of --dataset, --clim, --vars."
            )
        dataset_path  = args.dataset
        clim_path     = args.clim
        variables     = [parse_variable_spec(v) for v in args.vars]
        output_prefix = args.output_prefix
        output_dir    = args.output_dir

    calc = AnomalyCalculator(
        dataset_path=dataset_path,
        clim_path=clim_path,
        variables=variables,
        plevels=plevels,
        output_prefix=output_prefix,
        output_dir=output_dir,
    )

    calc.run(submit_slurm=not args.no_slurm, dry_run=args.dry_run)


if __name__ == '__main__':
    main()