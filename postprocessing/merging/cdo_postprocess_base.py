#!/usr/bin/env python3
"""
Python wrapper for CDO-based GCM output postprocessing (courtesy of Claude Sonnet 4.5).
Allows flexible specification of variables and pressure levels to extract and merge.

Usage:
    python cdo_postprocess_base.py --config config.yaml
    or
    python cdo_postprocess_base.py --exproot /path/to/exp --chunks 1 10 \
        --vars ps:surface vor:850 temp:850 ucomp:850
"""

import argparse
import subprocess
import sys
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import tempfile

from cdo_utils import (
    PLEVELS,
    parse_variable_spec,
    load_config,
)

import pdb

class CDOPostProcessor:
    """Wrapper for CDO postprocessing operations."""
    
    def __init__(self, 
                 exproot: str, 
                 chunk_start: int, 
                 chunk_end: int,
                 variables: List[Tuple[str, Union[str, int, float]]], 
                 plevels: Optional[np.ndarray] = None,
                 compute_anomalies: bool = False,
                 input_filename: str = 'atmos_6_hourly.nc'):
        """
        Initialize the postprocessor.
        
        Parameters:
        -----------
        exproot : str
            Root directory containing run chunks (run0001, run0002, etc.)
        chunk_start : int
            First chunk number to process
        chunk_end : int
            Last chunk number to process
        variables : List[Tuple[str, Union[str, int, float]]]
            List of (variable_name, level) tuples
            Level can be:
              - String 'surface' or 'surf' for surface level
              - Integer for direct model level index (1-indexed as in CDO)
              - Float/int for pressure level in hPa (will find nearest)
            Examples: [('vor', 850), ('ps', 'surface'), ('temp', 22)]
        plevels : np.ndarray, optional
            Array of model pressure levels in hPa. If None, uses global PLEVELS.
        compute_anomalies : bool
            Whether to compute anomalies from time-mean for base_ps_vor file
        input_filename : str
            Name of input files in each chunk directory (default: 'atmos_6_hourly.nc')
        """
        self.exproot = Path(exproot)
        self.chunk_start = chunk_start
        self.chunk_end = chunk_end
        self.variables = variables
        self.plevels = plevels if plevels is not None else PLEVELS
        self.compute_anomalies = compute_anomalies
        self.input_filename = input_filename
        
        self.postproc_dir = self.exproot / 'postprocessed'
        self.postproc_dir.mkdir(exist_ok=True)
        
        # Validate variables
        self._validate_variables()
        
    def _validate_variables(self):
        """Check that ps and vor are both present if either is specified."""
        var_names = [v[0] for v in self.variables]
        has_ps = 'ps' in var_names
        has_vor = 'vor' in var_names
        
        if has_ps or has_vor:
            if not (has_ps and has_vor):
                print("Warning: ps and vor will be merged into base_ps_vor.nc. "
                      "Consider including both variables.", file=sys.stderr)
    
    def find_nearest_level_index(self, pressure: float) -> int:
        """
        Find the model level index nearest to the requested pressure level.
        
        Parameters:
        -----------
        pressure : float
            Desired pressure level in hPa
            
        Returns:
        --------
        int : 1-indexed model level number (for CDO compatibility)
        """
        idx = np.argmin(np.abs(self.plevels - pressure))
        return idx + 1  # CDO uses 1-indexed levels
    
    def parse_level(self, level: Union[str, int, float]) -> Tuple[int, str]:
        """
        Parse level specification and return (level_index, level_suffix).
        
        Parameters:
        -----------
        level : str, int, or float
            Level specification
            
        Returns:
        --------
        Tuple[int, str]: (model_level_index, suffix_for_output_filename)
        """
        # Surface level
        if isinstance(level, str) and level.lower() in ['surface', 'surf']:
            # Assume surface is the last level
            return len(self.plevels), ''
        
        # Direct index specification
        if isinstance(level, int) and 1 <= level <= len(self.plevels):
            # Assume this is a direct index if it's small enough
            # Use pressure value for suffix
            pressure = self.plevels[level - 1]
            suffix = str(int(pressure)) if pressure == int(pressure) else str(pressure)
            return level, suffix
        
        # Pressure level specification
        try:
            pressure = float(level)
            idx = self.find_nearest_level_index(pressure)
            actual_pressure = self.plevels[idx - 1]
            
            # Warn if the match isn't exact
            if abs(actual_pressure - pressure) > 10:  # More than 10 hPa difference
                print(f"Warning: Requested {pressure} hPa, using nearest level "
                      f"at {actual_pressure} hPa (index {idx})", file=sys.stderr)
            
            suffix = str(int(pressure)) if pressure == int(pressure) else str(pressure)
            return idx, suffix
            
        except (ValueError, TypeError):
            raise ValueError(f"Cannot parse level specification: {level}")
    
    def get_output_suffix(self, var_name: str, level: Union[str, int, float]) -> str:
        """Generate output filename suffix for a variable."""
        _, level_suffix = self.parse_level(level)
        
        if level_suffix:
            return f"{var_name}{level_suffix}"
        else:
            return var_name
    
    def generate_bash_script(self) -> str:
        """Generate bash script for CDO operations."""
        
        script_lines = [
            "#!/bin/bash",
            "#SBATCH --mem=64G",
            "#SBATCH -p mit_normal",
            "#SBATCH -o merge_files.out",
            "#SBATCH -e merge_files.err",
            "",
            "module load miniforge",
            "source activate isca_env",
            "",
            f"EXPROOT={self.exproot}",
            f"CH_I={self.chunk_start}",
            f"CH_F={self.chunk_end}",
            "",
            "cd $EXPROOT",
            "mkdir -p postprocessed",
            "",
            "# Extract variables from each chunk",
            "for CHNO in $(seq -f \"%04g\" $CH_I $CH_F); do",
        ]
        
        # Generate extraction commands for each variable
        extracted_vars = []
        for var_name, level in self.variables:
            level_idx, _ = self.parse_level(level)
            output_suffix = self.get_output_suffix(var_name, level)
            extracted_vars.append(output_suffix)
            
            cmd = (f"    cdo -sellevidx,{level_idx} -selname,{var_name} "
                  f"run${{CHNO}}/{self.input_filename} "
                  f"postprocessed/{output_suffix}_${{CHNO}}.nc")
            
            script_lines.append(cmd)
        
        script_lines.extend([
            "done",
            "",
            "cd postprocessed",
            "",
            "# Merge timeseries for each variable",
        ])
        
        # Generate merge commands and track which files to create
        merged_files = []
        for var_name, level in self.variables:
            output_suffix = self.get_output_suffix(var_name, level)
            merged_name = f"merged_{output_suffix}.nc"
            merged_files.append((output_suffix, merged_name))
            
            script_lines.append(
                f'cdo -mergetime $( ls {output_suffix}_*.nc ) {merged_name}'
            )
        
        # Clean up individual chunk files
        script_lines.append("")
        script_lines.append("# Clean up individual chunk files")
        for output_suffix, _ in merged_files:
            script_lines.append(f"rm {output_suffix}_*.nc")
        
        # Apply specific merging logic: ps + vor850 -> base_ps_vor.nc
        script_lines.append("")
        script_lines.append("# Create merged datasets following original logic")
        
        var_dict = {self.get_output_suffix(v[0], v[1]): v[0] 
                    for v in self.variables}
        
        # Find ps and vor files
        ps_file = None
        vor_file = None
        for suffix, varname in var_dict.items():
            if varname == 'ps':
                ps_file = f"merged_{suffix}.nc"
            elif varname == 'vor':
                vor_file = f"merged_{suffix}.nc"
        
        if ps_file and vor_file:
            script_lines.append(f"cdo -merge {ps_file} {vor_file} base_ps_vor.nc")
            script_lines.append(f"rm {ps_file} {vor_file}")
            
            # Compute anomalies if requested
            if self.compute_anomalies:
                script_lines.extend([
                    "",
                    "# Compute anomalies from time-zonal mean for base_ps_vor",
                    "cdo -griddes base_ps_vor.nc > output_grid.txt",
                    'cdo -enlarge,"output_grid.txt" -timmean base_ps_vor.nc base_t_mean.nc',
                    'cdo -sub base_ps_vor.nc base_t_mean.nc base_anoms.nc',
                    "# rm base_t_mean.nc  # Uncomment to clean up"
                ])
        
        if self.compute_anomalies:
            
            for var_name, level in self.variables:
                if (var_name != 'ps') and (var_name != 'vor'):
                    output_suffix = f"{var_name}{int(level)}" if str(level).lower() not in ['surf', 'surface'] else var_name
                    script_lines.extend([
                        "",
                        "# Compute anomalies from time-zonal mean",
                        f"cdo -griddes merged_{output_suffix}.nc > output_grid.txt",
                        f'cdo -enlarge,"output_grid.txt" -timmean merged_{output_suffix}.nc tmean_{output_suffix}.nc',
                        f'cdo -sub merged_{output_suffix}.nc tmean_{output_suffix}.nc anomaly_{output_suffix}.nc',
                        f'# rm tmean_{output_suffix}.nc     # Uncomment to clean up'
                    ])

        # Rename other merged files to match original naming convention
        # Keep them as separate timeseries
        for output_suffix, merged_name in merged_files:
            varname = var_dict[output_suffix]
            if varname not in ['ps', 'vor']:
                final_name = f"base_{output_suffix}.nc"
                script_lines.append(f"mv {merged_name} {final_name}")
        
        return "\n".join(script_lines)
    
    def run(self, submit_slurm: bool = True, dry_run: bool = False):
        """
        Execute the postprocessing.
        
        Parameters:
        -----------
        submit_slurm : bool
            If True, submit as SLURM job. If False, run directly.
        dry_run : bool
            If True, only print the script without executing.
        """
        bash_script = self.generate_bash_script()
        
        if dry_run:
            print("=" * 80)
            print("Generated bash script:")
            print("=" * 80)
            print(bash_script)
            print("=" * 80)
            return
        
        # Write script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(bash_script)
            script_path = f.name
        
        try:
            # Make executable
            subprocess.run(['chmod', '+x', script_path], check=True)
            
            # Submit or run
            if submit_slurm:
                result = subprocess.run(['sbatch', script_path], 
                                      capture_output=True, text=True)
                print(f"SLURM job submitted: {result.stdout}")
                if result.returncode != 0:
                    print(f"Error: {result.stderr}", file=sys.stderr)
            else:
                print("Running script directly...")
                result = subprocess.run(['bash', script_path])
                
        finally:
            # Clean up temp file
            Path(script_path).unlink()


def parse_variable_spec(var_spec: str) -> Tuple[str, Union[str, float]]:
    """
    Parse variable specification like 'vor:850' into ('vor', 850.0).
    
    Returns variable name and level (as string 'surface' or float pressure).
    """
    parts = var_spec.split(':')
    if len(parts) != 2:
        raise ValueError(f"Variable spec must be in format 'name:level', got: {var_spec}")
    
    var_name, level_str = parts
    
    # Check if it's surface
    if level_str.lower() in ['surface', 'surf']:
        return var_name, 'surface'
    
    # Try to parse as number
    try:
        level = float(level_str)
        return var_name, level
    except ValueError:
        raise ValueError(f"Level must be 'surface' or a number, got: {level_str}")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Postprocess GCM output using CDO',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using command line arguments
  python %(prog)s --exproot /path/to/exp --chunks 1 10 \\
      --vars ps:surface vor:850 temp:850 ucomp:850 vcomp:850

  # Using config file
  python %(prog)s --config config.yaml
  
  # Dry run to see generated script
  python %(prog)s --config config.yaml --dry-run
  
  # Compute anomalies for base_ps_vor file
  python %(prog)s --config config.yaml --anomalies
        """
    )
    
    parser.add_argument('--config', type=str, 
                       help='Path to YAML configuration file')
    parser.add_argument('--exproot', type=str,
                       help='Root directory of experiment')
    parser.add_argument('--chunks', type=int, nargs=2, metavar=('START', 'END'),
                       help='Start and end chunk numbers')
    parser.add_argument('--vars', type=str, nargs='+',
                       help='Variables to extract in format "name:level" '
                            '(e.g., vor:850 ps:surface temp:700)')
    parser.add_argument('--plevels-file', type=str,
                       help='Path to numpy file containing pressure levels array')
    parser.add_argument('--no-slurm', action='store_true',
                       help='Run directly instead of submitting to SLURM')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print generated script without executing')
    parser.add_argument('--anomalies', action='store_true',
                       help='Compute anomalies from time-mean for base_ps_vor')
    parser.add_argument('--input-file', type=str, default='atmos_6_hourly.nc',
                       help='Name of input NetCDF files (default: atmos_6_hourly.nc)')
    
    args = parser.parse_args()
    
    # Load pressure levels if specified
    plevels = None
    if args.plevels_file:
        plevels = np.load(args.plevels_file)
        print(f"Loaded {len(plevels)} pressure levels from {args.plevels_file}")
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        exproot = config['exproot']
        chunk_start, chunk_end = config['chunks']
        variables = [parse_variable_spec(v) for v in config['variables']]
        compute_anomalies = config.get('compute_anomalies', False)
        input_filename = config.get('input_filename', 'atmos_6_hourly.nc')
        
        # Override with plevels from config if not specified on command line
        if plevels is None and 'plevels' in config:
            plevels = np.array(config['plevels'])
            
    else:
        if not all([args.exproot, args.chunks, args.vars]):
            parser.error("Must provide either --config or (--exproot, --chunks, --vars)")
        
        exproot = args.exproot
        chunk_start, chunk_end = args.chunks
        variables = [parse_variable_spec(v) for v in args.vars]
        compute_anomalies = args.anomalies
        input_filename = args.input_file
    
    # Create processor and run
    processor = CDOPostProcessor(
        exproot=exproot,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        variables=variables,
        plevels=plevels,
        compute_anomalies=compute_anomalies,
        input_filename=input_filename
    )
    
    processor.run(submit_slurm=not args.no_slurm, dry_run=args.dry_run)


if __name__ == '__main__':
    main()