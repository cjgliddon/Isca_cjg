# cdo_utils.py
"""
Shared utilities for CDO-based GCM postprocessing scripts.

Provides:
    - Default model pressure level array (PLEVELS)
    - Level parsing and nearest-level lookup
    - Variable specification parsing
    - YAML config loading
    - SLURM header generation
    - Bash script execution / SLURM submission
"""

import subprocess
import sys
import tempfile
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Default model pressure levels
# ---------------------------------------------------------------------------

# Pressure levels in hPa, ordered from model top to bottom.
# Edit to match your model's vertical coordinate.
PLEVELS = np.array([  8.47351309,  25.82931981,  32.07866536,  39.52406696,
        48.31838009,  58.6182856 ,  70.58131668,  84.36252112,
       100.11086254, 117.96549002, 138.05202803, 160.47905414,
       185.33494127, 212.68524175, 242.57078409, 275.00663799,
       309.98208375, 347.46169706, 387.38763601, 429.68319536,
       474.25768074, 521.01265952, 569.84967736, 620.6796093 ,
       673.43397873, 728.07890684, 784.63304473, 843.19246746,
       903.97029329, 967.38247962])


# ---------------------------------------------------------------------------
# Level utilities
# ---------------------------------------------------------------------------

def find_nearest_level_index(pressure: float, plevels: np.ndarray) -> int:
    """
    Return the 1-indexed model level whose pressure is nearest to `pressure`.

    Parameters
    ----------
    pressure : float
        Target pressure in hPa.
    plevels : np.ndarray
        Ordered array of model pressure levels in hPa.

    Returns
    -------
    int
        1-indexed level number, compatible with CDO's sellevidx operator.
    """
    idx = int(np.argmin(np.abs(plevels - pressure)))
    return idx + 1  # CDO uses 1-indexed levels


def parse_level(
    level: Union[str, int, float],
    plevels: np.ndarray,
) -> Tuple[int, str]:
    """
    Parse a level specification into a (CDO level index, filename suffix) pair.

    Accepted formats
    ----------------
    'surface' / 'surf'
        Maps to the last model level; suffix is an empty string.
    int in [1, nlev]
        Treated as a direct 1-indexed CDO level index; suffix is derived from
        the corresponding pressure value in `plevels`.
    numeric value (int or float outside [1, nlev], or any float)
        Treated as a pressure in hPa; the nearest level in `plevels` is used.

    Parameters
    ----------
    level : str | int | float
        Level specification supplied by the user.
    plevels : np.ndarray
        Ordered array of model pressure levels in hPa.

    Returns
    -------
    Tuple[int, str]
        (1-indexed CDO level index, suffix string for output filenames)

    Raises
    ------
    ValueError
        If `level` cannot be interpreted under any of the above rules.
    """
    # Surface shorthand
    if isinstance(level, str) and level.lower() in ('surface', 'surf'):
        return len(plevels), ''

    # Direct level-index path: small positive integer within the valid range
    if isinstance(level, int) and 1 <= level <= len(plevels):
        pressure = plevels[level - 1]
        suffix = str(int(pressure)) if pressure == int(pressure) else str(pressure)
        return level, suffix

    # Pressure-value path (covers floats and ints outside the index range)
    try:
        pressure = float(level)
        idx = find_nearest_level_index(pressure, plevels)
        actual = plevels[idx - 1]
        if abs(actual - pressure) > 10:
            print(
                f"Warning: requested {pressure} hPa; "
                f"using nearest model level at {actual} hPa (index {idx}).",
                file=sys.stderr,
            )
        suffix = str(int(pressure)) if pressure == int(pressure) else str(pressure)
        return idx, suffix

    except (TypeError, ValueError):
        raise ValueError(f"Cannot interpret level specification: {level!r}")


def get_output_suffix(
    var_name: str,
    level: Union[str, int, float],
    plevels: np.ndarray,
) -> str:
    """
    Build the output filename suffix for a (variable, level) pair.

    For example: ('vor', 850) → 'vor850', ('ps', 'surface') → 'ps'.

    Parameters
    ----------
    var_name : str
        NetCDF variable name.
    level : str | int | float
        Level specification (see parse_level).
    plevels : np.ndarray
        Ordered array of model pressure levels in hPa.

    Returns
    -------
    str
        Suffix string, e.g. 'vor850' or 'ps'.
    """
    _, level_suffix = parse_level(level, plevels)
    return f"{var_name}{level_suffix}"


# ---------------------------------------------------------------------------
# Variable specification parsing
# ---------------------------------------------------------------------------

def parse_variable_spec(spec: str) -> Tuple[str, Union[str, float]]:
    """
    Parse a 'varname:level' string into a (name, level) tuple.

    The level component is returned either as the string 'surface' or as a
    float (pressure in hPa).

    Parameters
    ----------
    spec : str
        Specification such as 'vor:850', 'ps:surface', or 'temp:700'.

    Returns
    -------
    Tuple[str, str | float]
        (variable name, level)

    Raises
    ------
    ValueError
        If `spec` is not in the expected 'name:level' format.
    """
    parts = spec.split(':')
    if len(parts) != 2:
        raise ValueError(
            f"Variable spec must be 'name:level' (e.g. vor:850), got: {spec!r}"
        )
    var_name, level_str = parts
    if level_str.lower() in ('surface', 'surf'):
        return var_name, 'surface'
    try:
        return var_name, float(level_str)
    except ValueError:
        raise ValueError(
            f"Level must be 'surface' or a number (hPa), got: {level_str!r}"
        )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict:
    """
    Load a YAML configuration file and return its contents as a dict.

    Parameters
    ----------
    path : str
        Path to the YAML file.

    Returns
    -------
    Dict
        Parsed configuration dictionary.
    """
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Bash script helpers
# ---------------------------------------------------------------------------

def make_slurm_header(
    mem: str = '16G',
    partition: str = 'mit_normal',
    stdout: str = 'cdo_job.out',
    stderr: str = 'cdo_job.err',
    extra_directives: Optional[List[str]] = None,
) -> List[str]:
    """
    Return a list of lines forming a standard SLURM header block.

    Parameters
    ----------
    mem : str
        Memory request (e.g. '16G', '64G').
    partition : str
        SLURM partition name.
    stdout : str
        Path for standard output log.
    stderr : str
        Path for standard error log.
    extra_directives : list of str, optional
        Additional #SBATCH lines (without the leading '#SBATCH ').

    Returns
    -------
    List[str]
        Lines to prepend to a bash script, including the shebang.
    """
    lines = [
        "#!/bin/bash",
        f"#SBATCH --mem={mem}",
        f"#SBATCH -p {partition}",
        f"#SBATCH -o {stdout}",
        f"#SBATCH -e {stderr}",
    ]
    for directive in (extra_directives or []):
        lines.append(f"#SBATCH {directive}")
    lines += [
        "",
        "module load miniforge",
        "source activate isca_env",
        "",
    ]
    return lines


def run_script(
    bash_script: str,
    submit_slurm: bool = True,
    dry_run: bool = False,
) -> None:
    """
    Write `bash_script` to a temporary file and execute or submit it.

    Parameters
    ----------
    bash_script : str
        Complete bash script as a single string.
    submit_slurm : bool
        If True, submit with sbatch; if False, run directly with bash.
    dry_run : bool
        If True, print the script and return without executing anything.
    """
    if dry_run:
        print("=" * 72)
        print("Generated bash script (dry run):")
        print("=" * 72)
        print(bash_script)
        print("=" * 72)
        return

    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(bash_script)
        script_path = f.name

    try:
        subprocess.run(['chmod', '+x', script_path], check=True)

        if submit_slurm:
            result = subprocess.run(
                ['sbatch', script_path],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(f"SLURM job submitted: {result.stdout.strip()}")
            else:
                print(f"sbatch error:\n{result.stderr}", file=sys.stderr)
                sys.exit(result.returncode)
        else:
            print("Running script directly ...")
            subprocess.run(['bash', script_path], check=True)

    finally:
        Path(script_path).unlink()