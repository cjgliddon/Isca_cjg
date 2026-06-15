#!/usr/bin/env python3
"""
exp_workflow.py

A Python wrapper for submitting and managing SLURM jobs for the ensemble 
predictability experiment workflow.

Usage:
    python exp_workflow.py --do-control --do-ensemble [--dry-run] [--preview-dir DIR]
    python exp_workflow.py --help
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

from shared_vars import expname

# -----------------------------------------------------------------------------
# basic config details:
# how long of a control simulation do we want?
# which ensemble members do we want to run? 

n_months_control = 120
ensemble_t0, ensemble_tf, dt = 360.0, 720.0, 10.0
mems = range(15, 31)

# -----------------------------------------------------------------------------

class SLURMJobManager:
    """Manages SLURM job submission with dependency tracking."""
    
    def __init__(
        self,
        script_dir: Optional[str] = None,
        dry_run: bool = False,
        preview_dir: Optional[str] = None,
    ):
        """
        Initialize the SLURM job manager.
        
        Parameters
        ----------
        script_dir : str, optional
            Directory containing the experiment scripts. If None, uses current directory.
        dry_run : bool, optional
            If True, print scripts without submitting (default: False)
        preview_dir : str, optional
            Directory to save preview scripts. If None, scripts printed to stdout only.
        """
        self.script_dir = Path(script_dir) if script_dir else Path.cwd()
        self.dry_run = dry_run
        self.preview_dir = Path(preview_dir) if preview_dir else None
        self.job_ids: Dict[str, int] = {}
        self.script_counter = 0  # For numbering preview files
        
        # Create preview directory if needed
        if self.preview_dir and not self.preview_dir.exists():
            self.preview_dir.mkdir(parents=True, exist_ok=True)
            if dry_run:
                print(f"✓ Created preview directory: {self.preview_dir}\n")
        
    def submit_job(
        self,
        script_name: str,
        script_args: List[str],
        job_name: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
        partition: str = "mit_normal",
        time_limit: str = "12:00:00",
        n_tasks: int = 1,
        verbose: bool = True,
    ) -> int:
        """
        Submit a SLURM job using sbatch.
        
        Parameters
        ----------
        script_name : str
            Name of the script to submit (e.g., 'control.py')
        script_args : list of str
            Arguments to pass to the script
        job_name : str, optional
            Name for the job (for tracking)
        depends_on : list of str, optional
            List of job names to depend on (e.g., ['control_run_1'])
        partition : str
            SLURM partition to submit to
        time_limit : str
            Time limit for the job (HH:MM:SS format)
        n_tasks : int
            Number of tasks to request
        verbose : bool
            Print submission details
            
        Returns
        -------
        int
            SLURM job ID (or mock ID in dry-run mode)
        """
        script_path = self.script_dir / script_name
        
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        
        # Determine how to execute the script
        if script_name.endswith('.py'):
            cmd = ['python', str(script_path)] + script_args
        else:
            cmd = ['bash', str(script_path)] + script_args
        
        # Build sbatch command
        sbatch_cmd = [
            'sbatch',
            '--parsable',  # Output only job ID
            f'--job-name={job_name or script_name}',
            f'--partition={partition}',
            f'--time={time_limit}',
            f'--ntasks={n_tasks}',
        ]
        
        # Add dependencies if specified
        if depends_on:
            dep_job_ids = [str(self.job_ids[dep]) for dep in depends_on]
            dep_type = 'afterok'  # Default: only run if previous jobs succeeded
            sbatch_cmd.append(f'--dependency={dep_type}:{":".join(dep_job_ids)}')
        
        # Create wrapper script
        wrapper_content = f"""#!/bin/bash
#SBATCH --job-name={job_name or script_name}
#SBATCH --partition={partition}
#SBATCH --time={time_limit}
#SBATCH --ntasks={n_tasks}"""
        
        if depends_on:
            dep_job_ids = [str(self.job_ids[dep]) for dep in depends_on]
            dep_str = ":".join(dep_job_ids)
            wrapper_content += f"\n#SBATCH --dependency=afterok:{dep_str}"
        
        wrapper_content += f"\n\n{' '.join(cmd)}\n"
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Job: {job_name or script_name}")
            print(f"{'='*70}")
            print(wrapper_content)
            if depends_on:
                print(f"Dependencies: {depends_on}")
            print(f"{'='*70}")
        
        # Dry-run mode: save or print scripts without submitting
        if self.dry_run:
            if self.preview_dir:
                self.script_counter += 1
                preview_file = self.preview_dir / f"{self.script_counter:02d}_{job_name or script_name}.sh"
                with open(preview_file, 'w') as f:
                    f.write(wrapper_content)
                print(f"[DRY-RUN] Script saved to: {preview_file}\n")
            
            # Assign mock job ID for dependency tracking
            mock_job_id = 100000 + self.script_counter
            if job_name:
                self.job_ids[job_name] = mock_job_id
            return mock_job_id
        
        # Normal mode: submit via sbatch
        result = subprocess.run(
            ['sbatch', '--parsable'] + sbatch_cmd[2:],
            input=wrapper_content,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"sbatch submission failed: {result.stderr}")
        
        job_id = int(result.stdout.strip())
        if job_name:
            self.job_ids[job_name] = job_id
        
        if verbose:
            print(f"✓ Job submitted with ID: {job_id}\n")
        
        return job_id


class ExperimentWorkflow:
    """Manages the complete ensemble predictability experiment workflow."""
    
    def __init__(
        self,
        expname: str = expname,
        script_dir: Optional[str] = None,
        do_control: bool = False,
        do_ensemble: bool = False,
        dry_run: bool = False,
        preview_dir: Optional[str] = None,
    ):
        """
        Initialize the experiment workflow.
        
        Parameters
        ----------
        expname : str
            Name of the experiment (must match shared_vars.py)
        script_dir : str, optional
            Directory containing experiment scripts
        do_control : bool
            Whether to run control experiment
        do_ensemble : bool
            Whether to run ensemble perturbations
        dry_run : bool, optional
            If True, generate scripts without submitting (default: False)
        preview_dir : str, optional
            Directory to save preview scripts (only used if dry_run=True)
        """
        self.expname = expname
        self.script_dir = Path(script_dir) if script_dir else Path.cwd()
        self.do_control = do_control
        self.do_ensemble = do_ensemble
        self.dry_run = dry_run
        self.preview_dir = preview_dir
        self.job_manager = SLURMJobManager(
            self.script_dir,
            dry_run=self.dry_run,
            preview_dir=self.preview_dir,
        )
        
    def run_control(self, n_months: int = n_months_control, chunk_len: int = 24, 
                    move_to_storage: bool = True, verbose: bool = True) -> List[str]:
        """
        Submit control run jobs.
        
        Parameters
        ----------
        n_months : int, optional
            Length of control run in months.
        chunk_len : int, optional
            Length of each control "chunk" in months. Default: 24
        move_to_storage : bool
            Whether to move output to storage after run
        verbose : bool
            Print submission details
            
        Returns
        -------
        list of str
            Names of submitted control jobs for dependency tracking
        """
        
        if verbose:
            print(f"\n{'#'*70}")
            print("# CONTROL RUN SUBMISSION")
            print(f"{'#'*70}")
        
        control_jobs = []
        prev_job = None
        
        n_chunks = int(n_months_control/chunk_len)
        for i in range(n_chunks):
            job_name = f"control_chunk_{i}"
            start_day = i*chunk_len + 1
            end_day = min((i+1)*chunk_len, n_months_control)
            script_args = [str(start_day), str(end_day), str(int(move_to_storage))]
            
            depends_on = [prev_job] if prev_job else None
            
            self.job_manager.submit_job(
                'control.py',
                script_args,
                job_name=job_name,
                depends_on=depends_on,
                n_tasks=16,
                verbose=verbose,
            )
            
            control_jobs.append(job_name)
            prev_job = job_name
        
        return control_jobs
    
    def run_ensemble(
        self,
        control_jobs: List[str],
        day_start: float = ensemble_t0,
        day_end: float = ensemble_tf,
        day_step: float = dt,
        members: range = mems,
        verbose: bool = True,
    ) -> tuple:
        """
        Submit ensemble perturbation jobs.
        
        Parameters
        ----------
        control_job : str
            Name of control job to depend on
        day_start : float
            Starting day for perturbations
        day_end : float
            Ending day for perturbations
        day_step : float
            Step size in days between perturbations
        members : range
            Range of ensemble members per perturbation time
        verbose : bool
            Print submission details
            
        Returns
        -------
        tuple
            (list of perturbation times, list of all perturbation jobs)
        """
        if verbose:
            print(f"\n{'#'*70}")
            print("# ENSEMBLE PERTURBATION SUBMISSION")
            print(f"{'#'*70}")
        
        perturbation_times = list(range(int(day_start), int(day_end) + 1, int(day_step)))
        all_pert_jobs = []
        
        for t0 in perturbation_times:
            mem0 = members[0]
            memf = members[-1]
            job_name = f"specpert_t0_{t0}_mem_{mem0}-{memf}"
            script_args = [str(t0), str(mem0), str(memf)]
            
            self.job_manager.submit_job(
                'specpert.py',
                script_args,
                job_name=job_name,
                depends_on=control_jobs,
                n_tasks=16,
                verbose=verbose,
            )
            
            all_pert_jobs.append(job_name)
            
            if verbose:
                print(f"✓ Submitted ensemble members {mem0}-{memf} for t0={t0}")
        
        return perturbation_times, all_pert_jobs
    
    def run_ensemble_patch(
        self,
        pert_jobs: List[str],
        day_start: float = ensemble_t0,
        day_end: float = ensemble_tf,
        day_step: int = dt,
        members: range = mems,
        verbose: bool = True,
    ) -> str:
        """
        Submit ensemble patching job to consolidate all ensemble output.
        
        Parameters
        ----------
        pert_jobs : list of str
            Names of all perturbation jobs to depend on
        day_start : float
            Starting day (must match ensemble run config)
        day_end : float
            Ending day (must match ensemble run config)
        day_step : int
            Step size in days (must match ensemble run config)
        verbose : bool
            Print submission details
            
        Returns
        -------
        str
            Name of patching job
        """
        if verbose:
            print(f"\n{'#'*70}")
            print("# ENSEMBLE PATCHING SUBMISSION")
            print(f"{'#'*70}")
        
        job_name = "specpert_patch"
        script_args = [str(day_start), str(day_end), str(day_step), 
                       str(members[0]), str(members[-1])] 
        
        self.job_manager.submit_job(
            'specpert_patch.py',
            script_args,
            job_name=job_name,
            depends_on=pert_jobs,
            n_tasks=16,
            verbose=verbose,
        )
        
        return job_name
        
    def run(self):
        """Execute the complete workflow based on configuration."""
        mode_str = "DRY-RUN" if self.dry_run else "NORMAL"
        
        print(f"\n{'='*70}")
        print(f"ISCA Ensemble Predictability Experiment Workflow")
        print(f"Mode: {mode_str}")
        print(f"Experiment: {self.expname}")
        print(f"Control: {self.do_control}, Ensemble: {self.do_ensemble}")
        print(f"Script directory: {self.script_dir}")
        if self.preview_dir:
            print(f"Preview directory: {self.preview_dir}")
        print(f"{'='*70}\n")
        
        control_jobs = []
        patch_job = None
        
        if self.do_control:
            # Run control simulation
            control_jobs = self.run_control()
        
        if self.do_ensemble:
            if control_jobs is None:
                raise ValueError("Cannot run ensemble without control jobs. "
                               "Either run control first or ensure it completed.")
            # Run ensemble perturbations
            pert_times, pert_jobs = self.run_ensemble(control_jobs)
            
            # Run ensemble patching
            patch_job = self.run_ensemble_patch(
                pert_jobs,
                day_start=360,
                day_end=1440,
                day_step=10,
            )
        
        print(f"\n{'='*70}")
        if self.dry_run:
            print("✓ Dry-run complete! Preview scripts generated.")
            if self.preview_dir:
                print(f"Scripts saved to: {self.preview_dir}")
                print(f"Total scripts: {self.job_manager.script_counter}")
        else:
            print("✓ All jobs submitted successfully!")
            print(f"Job IDs: {self.job_manager.job_ids}")
        print(f"{'='*70}\n")


def main():
    """Parse command-line arguments and run workflow."""
    parser = argparse.ArgumentParser(
        description="Python wrapper for ISCA ensemble predictability experiment workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run control and ensemble
  python exp_workflow.py frierson_my_experiment --do-control --do-ensemble
  
  # Preview scripts without submitting
  python exp_workflow.py frierson_my_experiment --do-control --do-ensemble --dry-run
  
  # Preview scripts and save to directory
  python exp_workflow.py frierson_my_experiment --do-control --do-ensemble \\
    --dry-run --preview-dir ./scripts_preview
  
  # Run only control
  python exp_workflow.py frierson_my_experiment --do-control
  
  # Run only ensemble (assumes control already exists)
  python exp_workflow.py frierson_my_experiment --do-ensemble
        """,
    )
    
    parser.add_argument(
        '--do-control',
        action='store_true',
        help='Run control experiment',
    )
    
    parser.add_argument(
        '--do-ensemble',
        action='store_true',
        help='Run ensemble perturbations',
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Generate preview scripts without submitting to SLURM',
    )
    
    parser.add_argument(
        '--preview-dir',
        default=None,
        help='Directory to save preview bash scripts (only used with --dry-run)',
    )
    
    parser.add_argument(
        '--script-dir',
        default=None,
        help='Directory containing experiment scripts (default: current directory)',
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print detailed submission information (default: True)',
    )
    
    args = parser.parse_args()
    
    if not (args.do_control or args.do_ensemble):
        parser.error("Must specify at least --do-control or --do-ensemble")
    
    if args.preview_dir and not args.dry_run:
        print("Warning: --preview-dir specified but --dry-run not enabled. "
              "Preview scripts will not be saved.")
    
    workflow = ExperimentWorkflow(
        expname=expname,
        script_dir=args.script_dir,
        do_control=args.do_control,
        do_ensemble=args.do_ensemble,
        dry_run=args.dry_run,
        preview_dir=args.preview_dir,
    )
    
    workflow.run()


if __name__ == '__main__':
    main()