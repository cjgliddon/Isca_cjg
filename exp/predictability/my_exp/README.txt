This folder contains sample scripts for running a general ensemble predictability experiment in ISCA.

A description of the components general workflow:
- shared_vars.py :          This Python file defines some key shared variables -- the experiment name, diagnostics table, and
                            run namelist -- for the control and perturbed experiments.
- control.py :              This script allows you to specify the name of the experiment and the basic model parameters which
                            affect the GCM's "climate" (atmospheric optical depth, Clausius-Clapeyron coefficient, insolation,
                            etc.). It then produces a long control run of the model in 30-day "chunks."
- control-tracking.sh :     This script does two things: (1) produces "merged" files containing the surface pressure and
                            vorticity fields and their anomalies from the time mean over the entire model run, and (2) performs
                            Hodges' tracking algorithm on the anomaly fields. It does this by calling two scripts in the 
                            Isca/postprocessing directory as subprocesses.
- specpert.py :             This script generates the ensemble runs. 