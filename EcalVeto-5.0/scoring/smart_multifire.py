#!/usr/bin/env python3
import argparse
import logging
import os
import subprocess
import time
from datetime import datetime


# THINGS TO CHANGE

# 0) BDT - maybe
# 1) DIRECTORIES: INDIR, OUTDIR
# 2) GROUP LABEL
# 3) PASS NAME - Only change once the config has been done running or else you will ruin the whole batch
# 4) Recoil From Tracking

# ============================================================
# Variables
# ============================================================

##### --- Directories --- #####

BACKGROUND = "/path/to/base/for/bkg/files"
SIGNAL = "/path/to/base/for/signal/files"

INDIR  = BACKGROUND
OUTDIR = SIGNAL


# --- File selection ---
RUN_ALL  = True   # True = process every .root file in INDIR
N_FILES  = 100     # Number of files to process when RUN_ALL is False

# --- Run numbering ---
START_RUN = 1      # First run number assigned to the first batch

# --- Batching ---
FILES_PER_TASK = 1   # signal skim: keep per-mass files separate (NO merge)
TASKS_PER_JOB  = 8   # skim is light (no tracking)

# --- Config script arguments ---
MAX_EVENTS  = -1     # -1 = process all events in input files
GROUP_LABEL = "Default_BDT"   # match ACTIVE pass (Process() in ecalveto_config.py)

# --- Partition ---
# 1 = batch (default, up to 58 days)
# 2 = short (2-hour limit)
# 3 = largemem
# 4 = aws (2-hour limit)
PARTITION_CHOICE = 2   # short (2h) - skim is fast, no tracking
PARTITION_MAP = {
    1: ("batch",   "1-00:00:00"),
    2: ("short",   "2:00:00"),
    3: ("largemem","12:00:00"),
    4: ("aws",     "2:00:00"),
}
PARTITION, TIME_LIMIT = PARTITION_MAP[PARTITION_CHOICE]

# --- Throttling ---
QUEUE_LIMIT     = 300   # Max queued jobs before pausing submissions
PAUSE_TIME      = 600  # Seconds to wait when queue is full 
SUBMIT_INTERVAL = 0   # Seconds between successive job submissions 

print("=== Smart Multifire Configuration ===")
print("INPUT DIR:", INDIR)
print("OUTPUT DIR:", OUTDIR)
print("===================================")
# ============================================================

# Default argument is set to ecalveto_config.py
# Default command to run: python3 smart_multifire.py (do not use denv python3)
# To run other scripts (i.e. digi_tracking_ecalveto_config.py)
# python3 smart_multifire.py -py digi_tracking_ecalveto_config.py

# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description='Smart Multifire Script for LDMX')
    parser.add_argument('-py', '--script', action='store', dest='script',
                        default='/home/jihoonyoo/LDMX/v4.5.11/ldmx-sw/Hanna_BDT_Scripts/ecalveto_config.py',
                        help='Configuration script to run')
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(format='[ smart_multifire ][ %(levelname)s ]: %(message)s', level=logging.INFO)


def check_queue(limit):
    """Checks if the user's SLURM queue size exceeds the limit."""
    try:
        cmd = "squeue -u $USER -h | wc -l"
        result = subprocess.check_output(cmd, shell=True).strip()
        count = int(result)
        return count >= limit, count
    except Exception as e:
        logging.warning(f"Failed to check queue size: {e}. Assuming queue is not full.")
        return False, 0


def get_files(indir, run_all, n_files):
    """Retrieves list of .root files from indir."""
    if not os.path.exists(indir):
        logging.error(f"Input directory not found: {indir}")
        return []

    files = sorted(os.path.join(indir, f) for f in os.listdir(indir) if f.endswith('.root'))

    if not run_all:
        logging.info(f"Limiting to first {n_files} files (RUN_ALL=False).")
        files = files[:n_files]

    return files


def main():
    args = parse_arguments()
    setup_logging()

    script = args.script
    if not os.path.exists(script):
        logging.error(f"Config script not found: {script}")
        return

    # Create output directory
    if not os.path.exists(OUTDIR):
        logging.info(f"Creating output directory: {OUTDIR}")
        os.makedirs(OUTDIR)

    # Job and Log directories inside ~/LDMX as requested
    home_dir = os.path.expanduser("~")
    jobdir = os.path.join(home_dir, 'LDMX/slurm/jobs/')
    logdir = os.path.join(home_dir, 'LDMX/slurm/logs/')

    os.makedirs(jobdir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    logging.info(f"Jobs dir: {jobdir}")
    logging.info(f"Logs dir: {logdir}")

    # Gather input files
    all_files = get_files(INDIR, RUN_ALL, N_FILES)
    total_files = len(all_files)
    logging.info(f"Found {total_files} input files to process.")

    if total_files == 0:
        logging.warning("No files to process. Exiting.")
        return

    # ------------------------------------------------------------------
    # Batching:
    #   Task = 1 fire process processing FILES_PER_TASK input files
    #   Job  = 1 SLURM submission running TASKS_PER_JOB tasks in parallel
    # ------------------------------------------------------------------
    tasks = []
    current_run = START_RUN

    for i in range(0, total_files, FILES_PER_TASK):
        batch = all_files[i : i + FILES_PER_TASK]
        file_string = ",".join(batch)  # comma-separated for the config script
        tasks.append((file_string, current_run))
        current_run += 1

    total_tasks = len(tasks)
    logging.info(f"Created {total_tasks} tasks (batches of up to {FILES_PER_TASK} files).")

    jobs = [tasks[i : i + TASKS_PER_JOB] for i in range(0, total_tasks, TASKS_PER_JOB)]
    total_jobs = len(jobs)
    logging.info(f"Grouped tasks into {total_jobs} SLURM jobs (up to {TASKS_PER_JOB} tasks per job).")

    script_path = os.path.abspath(script)

    # ------------------------------------------------------------------
    # Submission loop
    # ------------------------------------------------------------------
    for job_idx, job_tasks in enumerate(jobs):

        # --- Throttling ---
        while True:
            is_full, count = check_queue(QUEUE_LIMIT)
            if is_full:
                logging.info(f"Queue full ({count} >= {QUEUE_LIMIT}). Pausing for {PAUSE_TIME}s ...")
                time.sleep(PAUSE_TIME)
            else:
                logging.info(f"Queue ok ({count} < {QUEUE_LIMIT}). Proceeding.")
                break

        # Build per-task argument lists for GNU parallel --link
        file_args = [t[0] for t in job_tasks]
        run_args  = [str(t[1]) for t in job_tasks]

        files_str = " ".join(f"'{f}'" for f in file_args)
        runs_str  = " ".join(run_args)

        job_name = f"ecal_veto_{job_idx}"
        job_file = os.path.join(jobdir, f"{job_name}.job")

        with open(job_file, 'w') as f:
            f.write('#!/bin/bash\n\n')
            f.write(f'#SBATCH --job-name={job_name}\n')
            f.write('#SBATCH --nodes=1\n')
            f.write(f'#SBATCH --ntasks={TASKS_PER_JOB}\n')
            f.write('#SBATCH --cpus-per-task=1\n')
            f.write('#SBATCH --mem=16000M\n')
            f.write(f'#SBATCH --error={logdir}/slurm-%J.err\n')
            f.write(f'#SBATCH --output={logdir}/slurm-%J.out\n')
            f.write(f'#SBATCH --partition={PARTITION}\n')
            f.write(f'#SBATCH --time={TIME_LIMIT}\n')
            f.write('#SBATCH --mail-type=FAIL\n')
            f.write(f'#SBATCH --mail-user={os.environ.get("USER")}@ucsb.edu\n')
            f.write('\n')
            f.write('cd $SLURM_SUBMIT_DIR\n')
            f.write('echo "Starting job on $(hostname)"\n\n')

            cmd = (
                f"just fire-parallel {script_path} "
                f"-i {{1}} -r {{2}} -o {OUTDIR} -m {MAX_EVENTS} -g {GROUP_LABEL} "
                "-- --link "
                f"::: {files_str} "
                f"::: {runs_str}"
            )
            f.write(f"{cmd}\n")

        # Submit
        submit_cmd = f"sbatch {job_file}"
        logging.info(f"Submitting job {job_idx+1}/{total_jobs}: {submit_cmd}")
        subprocess.Popen(submit_cmd, shell=True).wait()

        # Inter-submission delay
        time.sleep(SUBMIT_INTERVAL)


if __name__ == "__main__":
    main()
