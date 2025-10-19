# MLCompareRunnerConfig.py - Final Orchestrator

# ================================ FRAMEWORK IMPORTS ================================
from EventManager.Models.RunnerEvents import RunnerEvents
from EventManager.EventSubscriptionController import EventSubscriptionController
from ConfigValidator.Config.Models.RunTableModel import RunTableModel
from ConfigValidator.Config.Models.FactorModel import FactorModel
from ConfigValidator.Config.Models.RunnerContext import RunnerContext
from ConfigValidator.Config.Models.OperationType import OperationType
from ProgressManager.Output.OutputProcedure import OutputProcedure as output
# from Plugins.Profilers.Ps import Ps # We don't need Ps, we use PowerJoular

from typing import Dict, List, Any, Optional
from pathlib import Path
from os.path import dirname, realpath

# ================================ CUSTOM IMPORTS ================================
import numpy as np
import time
import subprocess
import shlex
import pandas as pd
import sys # For custom errors/exiting if needed
import os
# --- CONFIGURATION CONSTANTS (Defined outside class for easy access) ---
PROJECT_ROOT = Path("/home/rudexp/Downloads/greenLab_group33-main")
OUTPUT_DIR = PROJECT_ROOT / "experiment_results"
RUN_ORDER_FILE = PROJECT_ROOT / "run_order.csv" # Initial path (will be changed for GPU block)

# --- GLOBAL VARIABLES ---
RUN_SEQUENCE = []
os.makedirs(OUTPUT_DIR / "raw_powerjoular_data", exist_ok=True) # Ensure powerjoular dir exists

# --- HELPER FUNCTION ---
def load_run_sequence():
    """Loads the run sequence from the external CSV file."""
    try:
        df = pd.read_csv(RUN_ORDER_FILE)
        return df['ML_Library'].tolist()
    except FileNotFoundError:
        output.console_log(f"ERROR: Run order file not found at {RUN_ORDER_FILE}", is_error=True)
        # Exit is handled outside the class, but we use sys.exit here for safety in a real script.
        # However, for framework integration, we'll just return an empty list and let the framework fail gracefully.
        return []
    except KeyError:
        output.console_log("ERROR: Run order CSV must contain a column named 'ML_Library'", is_error=True)
        return []

# Load the sequence immediately when the script runs
RUN_SEQUENCE = load_run_sequence()

# ================================ FRAMEWORK CLASS ================================

class RunnerConfig:
    ROOT_DIR = PROJECT_ROOT # Set ROOT_DIR to the actual project root
    
    # ================================ USER SPECIFIC CONFIG ================================
    """The name of the experiment."""
    name:                    str               = "ML_Energy_Comparison_CPU_Only"

    """The path in which Experiment Runner will create a folder with the name `self.name`, in order to store the
    results from this experiment. (Path does not need to exist - it will be created if necessary.)
    Output path defaults to the config file's path, inside the folder 'experiments'"""
    results_output_path:     Path              = OUTPUT_DIR # Use our custom output directory

    """Experiment operation type. Unless you manually want to initiate each run, use `OperationType.AUTO`."""
    operation_type:          OperationType     = OperationType.AUTO

    """The time Experiment Runner will wait after a run completes.
    This can be essential to accommodate for cooldown periods on some systems."""
    time_between_runs_in_ms: int               = 30000 # 🚨 30 seconds (30,000 ms) as required

    # Dynamic configurations can be one-time satisfied here before the program takes the config as-is
    # e.g. Setting some variable based on some criteria
    def __init__(self):
        """Executes immediately after program start, on config load"""
        
        # 🚨 CRITICAL: HARD-CODED METRICS PATHS TO MATCH ML SCRIPTS' OUTPUT LOCATION
        self.metrics_log_paths = {
            'TF_CPU': PROJECT_ROOT / "src" / "libraries" / "tensorflow" / "metrics_tfNN_CPU.csv",
            'PT_CPU': PROJECT_ROOT / "src" / "libraries" / "Pytorch" / "metrics_ptNN_CPU.csv",
            'TF_GPU': PROJECT_ROOT / "src" / "libraries" / "tensorflow" / "metrics_tfNN_GPU.csv",
            'PT_GPU': PROJECT_ROOT / "src" / "libraries" / "Pytorch" / "metrics_ptNN_GPU.csv",
        }
        self.power_process = None
        self.power_csv_path = None
        
        EventSubscriptionController.subscribe_to_multiple_events([
            (RunnerEvents.BEFORE_EXPERIMENT, self.before_experiment),
            (RunnerEvents.BEFORE_RUN        , self.before_run        ),
            (RunnerEvents.START_RUN         , self.start_run         ),
            (RunnerEvents.START_MEASUREMENT , self.start_measurement),
            (RunnerEvents.INTERACT          , self.interact          ),
            (RunnerEvents.STOP_MEASUREMENT  , self.stop_measurement ),
            (RunnerEvents.STOP_RUN          , self.stop_run          ),
            (RunnerEvents.POPULATE_RUN_DATA , self.populate_run_data),
            (RunnerEvents.AFTER_EXPERIMENT  , self.after_experiment )
        ])
        self.run_table_model = None  # Initialized later
        output.console_log("Custom config loaded")

    def create_run_table_model(self) -> RunTableModel:
        """Creates a run table model that generates two blocks of runs:
        15 repetitions of TF_CPU/PT_CPU, followed by 15 repetitions of TF_GPU/PT_GPU."""

        # 1. Define Factors
        # Factor to control the block order (CPU runs first, then GPU runs)
        block_factor = FactorModel("Block_Type", ['CPU_Block'])#'GPU_Block'
        
        # Factor for the library type (used to ensure 15 reps of each)
        # Use only the base library names, as Block_Type handles the CPU/GPU suffix
        library_factor = FactorModel("Library_Name", ['TF', 'PT'])

        # 2. Define Exclusions (CRITICAL FOR BLOCK LOGIC)
        # The framework generates ALL combinations. We exclude the impossible/unwanted ones:
        exclude_combinations = [
            # Exclude running TF/PT in the CPU Block (redundant, handled by factor below)
            # {block_factor: ['CPU_Block'], library_factor: ['TF', 'PT']}, 
            
            # Exclude running TF/PT in the GPU Block (redundant, handled by factor below)
            # {block_factor: ['GPU_Block'], library_factor: ['TF', 'PT']},
        ]

        # 3. Create the RunTableModel
        self.run_table_model = RunTableModel(
            factors=[block_factor, library_factor],
            exclude_combinations=exclude_combinations,
            repetitions=15, # Total repetitions per unique combination of factors (15 * 2 * 2 = 60 runs)
            data_columns=[
                "Timestamp_Start",
                "Total_Energy_J",
                "CPU_Energy_J",
                "GPU_Energy_J",
                "Accuracy_Pct",
                "Model_Name"
            ]
        )
        return self.run_table_model

    def before_experiment(self) -> None:
        """Perform any activity required before starting the experiment here."""

        output.console_log("Config.before_experiment() called! Ensuring directories exist.")

        # We must ensure the raw power data directory exists before any runs start
        raw_power_dir = self.results_output_path / "raw_powerjoular_data"
        
        # You must have 'import os' at the top of the file for os.makedirs() to work
        if not raw_power_dir.exists():
            os.makedirs(raw_power_dir)

    # NOTE: No compilation is required as ML scripts are Python files

    
       

    def before_run(self) -> None:
        """Perform any activity required before starting a run.
        Note: Context is required by the framework, but often unused here."""
        
        # You can add logic here if you need to perform checks/setup for each run,
        # but based on your previous code, 'pass' is sufficient for now.
        pass

    def start_run(self, context: RunnerContext) -> None:
        """
        Performs activity required for starting the run.
        Starts the PowerJoular profiler and stores the script path for 'interact'.
        """

        # Fetch the factor values for the current run
        current_block = context.execute_run['Block_Type']
        current_library = context.execute_run['Library_Name']
        
        # Dynamically determine the full library code
        # This correctly generates 'TF_CPU', 'PT_GPU', etc.
        full_library_code = f"{current_library}_{current_block.split('_')[0]}" 
        
        # CRITICAL: Dynamically select the script path based on the generated factors
        # You can move this dictionary definition to __init__ if you want to clean up start_run
        LIBRARY_MAP = {
            'TF_CPU': self.ROOT_DIR / "src" / "libraries" / "tensorflow" / "tensorflow_neural_network.py",
            'PT_CPU': self.ROOT_DIR / "src" / "libraries" / "Pytorch" / "pytorch_neural_network.py",
            'TF_GPU': self.ROOT_DIR / "src" / "libraries" / "tensorflow" / "tensorflow_neural_network_GPU.py",
            'PT_GPU': self.ROOT_DIR / "src" / "libraries" / "Pytorch" / "pytorch_neural_network_GPU.py",
        }
        
        ml_script_path = LIBRARY_MAP.get(full_library_code)
        
        # --- Error Check (FIXED: removed unsupported 'is_error=True') ---
        if not ml_script_path:
            # Note: Removed is_error=True since it caused a TypeError earlier
            output.console_log(f"ERROR: Could not find script path for combination: {full_library_code}")
            raise ValueError(f"Invalid library combination: {full_library_code}")

        # --- 1. State Storage (CRITICAL) ---
        # Store the path and code for use in 'interact' and 'populate_run_data'
        self.current_ml_script_path = ml_script_path
        self.current_library_code = full_library_code

        # --- 2. Start PowerJoular (CRITICAL) ---
        # Define the output path for this specific run's raw power data
        self.power_csv_path = self.results_output_path / "raw_powerjoular_data" / f"{context.run_dir.name}.csv"

        output.console_log(f"--- Starting PowerJoular for {context.run_dir.name} ({full_library_code}) ---")
        
        try:
            # Use sudo as verified earlier, and start the process in the background
            self.power_process = subprocess.Popen(
                ['sudo', '/usr/bin/powerjoular', '-o', str(self.power_csv_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except FileNotFoundError:
             output.console_log("ERROR: powerjoular not found. Check installation and $PATH.")
             raise 
             
        # Allow PowerJoular a moment to initialize before the benchmark starts
        time.sleep(1)
        
    def start_measurement(self, context: RunnerContext) -> None:
        """Perform any activity required for starting measurements."""
        # Measurement starts when powerjoular is launched in start_run.
        pass
        



    def interact(self, context: RunnerContext) -> None:
        """Performs interaction with the running target system."""

        output.console_log(f"--- Running Benchmark: {self.current_library_code} ---")
        
        # Execute the ML script using the stored path
        try:
            subprocess.run(
                ['python', str(self.current_ml_script_path)],
                cwd=self.ROOT_DIR,
                check=True, 
                capture_output=False
            )
        except subprocess.CalledProcessError as e:
            output.console_log(f"ERROR: Benchmark script failed for run {context.run_dir.name}. Output: {e.output.decode() if e.output else 'No output'}")
            self.power_process.terminate() 
            raise 

        output.console_log("Program finished.")
        # No interaction. We just run it until the target finishes.

    def stop_measurement(self, context: RunnerContext) -> None:
        """Perform any activity here required for stopping measurements."""

        output.console_log(f"--- Stopping PowerJoular ---")
        if self.power_process:
            self.power_process.terminate() 
            self.power_process.wait()

    def stop_run(self, context: RunnerContext) -> None:
        """Perform any activity here required for stopping the run.
        Activities after stopping the run should also be performed here."""
        
        # NOTE: The required 30-second cooldown is handled automatically by 
        # `time_between_runs_in_ms = 30000` in the class variables.
        pass # No extra cleanup needed, already handled in stop_measurement
    
    def populate_run_data(self, context: RunnerContext) -> Optional[Dict[str, Any]]:
        """Parse and process any measurement data here.
        You can also store the raw measurement data under `context.run_dir`
        Returns a dictionary with keys `self.run_table_model.data_columns` and their values populated"""

        current_library = context.execute_run['Library_Name']
        current_block_value = context.execute_run['Block_Type']
        current_device = current_block_value.split('_')[0]
        full_benchmark_key = f"{current_library}_{current_device}"
        # --- 1. ENERGY CALCULATION (From unique PowerJoular CSV) ---
        try:
            df_power = pd.read_csv(self.power_csv_path, header=None)
            df_power.columns = ['Timestamp', 'Delta_T', 'Total_P', 'CPU_P', 'GPU_P']
            
            # Convert power (W) and time (s) to energy (J = W*s)
            cpu_energy = (df_power['CPU_P'] * df_power['Delta_T']).sum()
            gpu_energy = (df_power['GPU_P'] * df_power['Delta_T']).sum()
            total_energy = cpu_energy + gpu_energy
            timestamp = df_power['Timestamp'].iloc[0]
        except Exception as e:
            output.console_log(f"WARNING: Could not process PowerJoular data for {context.run_dir.name}. Error: {e}", is_error=True)
            total_energy, cpu_energy, gpu_energy, timestamp = 0.0, 0.0, 0.0, "FAILED"


        # --- 2. ACCURACY EXTRACTION (From the specific ML Metrics CSV - LAST ROW) ---
        metrics_log_path = self.metrics_log_paths[full_benchmark_key] # Use the hard-coded path

        try:
            df_metrics = pd.read_csv(metrics_log_path)
            
            # Select the very last row, which must correspond to the run that just completed
            final_row = df_metrics.iloc[-1] 
            
            # Extract and format metrics
            accuracy = final_row['accuracy']
            model_name = final_row['model']

        except Exception as e:
            output.console_log(f"WARNING: Could not process Metrics CSV at {metrics_log_path} for {context.run_dir.name}. Error: {e}", is_error=True)
            accuracy, model_name = 0.0, "FAILED"

        # --- 3. Return Data for the Experiment Runner Table ---
        return {
            "Timestamp_Start": timestamp,
            "Total_Energy_J": total_energy,
            "CPU_Energy_J": cpu_energy,
            "GPU_Energy_J": gpu_energy,
            "Accuracy_Pct": accuracy * 100, # Convert accuracy to percentage
            "Model_Name": model_name
        }

    def after_experiment(self) -> None:
        """Perform any activity required after stopping the experiment here
        Invoked only once during the lifetime of the program."""
        output.console_log("Experiment complete. Remember to manually initiate the long cooldown now.")
        pass

    # ================================ DO NOT ALTER BELOW THIS LINE ================================
    experiment_path:             Path              = None
