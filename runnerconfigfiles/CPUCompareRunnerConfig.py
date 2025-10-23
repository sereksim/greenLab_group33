# ================================ FRAMEWORK IMPORTS ================================
from EventManager.Models.RunnerEvents import RunnerEvents
from EventManager.EventSubscriptionController import EventSubscriptionController
from ConfigValidator.Config.Models.RunTableModel import RunTableModel
from ConfigValidator.Config.Models.FactorModel import FactorModel
from ConfigValidator.Config.Models.RunnerContext import RunnerContext
from ConfigValidator.Config.Models.OperationType import OperationType
from Plugins.Profilers.PowerJoular import PowerJoular
from ProgressManager.Output.OutputProcedure import OutputProcedure as Output

from typing import Dict, Any, Optional
from pathlib import Path

# ================================ CUSTOM IMPORTS ================================
import time
import subprocess
import pandas as pd
import os

# --- CONFIGURATION CONSTANTS ---
PROJECT_ROOT = Path("/media/user/data_usb/green_lab_pip")
OUTPUT_DIR = PROJECT_ROOT / "results" / "RaspPi"
PATH_CANCER = PROJECT_ROOT / "data" / "yasserh" / "breast-cancer-dataset" / "versions" / "1" / "breast-cancer.csv"


# ================================ FRAMEWORK CLASS ================================

class RunnerConfig:
    ROOT_DIR = PROJECT_ROOT  # Set ROOT_DIR to the actual project root

    # ================================ USER SPECIFIC CONFIG ================================
    """The name of the experiment."""
    name: str = "ML_Energy_Comparison_CPU_Only"

    """The path in which Experiment Runner will create a folder with the name `self.name`, in order to store the
    results from this experiment. (Path does not need to exist - it will be created if necessary.)
    Output path defaults to the config file's path, inside the folder 'experiments'"""
    results_output_path: Path = OUTPUT_DIR  # Use our custom output directory

    """Experiment operation type. Unless you manually want to initiate each run, use `OperationType.AUTO`."""
    operation_type: OperationType = OperationType.AUTO

    """The time Experiment Runner will wait after a run completes.
    This can be essential to accommodate for cooldown periods on some systems."""
    time_between_runs_in_ms: int = 15000

    # repetitions per valid combination of library and model
    repetitions = 15
    block_size = 45  # runs before cooldown pause
    block_cooldown_in_sec = 300  # 5 min gap between blocks

    # Dynamic configurations can be one-time satisfied here before the program takes the config as-is
    # e.g. Setting some variable based on some criteria
    def __init__(self):
        """Executes immediately after program start, on config load"""
        Output.console_log(f"--- Initialize RunnerConfig ---")

        # Metrics paths for each library-model combination
        self.metrics_log_paths = {
            'TensorFlow_linReg': PROJECT_ROOT / "src" / "libraries" / "tensorflow" / "metrics_tfReg.csv",
            'PyTorch_linReg': PROJECT_ROOT / "src" / "libraries" / "Pytorch" / "metrics_ptLR.csv",
            'TensorFlow_logReg': PROJECT_ROOT / "src" / "libraries" / "tensorflow" / "metrics_logistic.csv",
            'PyTorch_logReg': PROJECT_ROOT / "src" / "libraries" / "Pytorch" / "metrics_ptLogR.csv",
            'sklearn_linReg': PROJECT_ROOT / "src" / "libraries" / "scikit-learn" / "metrics_linReg.csv",
            'sklearn_logReg': PROJECT_ROOT / "src" / "libraries" / "scikit-learn" / "metrics_logReg.csv",
        }

        # Script paths for each library-model combination
        self.script_paths = {
            'TensorFlow_linReg': PROJECT_ROOT / "src" / "libraries" / "tensorflow" / "tensorflow_linear_regression.py",
            'PyTorch_linReg': PROJECT_ROOT / "src" / "libraries" / "Pytorch" / "pytorch_linear_regression.py",
            'TensorFlow_logReg': PROJECT_ROOT / "src" / "libraries" / "tensorflow" / "tensorflow_logistic_regression.py",
            'PyTorch_logReg': PROJECT_ROOT / "src" / "libraries" / "Pytorch" / "pytorch_logistic_regression.py",
            'sklearn_linReg': PROJECT_ROOT / "src" / "libraries" / "scikit-learn" / "scikit-learn-linReg.py",
            'sklearn_logReg': PROJECT_ROOT / "src" / "libraries" / "scikit-learn" / "scikit-learn-logReg.py",
        }
        self.power_process = None
        self.power_csv_path = None

        self.completed_runs = 0

        EventSubscriptionController.subscribe_to_multiple_events([
            (RunnerEvents.BEFORE_EXPERIMENT, self.before_experiment),
            (RunnerEvents.BEFORE_RUN, self.before_run),
            (RunnerEvents.START_RUN, self.start_run),
            (RunnerEvents.START_MEASUREMENT, self.start_measurement),
            (RunnerEvents.INTERACT, self.interact),
            (RunnerEvents.STOP_MEASUREMENT, self.stop_measurement),
            (RunnerEvents.STOP_RUN, self.stop_run),
            (RunnerEvents.POPULATE_RUN_DATA, self.populate_run_data),
            (RunnerEvents.AFTER_EXPERIMENT, self.after_experiment)
        ])
        self.run_table_model = None  # Initialized later
        Output.console_log("Custom config loaded")

    def create_run_table_model(self) -> RunTableModel:
        """Creates a run table model for CPU-only library comparison."""
        Output.console_log(f"--- Create run table (\"create_run_table_model()\")---")
        # Define factors for library and model type
        library_factor = FactorModel("Library", ['TensorFlow', 'PyTorch', 'sklearn'])
        model_factor = FactorModel("Model_Type", ['linReg', 'logReg'])
        data_columns = [
            "Timestamp_Start",
            "Duration_ns",
            "CPU_Energy_J",
            "Accuracy_Pct",
            "Precision_Pct",
            "MSE",
            "r2_score",
            "Model_Name"
        ]
        self.run_table_model = RunTableModel(
            factors=[library_factor, model_factor],
            repetitions=self.repetitions,  # number of repetitions per valid combination
            data_columns=data_columns,
            shuffle=True  # randomize order
        )
        return self.run_table_model

    def before_experiment(self) -> None:
        """Perform any activity required before starting the experiment here."""

        Output.console_log("--- Prepare Experiment (\"before_experiment()\")")

        # Ensure raw power data directory exists
        raw_power_dir = self.results_output_path / "raw_powerjoular_data"
        os.makedirs(raw_power_dir, exist_ok=True)

        Output.console_log(
            f"Experiment will run in {self.repetitions} randomized repetitions per run, {self.block_size} runs per block, "
            f"with {self.block_cooldown_in_sec / 60:.0f} min cooldowns between blocks.")

    def before_run(self) -> None:
        """Perform any activity required before starting a run."""
        pass

    def start_run(self, context: RunnerContext) -> None:
        """Start the run, including PowerJoular profiler."""
        Output.console_log(f"--- Start Run (\"start_run()\")---")
        # Get current run's library and model type
        current_library = context.execute_run['Library']
        current_model = context.execute_run['Model_Type']

        # Create the combined key
        library_model_key = f"{current_library}_{current_model}"

        # Get the script path
        ml_script_path = self.script_paths.get(library_model_key)

        if not ml_script_path:
            Output.console_log(f"ERROR: No script found for {library_model_key}")
            raise ValueError(f"Invalid library-model combination: {library_model_key}")

        # Store for later use
        self.current_ml_script_path = ml_script_path
        self.current_library_model = library_model_key

        self.power_csv_path = self.results_output_path / "raw_powerjoular_data" / f"{context.run_dir.name}.csv"

        Output.console_log(f"--- Starting {context.run_dir.name} ({library_model_key}) ---")

        try:
            self.start = time.perf_counter_ns()
            self.target = subprocess.Popen(
                ['python', str(ml_script_path)],
                cwd=self.ROOT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            Output.console_log(f"Started target process with PID: {self.target.pid}")
        except FileNotFoundError as e:
            Output.console_log(f"ERROR: Script not found at {ml_script_path}")
            raise


    def start_measurement(self, context: RunnerContext) -> None:
        """Perform any activity required for starting measurements."""
        Output.console_log(f"--- Starting PowerJoular (\"start_measurement()\")---")
        # Set up the powerjoular object, provide an (optional) target and output file name
        self.meter = PowerJoular(out_file=self.power_csv_path, target_pid=self.target.pid)
        # Start measuring with powerjoular
        self.meter.start()
        Output.console_log(f"PowerJoular monitoring PID {self.target.pid}")

    def interact(self, context: RunnerContext) -> None:
        """Wait for the ML script to finish."""

        # Wait for process to finish and capture output
        stdout, stderr = self.target.communicate()
        self.duration = time.perf_counter_ns() - self.start

        # Check if it completed successfully
        if self.target.returncode != 0:
            Output.console_log(f"ERROR: Benchmark failed with return code {self.target.returncode}")
            Output.console_log(f"STDOUT: {stdout.decode('utf-8', errors='replace')}")
            Output.console_log(f"STDERR: {stderr.decode('utf-8', errors='replace')}")
            raise subprocess.CalledProcessError(
                self.target.returncode,
                self.current_ml_script_path,
                output=stdout,
                stderr=stderr
            )

    def stop_measurement(self, context: RunnerContext) -> None:
        """Perform any activity here required for stopping measurements."""
        self.meter.stop()
        Output.console_log(f"--- Stopping PowerJoular (\"stop_measurement()\")---")


    def stop_run(self, context: RunnerContext) -> None:
        """Perform any activity here required for stopping the run.
        Activities after stopping the run should also be performed here."""
        if self.target.poll() is None:
            self.target.kill()
            self.target.wait()
        self.completed_runs += 1
        Output.console_log(f"Completed run #{self.completed_runs} (\"stop_run()\")")

        # Every block_size runs, take a long pause
        if self.completed_runs % self.block_size == 0 and self.completed_runs < 150:
            Output.console_log(
                f"=== Block {self.completed_runs // self.block_size} finished. "
                f"Cooling down for {self.block_cooldown_in_sec / 60:.0f} minutes... ==="
            )
            time.sleep(self.block_cooldown_in_sec)

    def populate_run_data(self, context: RunnerContext) -> Optional[Dict[str, Any]]:
        """Parse and process any measurement data here.
        You can also store the raw measurement data under `context.run_dir`
        Returns a dictionary with keys `self.run_table_model.data_columns` and their values populated"""
        Output.console_log(f"--- Saving Measurements (\"populate_run_data()\")---")

        # --- 1. ENERGY CALCULATION (From unique PowerJoular CSV) ---
        duration = -999999
        try:
            df_power = self.meter.parse_log(self.meter.target_logfile)
            print(df_power.info())
            print(df_power.info)
            df_power['Date'] = pd.to_datetime(df_power['Date'])  # TODO: explicitly specify unit ms?
            df_power['Delta_T'] = df_power['Date'].diff()
            #print(df_power.info)
            # First row will have NaN delta, because there is no start-timestamp to calculate duration
            mean_delta = df_power['Delta_T'].iloc[1:].mean()  # ignore NaN in first
            df_power.loc[0, 'Delta_T'] = mean_delta  # set delta of first row to average

            # Calculate energy (J = W * s)
            delta_t_seconds = df_power['Delta_T'].dt.total_seconds()
            cpu_energy = (df_power['CPU Power'] * delta_t_seconds).sum()
            timestamp = df_power['Date'].iloc[0]
            duration = self.duration
            #print(df_power.info)
        except Exception as e:
            Output.console_log(
                f"WARNING: Error in processing PowerJoular data for {context.run_dir.name}. Error: "
                f"{e}\n{e.__traceback__}")
            cpu_energy, timestamp = 0.0, "FAILED"

        # --- 2. ACCURACY EXTRACTION ---
        metrics_log_path = self.metrics_log_paths[self.current_library_model]
        accuracy = -999999
        precision = -999999
        MSE = -999999
        r2_score = -999999
        try:
            df_metrics = pd.read_csv(metrics_log_path)
            #print(df_metrics.info())
            # Select last row, the run that just completed
            final_row = df_metrics.iloc[-1]
            measurements_class, measurements_reg = False, False
            if 'accuracy' in df_metrics.columns:
                accuracy = final_row['accuracy'] * 100  # Convert accuracy to percentage
                measurements_class = True
            if 'precision' in df_metrics.columns:
                precision = final_row['precision'] * 100  # Convert precision to percentage
                measurements_class = True
            if 'MSE' in df_metrics.columns:
                MSE = final_row['MSE']
                measurements_reg = True
            if 'r2_score' in df_metrics.columns:
                r2_score = final_row['r2_score']
                measurements_reg = True
            if not measurements_reg and not measurements_class:
                raise Exception("Missing measurements for accuracy etc.")
            model_name = final_row['model']
        except Exception as e:
            Output.console_log(
                f"WARNING: Could not process Metrics CSV at {metrics_log_path} for {context.run_dir.name}. Error: {e}\n{e.__traceback__}")
            model_name = "FAILED"

        # --- 3. Return Data for the Experiment Runner Table ---
        return {
            "Timestamp_Start": timestamp,
            "Duration_ns":duration,
            "CPU_Energy_J": cpu_energy,
            "Accuracy_Pct": accuracy,
            "Precision_Pct": precision,
            "MSE": MSE,
            "r2_score": r2_score,
            "Model_Name": model_name
        }

    def after_experiment(self) -> None:
        Output.console_log("Experiment complete. (\"after_experiment()\")")
        pass

    # ================================ DO NOT ALTER BELOW THIS LINE ================================
    experiment_path: Path = None
