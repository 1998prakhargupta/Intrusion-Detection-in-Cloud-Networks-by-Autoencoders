#!/usr/bin/env python3
"""
NIDS Autoencoder - Performance Profiling Tool

This tool profiles the performance of the NIDS autoencoder system
across different scenarios and configurations.

Usage:
    python tools/performance_profiling.py --config config/performance.yaml
"""

import argparse
import time
import memory_profiler
import cProfile
import pstats
import io
from pathlib import Path
import sys
import logging
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import psutil
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.utils.logger import get_logger

logger = get_logger(__name__)

class PerformanceProfiler:
    """Performance profiling tool for NIDS autoencoder system."""
    
    def __init__(self, output_dir: str = "benchmarks/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
    def profile_memory_usage(self, func, *args, **kwargs):
        """Profile memory usage of a function."""
        logger.info(f"Profiling memory usage for {func.__name__}")
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Use memory_profiler for detailed analysis
        mem_usage = memory_profiler.memory_usage((func, args, kwargs), interval=0.1)
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        results = {
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'peak_memory_mb': max(mem_usage),
            'memory_diff_mb': final_memory - initial_memory,
            'memory_timeline': mem_usage
        }
        
        logger.info(f"Memory usage - Peak: {results['peak_memory_mb']:.2f} MB, "
                   f"Diff: {results['memory_diff_mb']:.2f} MB")
        
        return results
    
    def profile_execution_time(self, func, *args, **kwargs):
        """Profile execution time of a function."""
        logger.info(f"Profiling execution time for {func.__name__}")
        
        # Warm-up run
        _ = func(*args, **kwargs)
        
        # Multiple runs for accuracy
        times = []
        for i in range(5):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        execution_results = {
            'mean_time_seconds': np.mean(times),
            'std_time_seconds': np.std(times),
            'min_time_seconds': min(times),
            'max_time_seconds': max(times),
            'all_times': times
        }
        
        logger.info(f"Execution time - Mean: {execution_results['mean_time_seconds']:.4f}s ± "
                   f"{execution_results['std_time_seconds']:.4f}s")
        
        return result, execution_results
    
    def profile_cpu_usage(self, func, *args, **kwargs):
        """Profile CPU usage during function execution."""
        logger.info(f"Profiling CPU usage for {func.__name__}")
        
        # CPU profiling with cProfile
        profiler = cProfile.Profile()
        
        # Start monitoring CPU usage
        process = psutil.Process()
        cpu_percent_start = process.cpu_percent()
        
        profiler.enable()
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        profiler.disable()
        
        cpu_percent_end = process.cpu_percent()
        
        # Analyze profiling results
        stats_buffer = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_buffer)
        stats.sort_stats('cumulative').print_stats(20)
        
        cpu_results = {
            'cpu_percent_start': cpu_percent_start,
            'cpu_percent_end': cpu_percent_end,
            'execution_time': end_time - start_time,
            'profile_stats': stats_buffer.getvalue()
        }
        
        logger.info(f"CPU usage - Start: {cpu_percent_start:.1f}%, "
                   f"End: {cpu_percent_end:.1f}%")
        
        return result, cpu_results
    
    def create_synthetic_data(self, n_samples: int, n_features: int) -> np.ndarray:
        """Create synthetic network data for testing."""
        logger.info(f"Creating synthetic data: {n_samples} samples, {n_features} features")
        
        np.random.seed(42)
        
        # Normal traffic (80%)
        normal_size = int(n_samples * 0.8)
        normal_data = np.random.normal(0, 1, (normal_size, n_features))
        
        # Anomalous traffic (20%)
        anomaly_size = n_samples - normal_size
        anomaly_data = np.random.normal(2, 1.5, (anomaly_size, n_features))
        
        # Combine and shuffle
        data = np.vstack([normal_data, anomaly_data])
        labels = np.hstack([np.zeros(normal_size), np.ones(anomaly_size)])
        
        # Shuffle
        indices = np.random.permutation(len(data))
        return data[indices], labels[indices]
    
    def benchmark_data_preprocessing(self, data_sizes: List[int]):
        """Benchmark data preprocessing performance."""
        logger.info("Benchmarking data preprocessing performance")
        
        results = {}
        
        for size in data_sizes:
            logger.info(f"Testing preprocessing with {size} samples")
            
            # Create test data
            X, y = self.create_synthetic_data(size, 20)
            
            def preprocess_data():
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                return scaler.fit_transform(X)
            
            # Profile preprocessing
            _, time_results = self.profile_execution_time(preprocess_data)
            memory_results = self.profile_memory_usage(preprocess_data)
            
            results[size] = {
                'time': time_results,
                'memory': memory_results
            }
        
        self.results['preprocessing'] = results
        return results
    
    def benchmark_model_training(self, model_configs: List[Dict]):
        """Benchmark model training performance."""
        logger.info("Benchmarking model training performance")
        
        results = {}
        
        # Create training data
        X_train, _ = self.create_synthetic_data(10000, 20)
        
        for i, config in enumerate(model_configs):
            config_name = f"config_{i+1}"
            logger.info(f"Testing training with {config_name}")
            
            def train_model():
                # Simulate model training
                from sklearn.neural_network import MLPRegressor
                model = MLPRegressor(
                    hidden_layer_sizes=config.get('hidden_layers', (50, 25)),
                    max_iter=config.get('epochs', 100),
                    random_state=42
                )
                return model.fit(X_train, X_train)  # Autoencoder-style training
            
            # Profile training
            _, time_results = self.profile_execution_time(train_model)
            memory_results = self.profile_memory_usage(train_model)
            
            results[config_name] = {
                'config': config,
                'time': time_results,
                'memory': memory_results
            }
        
        self.results['training'] = results
        return results
    
    def benchmark_inference(self, batch_sizes: List[int]):
        """Benchmark inference performance."""
        logger.info("Benchmarking inference performance")
        
        results = {}
        
        # Create and train a simple model
        from sklearn.neural_network import MLPRegressor
        X_train, _ = self.create_synthetic_data(5000, 20)
        model = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=50, random_state=42)
        model.fit(X_train, X_train)
        
        for batch_size in batch_sizes:
            logger.info(f"Testing inference with batch size {batch_size}")
            
            # Create test data
            X_test, _ = self.create_synthetic_data(batch_size, 20)
            
            def run_inference():
                return model.predict(X_test)
            
            # Profile inference
            _, time_results = self.profile_execution_time(run_inference)
            memory_results = self.profile_memory_usage(run_inference)
            
            # Calculate throughput
            throughput = batch_size / time_results['mean_time_seconds']
            
            results[batch_size] = {
                'time': time_results,
                'memory': memory_results,
                'throughput_samples_per_second': throughput
            }
        
        self.results['inference'] = results
        return results
    
    def generate_performance_report(self):
        """Generate comprehensive performance report."""
        logger.info("Generating performance report")
        
        report_path = self.output_dir / "performance_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# NIDS Autoencoder Performance Report\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # System information
            f.write("## System Information\n\n")
            f.write(f"- CPU Count: {psutil.cpu_count()}\n")
            f.write(f"- Total Memory: {psutil.virtual_memory().total / 1024**3:.2f} GB\n")
            f.write(f"- Python Version: {sys.version}\n\n")
            
            # Performance results
            if 'preprocessing' in self.results:
                f.write("## Data Preprocessing Performance\n\n")
                f.write("| Data Size | Mean Time (s) | Memory Usage (MB) |\n")
                f.write("|-----------|---------------|-------------------|\n")
                for size, results in self.results['preprocessing'].items():
                    time_mean = results['time']['mean_time_seconds']
                    memory_peak = results['memory']['peak_memory_mb']
                    f.write(f"| {size:,} | {time_mean:.4f} | {memory_peak:.2f} |\n")
                f.write("\n")
            
            if 'training' in self.results:
                f.write("## Model Training Performance\n\n")
                f.write("| Configuration | Mean Time (s) | Memory Usage (MB) |\n")
                f.write("|---------------|---------------|-------------------|\n")
                for config_name, results in self.results['training'].items():
                    time_mean = results['time']['mean_time_seconds']
                    memory_peak = results['memory']['peak_memory_mb']
                    f.write(f"| {config_name} | {time_mean:.4f} | {memory_peak:.2f} |\n")
                f.write("\n")
            
            if 'inference' in self.results:
                f.write("## Inference Performance\n\n")
                f.write("| Batch Size | Mean Time (s) | Throughput (samples/s) | Memory Usage (MB) |\n")
                f.write("|------------|---------------|------------------------|-------------------|\n")
                for batch_size, results in self.results['inference'].items():
                    time_mean = results['time']['mean_time_seconds']
                    throughput = results['throughput_samples_per_second']
                    memory_peak = results['memory']['peak_memory_mb']
                    f.write(f"| {batch_size} | {time_mean:.4f} | {throughput:.0f} | {memory_peak:.2f} |\n")
                f.write("\n")
        
        logger.info(f"Performance report saved to: {report_path}")
    
    def create_performance_plots(self):
        """Create performance visualization plots."""
        logger.info("Creating performance plots")
        
        plt.style.use('seaborn-v0_8')
        
        # Preprocessing performance plot
        if 'preprocessing' in self.results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            sizes = list(self.results['preprocessing'].keys())
            times = [self.results['preprocessing'][size]['time']['mean_time_seconds'] for size in sizes]
            memories = [self.results['preprocessing'][size]['memory']['peak_memory_mb'] for size in sizes]
            
            ax1.plot(sizes, times, 'o-', linewidth=2, markersize=8)
            ax1.set_xlabel('Data Size')
            ax1.set_ylabel('Execution Time (seconds)')
            ax1.set_title('Preprocessing Time vs Data Size')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(sizes, memories, 'o-', linewidth=2, markersize=8, color='red')
            ax2.set_xlabel('Data Size')
            ax2.set_ylabel('Peak Memory Usage (MB)')
            ax2.set_title('Memory Usage vs Data Size')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'preprocessing_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Inference performance plot
        if 'inference' in self.results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            batch_sizes = list(self.results['inference'].keys())
            throughputs = [self.results['inference'][size]['throughput_samples_per_second'] for size in batch_sizes]
            times = [self.results['inference'][size]['time']['mean_time_seconds'] for size in batch_sizes]
            
            ax1.plot(batch_sizes, throughputs, 'o-', linewidth=2, markersize=8, color='green')
            ax1.set_xlabel('Batch Size')
            ax1.set_ylabel('Throughput (samples/second)')
            ax1.set_title('Inference Throughput vs Batch Size')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(batch_sizes, times, 'o-', linewidth=2, markersize=8, color='purple')
            ax2.set_xlabel('Batch Size')
            ax2.set_ylabel('Execution Time (seconds)')
            ax2.set_title('Inference Time vs Batch Size')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'inference_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Performance plots saved to: {self.output_dir}")

def main():
    """Main function for performance profiling."""
    parser = argparse.ArgumentParser(description="NIDS Autoencoder Performance Profiling Tool")
    parser.add_argument("--output-dir", default="benchmarks/results", 
                       help="Output directory for results")
    parser.add_argument("--data-sizes", nargs='+', type=int, 
                       default=[1000, 5000, 10000, 50000],
                       help="Data sizes to test for preprocessing")
    parser.add_argument("--batch-sizes", nargs='+', type=int,
                       default=[1, 10, 50, 100, 500],
                       help="Batch sizes to test for inference")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting NIDS Autoencoder Performance Profiling")
    
    # Initialize profiler
    profiler = PerformanceProfiler(args.output_dir)
    
    try:
        # Benchmark preprocessing
        profiler.benchmark_data_preprocessing(args.data_sizes)
        
        # Benchmark training with different configurations
        model_configs = [
            {'hidden_layers': (25, 10), 'epochs': 50},
            {'hidden_layers': (50, 25), 'epochs': 100},
            {'hidden_layers': (100, 50, 25), 'epochs': 100}
        ]
        profiler.benchmark_model_training(model_configs)
        
        # Benchmark inference
        profiler.benchmark_inference(args.batch_sizes)
        
        # Generate report and plots
        profiler.generate_performance_report()
        profiler.create_performance_plots()
        
        logger.info("✅ Performance profiling completed successfully!")
        logger.info(f"📊 Results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Error during performance profiling: {str(e)}")
        raise

if __name__ == "__main__":
    main()
