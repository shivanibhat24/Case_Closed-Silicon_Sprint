#!/usr/bin/env python3
"""
File: trojan_detector.py
Description: Advanced side-channel analysis for hardware Trojan detection
Author: Hardware Security Lab
Date: November 2025

This script analyzes VCD files from clean and Trojan-infected designs,
performs toggle-based side-channel analysis, and visualizes deviations.
"""

import re
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class VCDParser:
    """Parse VCD files and extract switching activity data"""
    
    def __init__(self, vcd_file):
        self.vcd_file = vcd_file
        self.signal_map = {}  # Maps signal codes to signal names
        self.toggle_counts = defaultdict(int)
        self.signal_values = {}  # Track previous values for toggle detection
        
    def parse(self):
        """Parse VCD file and count toggles"""
        print(f"[*] Parsing VCD file: {self.vcd_file}")
        
        try:
            with open(self.vcd_file, 'r') as f:
                in_header = True
                
                for line in f:
                    line = line.strip()
                    
                    # Parse variable declarations in header
                    if in_header:
                        if line.startswith('$var'):
                            self._parse_var_declaration(line)
                        elif line.startswith('$enddefinitions'):
                            in_header = False
                            print(f"[+] Found {len(self.signal_map)} signals")
                    else:
                        # Parse value changes
                        self._parse_value_change(line)
            
            print(f"[+] Parsing complete. Total toggles detected: {sum(self.toggle_counts.values())}")
            return self.toggle_counts
            
        except FileNotFoundError:
            print(f"[!] ERROR: File not found: {self.vcd_file}")
            sys.exit(1)
        except Exception as e:
            print(f"[!] ERROR parsing VCD: {e}")
            sys.exit(1)
    
    def _parse_var_declaration(self, line):
        """Extract signal name and code from $var declaration"""
        # Format: $var wire 1 ! clk $end
        parts = line.split()
        if len(parts) >= 5:
            signal_code = parts[3]
            signal_name = parts[4]
            
            # Build hierarchical name for nested signals
            if '.' in signal_name or '[' in signal_name:
                self.signal_map[signal_code] = signal_name
            else:
                self.signal_map[signal_code] = signal_name
            
            # Initialize toggle counter
            self.toggle_counts[signal_name] = 0
    
    def _parse_value_change(self, line):
        """Detect value changes and count toggles"""
        if not line or line.startswith('$') or line.startswith('#'):
            return
        
        # Parse different VCD value change formats
        # Format 1: "0!" (value + code)
        # Format 2: "b1010 #" (binary value + code)
        
        if line[0] in ['0', '1', 'x', 'z']:
            # Single-bit change: value is first char, code is rest
            value = line[0]
            code = line[1:]
            
            if code in self.signal_map:
                signal_name = self.signal_map[code]
                
                # Check if value changed (toggle detection)
                if signal_name in self.signal_values:
                    if self.signal_values[signal_name] != value:
                        self.toggle_counts[signal_name] += 1
                
                self.signal_values[signal_name] = value
        
        elif line.startswith('b'):
            # Multi-bit change: bXXXX code
            parts = line.split()
            if len(parts) >= 2:
                value = parts[0][1:]  # Remove 'b' prefix
                code = parts[1]
                
                if code in self.signal_map:
                    signal_name = self.signal_map[code]
                    
                    # Count toggles in multi-bit signals
                    if signal_name in self.signal_values:
                        old_val = self.signal_values[signal_name]
                        if old_val != value:
                            # Count number of bit positions that changed
                            toggles = self._count_bit_differences(old_val, value)
                            self.toggle_counts[signal_name] += toggles
                    
                    self.signal_values[signal_name] = value
    
    def _count_bit_differences(self, val1, val2):
        """Count number of bit positions that differ between two binary strings"""
        # Pad to same length
        max_len = max(len(val1), len(val2))
        val1 = val1.zfill(max_len)
        val2 = val2.zfill(max_len)
        
        count = 0
        for b1, b2 in zip(val1, val2):
            if b1 != b2 and b1 in ['0', '1'] and b2 in ['0', '1']:
                count += 1
        return count


class TrojanDetector:
    """Analyze and compare toggle counts to detect hardware Trojans"""
    
    def __init__(self, clean_toggles, trojan_toggles, threshold=25.0):
        self.clean_toggles = clean_toggles
        self.trojan_toggles = trojan_toggles
        self.threshold = threshold
        self.suspicious_signals = []
        
    def analyze(self):
        """Compare toggle counts and identify suspicious signals"""
        print(f"\n[*] Analyzing toggle count deviations (threshold: {self.threshold}%)")
        
        # Find common signals
        common_signals = set(self.clean_toggles.keys()) & set(self.trojan_toggles.keys())
        print(f"[+] Comparing {len(common_signals)} common signals")
        
        results = []
        
        for signal in sorted(common_signals):
            clean_count = self.clean_toggles[signal]
            trojan_count = self.trojan_toggles[signal]
            
            # Calculate percentage deviation
            if clean_count == 0:
                if trojan_count > 0:
                    deviation = float('inf')
                else:
                    deviation = 0.0
            else:
                deviation = abs(trojan_count - clean_count) / clean_count * 100
            
            results.append({
                'signal': signal,
                'clean': clean_count,
                'trojan': trojan_count,
                'deviation': deviation,
                'suspicious': deviation > self.threshold
            })
            
            if deviation > self.threshold:
                self.suspicious_signals.append(signal)
        
        return results
    
    def print_report(self, results):
        """Print detailed analysis report"""
        print("\n" + "="*80)
        print("HARDWARE TROJAN DETECTION REPORT")
        print("="*80)
        
        print(f"\nDetection Threshold: {self.threshold}% deviation")
        print(f"Suspicious Signals Found: {len(self.suspicious_signals)}")
        
        if self.suspicious_signals:
            print(f"\n{'Signal Name':<30} {'Clean':<10} {'Trojan':<10} {'Deviation':<12} {'Status'}")
            print("-"*80)
            
            for result in results:
                if result['suspicious']:
                    print(f"{result['signal']:<30} "
                          f"{result['clean']:<10} "
                          f"{result['trojan']:<10} "
                          f"{result['deviation']:>10.2f}% "
                          f"{'⚠ SUSPICIOUS' if result['suspicious'] else 'OK'}")
        
        print("\n" + "="*80)
        
        # Detailed statistics
        total_clean = sum(r['clean'] for r in results)
        total_trojan = sum(r['trojan'] for r in results)
        
        print(f"\nTotal Toggle Activity:")
        print(f"  Clean Design:  {total_clean:,} toggles")
        print(f"  Trojan Design: {total_trojan:,} toggles")
        print(f"  Difference:    {total_trojan - total_clean:,} toggles "
              f"({(total_trojan - total_clean)/total_clean*100:+.2f}%)")
        
        print("\n" + "="*80 + "\n")


class TrojanVisualizer:
    """Create visualizations for Trojan detection analysis"""
    
    def __init__(self, results, threshold):
        self.results = results
        self.threshold = threshold
    
    def create_comparison_plot(self):
        """Create bar chart comparing toggle counts"""
        print("[*] Generating comparison visualization...")
        
        # Filter to show only signals with significant activity
        significant_results = [r for r in self.results if r['clean'] > 10 or r['trojan'] > 10]
        
        if not significant_results:
            print("[!] No significant signals to plot")
            return
        
        # Sort by deviation
        significant_results.sort(key=lambda x: x['deviation'], reverse=True)
        
        # Limit to top 20 for readability
        plot_results = significant_results[:20]
        
        signals = [r['signal'] for r in plot_results]
        clean_counts = [r['clean'] for r in plot_results]
        trojan_counts = [r['trojan'] for r in plot_results]
        suspicious = [r['suspicious'] for r in plot_results]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Side-by-side comparison
        x = np.arange(len(signals))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, clean_counts, width, label='Clean Design', 
                        color='#2ecc71', alpha=0.8)
        bars2 = ax1.bar(x + width/2, trojan_counts, width, label='Trojan Design',
                        color='#e74c3c', alpha=0.8)
        
        ax1.set_xlabel('Signal Name', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Toggle Count', fontsize=11, fontweight='bold')
        ax1.set_title('Toggle Activity Comparison: Clean vs Trojan Design', 
                     fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(signals, rotation=45, ha='right', fontsize=9)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Highlight suspicious signals
        for i, susp in enumerate(suspicious):
            if susp:
                ax1.axvspan(i - 0.5, i + 0.5, alpha=0.2, color='yellow')
                ax1.text(i, max(clean_counts[i], trojan_counts[i]) * 1.05, 
                        '⚠', ha='center', fontsize=14, color='red')
        
        # Plot 2: Deviation percentage
        deviations = [r['deviation'] for r in plot_results]
        colors = ['#e74c3c' if d > self.threshold else '#95a5a6' for d in deviations]
        
        bars3 = ax2.bar(x, deviations, color=colors, alpha=0.8)
        ax2.axhline(y=self.threshold, color='red', linestyle='--', 
                   linewidth=2, label=f'Threshold ({self.threshold}%)')
        
        ax2.set_xlabel('Signal Name', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Deviation (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Toggle Count Deviation Analysis', 
                     fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(signals, rotation=45, ha='right', fontsize=9)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save figure
        output_file = 'trojan_detection_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"[+] Visualization saved: {output_file}")
        
        plt.show()


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("HARDWARE TROJAN SIDE-CHANNEL DETECTOR")
    print("Advanced Toggle-Based Analysis for Security Verification")
    print("="*80 + "\n")
    
    # Configuration
    vcd_file = "alu_simulation.vcd"
    detection_threshold = 25.0  # 25% deviation threshold
    
    # Check if VCD file exists
    if not Path(vcd_file).exists():
        print(f"[!] ERROR: VCD file not found: {vcd_file}")
        print("[!] Please run the Verilog simulation first to generate the VCD file.")
        sys.exit(1)
    
    # Parse VCD files
    print("[Phase 1] VCD File Parsing")
    print("-" * 80)
    
    parser = VCDParser(vcd_file)
    all_toggles = parser.parse()
    
    # Separate clean and trojan toggle counts
    # Signals containing 'clean' belong to clean design
    # Signals containing 'trojan' belong to trojan design
    clean_toggles = {k: v for k, v in all_toggles.items() if 'clean' in k.lower()}
    trojan_toggles = {k: v for k, v in all_toggles.items() if 'trojan' in k.lower()}
    
    # Normalize signal names (remove module prefix)
    clean_normalized = {}
    trojan_normalized = {}
    
    for sig, count in clean_toggles.items():
        normalized = sig.replace('uut_clean.', '').replace('alu_clean.', '')
        clean_normalized[normalized] = count
    
    for sig, count in trojan_toggles.items():
        normalized = sig.replace('uut_trojan.', '').replace('alu_trojan.', '')
        trojan_normalized[normalized] = count
    
    # Analyze and detect Trojans
    print("\n[Phase 2] Trojan Detection Analysis")
    print("-" * 80)
    
    detector = TrojanDetector(clean_normalized, trojan_normalized, detection_threshold)
    results = detector.analyze()
    detector.print_report(results)
    
    # Generate visualizations
    print("[Phase 3] Visualization Generation")
    print("-" * 80)
    
    visualizer = TrojanVisualizer(results, detection_threshold)
    visualizer.create_comparison_plot()
    
    # Final summary
    print("\n[✓] Analysis Complete!")
    print(f"[✓] Suspicious signals detected: {len(detector.suspicious_signals)}")
    print(f"[✓] Report and visualization generated successfully.\n")


if __name__ == "__main__":
    main()
