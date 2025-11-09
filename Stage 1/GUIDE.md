# Complete Usage Guide
## Hardware Trojan Detection Project

---

## Table of Contents
1. [Platform Setup](#platform-setup)
2. [Vivado Workflow](#vivado-workflow)
3. [Alternative Simulators](#alternative-simulators)
4. [Analysis Execution](#analysis-execution)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [Performance Optimization](#performance-optimization)

---

## Platform Setup

### Windows Setup

#### Prerequisites
1. **Install Vivado Design Suite**
   - Download from Xilinx website (Free WebPACK Edition)
   - Minimum version: 2020.1
   - Installation size: ~20GB

2. **Install Python**
   ```powershell
   # Download Python 3.9+ from python.org
   # During installation, CHECK "Add Python to PATH"
   
   # Verify installation
   python --version
   pip --version
   ```

3. **Install Dependencies**
   ```powershell
   pip install numpy matplotlib
   ```

#### Environment Setup
```powershell
# Set Vivado environment variables (if not automatic)
cd C:\Xilinx\Vivado\2023.2
.\settings64.bat

# Verify Vivado
vivado -version
```

---

### Linux Setup

#### Prerequisites
1. **Install Vivado**
   ```bash
   # Download installer from Xilinx
   chmod +x Xilinx_Unified_2023.2_1013_2256_Lin64.bin
   ./Xilinx_Unified_2023.2_1013_2256_Lin64.bin
   ```

2. **Install Python and Dependencies**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3 python3-pip
   
   # CentOS/RHEL
   sudo yum install python3 python3-pip
   
   # Install packages
   pip3 install numpy matplotlib
   ```

3. **Environment Configuration**
   ```bash
   # Add to ~/.bashrc or ~/.zshrc
   source /tools/Xilinx/Vivado/2023.2/settings64.sh
   
   # Apply changes
   source ~/.bashrc
   ```

---

### macOS Setup

#### Prerequisites
1. **Note**: Vivado doesn't officially support macOS for synthesis
   - Use virtual machine (VMware/Parallels) with Linux
   - Or use Icarus Verilog (simulation only)

2. **Install Python (Homebrew)**
   ```bash
   brew install python3
   pip3 install numpy matplotlib
   ```

3. **Install Icarus Verilog**
   ```bash
   brew install icarus-verilog
   iverilog -V  # Verify installation
   ```

---

## Vivado Workflow

### Method 1: GUI-Based Workflow (Recommended for Beginners)

#### Step 1: Create Project

1. **Launch Vivado**
   ```
   Start → Xilinx Design Tools → Vivado 2023.2
   ```

2. **Create New Project**
   - Click "Create Project"
   - Project name: `hardware_trojan_detection`
   - Location: Choose your directory
   - Project type: **RTL Project**
   - ☑ Do not specify sources at this time

3. **Select Device** (for simulation, any device works)
   - Family: Artix-7
   - Part: xc7a35tcpg236-1
   - Click Finish

#### Step 2: Add Source Files

1. **Add Design Files**
   - Flow Navigator → PROJECT MANAGER → Add Sources
   - Select "Add or create design sources"
   - Add Files:
     - `alu_clean.v`
     - `alu_trojan.v`
   - Click Finish

2. **Add Testbench**
   - Add Sources → "Add or create simulation sources"
   - Add File: `alu_tb.v`
   - Click Finish

#### Step 3: Configure Simulation

1. **Set Simulation Top Module**
   - Sources window → Simulation Sources
   - Right-click `alu_tb` → Set as Top

2. **Simulation Settings**
   - Tools → Settings → Simulation
   - Simulation top module name: `alu_tb`
   - Simulation run time: `1000us` (default is fine)
   - ☑ Enable VCD file generation:
     - More Options → `xelab.more_options`: `-debug all`
     - More Options → `xsim.simulate.log_all_signals`: `true`

#### Step 4: Run Simulation

1. **Start Behavioral Simulation**
   - Flow Navigator → SIMULATION → Run Simulation → Run Behavioral Simulation
   - Wait for compilation (1-2 minutes)

2. **Monitor Console Output**
   - TCL Console shows testbench messages
   - Look for "Test Summary" section
   - Verify VCD file generation message

3. **View Waveforms** (Optional)
   - Waveform window shows signal activity
   - Use zoom controls to inspect transitions
   - Add signals from Scope window

#### Step 5: Locate VCD File

```tcl
# In TCL Console
pwd  # Shows current directory
# VCD file location: <project>.sim/sim_1/behav/xsim/alu_simulation.vcd
```

**Copy VCD to Analysis Directory:**
```powershell
# Windows
copy <project>.sim\sim_1\behav\xsim\alu_simulation.vcd .

# Linux/Mac
cp <project>.sim/sim_1/behav/xsim/alu_simulation.vcd .
```

---

### Method 2: Command-Line Workflow (Advanced)

#### Create TCL Script

**File: `run_simulation.tcl`**
```tcl
# Hardware Trojan Detection - Automated Simulation Script

# Create project
create_project hardware_trojan_detection ./vivado_project -part xc7a35tcpg236-1 -force

# Add design sources
add_files {alu_clean.v alu_trojan.v}
set_property file_type Verilog [get_files alu_clean.v]
set_property file_type Verilog [get_files alu_trojan.v]

# Add simulation sources
add_files -fileset sim_1 alu_tb.v
set_property file_type Verilog [get_files alu_tb.v]
set_property top alu_tb [get_filesets sim_1]
set_property top_lib xil_defaultlib [get_filesets sim_1]

# Configure simulation
set_property -name {xsim.simulate.runtime} -value {1000us} -objects [get_filesets sim_1]
set_property -name {xsim.elaborate.debug_level} -value {all} -objects [get_filesets sim_1]

# Run simulation
launch_simulation
run 1000us

# Close project
close_project

puts "Simulation complete. VCD file generated."
```

#### Execute Script

```bash
# Run in batch mode
vivado -mode batch -source run_simulation.tcl

# Or interactive mode
vivado -mode tcl
source run_simulation.tcl
```

---

## Alternative Simulators

### Icarus Verilog (Free, Cross-Platform)

#### Installation
```bash
# Linux
sudo apt install iverilog gtkwave  # Ubuntu/Debian
sudo yum install iverilog gtkwave  # CentOS/RHEL

# macOS
brew install icarus-verilog gtkwave

# Windows (use WSL or download from bleyer.org/icarus)
```

#### Compile and Simulate
```bash
# Compile all files
iverilog -o alu_sim alu_clean.v alu_trojan.v alu_tb.v

# Run simulation
vvp alu_sim

# Output: alu_simulation.vcd generated
```

#### View Waveforms
```bash
gtkwave alu_simulation.vcd &
```

---

### ModelSim (Intel/Altera)

#### Compile and Simulate
```bash
# Create work library
vlib work
vmap work work

# Compile sources
vlog alu_clean.v alu_trojan.v alu_tb.v

# Simulate
vsim -c alu_tb -do "run -all; quit"

# VCD is generated automatically
```

---

## Analysis Execution

### Running the Python Analyzer

#### Basic Execution
```bash
# Ensure VCD file is in current directory
ls alu_simulation.vcd

# Run analyzer
python trojan_detector.py
```

#### Expected Output
```
================================================================================
HARDWARE TROJAN SIDE-CHANNEL DETECTOR
Advanced Toggle-Based Analysis for Security Verification
================================================================================

[*] Parsing VCD file: alu_simulation.vcd
[+] Found 47 signals
[+] Parsing complete. Total toggles detected: 8,542

[Phase 2] Trojan Detection Analysis
--------------------------------------------------------------------------------
[*] Analyzing toggle count deviations (threshold: 25.0%)
[+] Comparing 23 common signals

================================================================================
HARDWARE TROJAN DETECTION REPORT
================================================================================

Detection Threshold: 25.0% deviation
Suspicious Signals Found: 4

Signal Name                    Clean      Trojan     Deviation    Status
--------------------------------------------------------------------------------
trojan_shadow_reg[0]          0          156            inf%     ⚠ SUSPICIOUS
trojan_shadow_reg[1]          0          156            inf%     ⚠ SUSPICIOUS
trojan_activation_counter[0]  0          12             inf%     ⚠ SUSPICIOUS
trojan_trigger_active         0          24             inf%     ⚠ SUSPICIOUS

================================================================================

Total Toggle Activity:
  Clean Design:  4,271 toggles
  Trojan Design: 4,619 toggles
  Difference:    +348 toggles (+8.15%)

================================================================================

[Phase 3] Visualization Generation
--------------------------------------------------------------------------------
[*] Generating comparison visualization...
[+] Visualization saved: trojan_detection_analysis.png

[✓] Analysis Complete!
[✓] Suspicious signals detected: 4
[✓] Report and visualization generated successfully.
```

### Output Files

1. **Console Report**: Real-time analysis results
2. **trojan_detection_analysis.png**: Visual comparison
   - Top plot: Side-by-side toggle counts
   - Bottom plot: Deviation percentages with threshold line

---

## Advanced Usage

### Customizing Detection Threshold

**Edit `trojan_detector.py`:**
```python
# Line ~480
detection_threshold = 20.0  # Lower for higher sensitivity
detection_threshold = 30.0  # Higher for fewer false positives
```

### Batch Processing Multiple VCD Files

**Create `batch_analyze.py`:**
```python
import os
import subprocess

vcd_files = [f for f in os.listdir('.') if f.endswith('.vcd')]

for vcd in vcd_files:
    print(f"\n{'='*60}")
    print(f"Analyzing: {vcd}")
    print('='*60)
    
    # Run analyzer
    subprocess.run(['python', 'trojan_detector.py', vcd])
```

### Exporting Results to CSV

**Add to `trojan_detector.py`:**
```python
import csv

def export_results(results, filename='results.csv'):
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['signal', 'clean', 'trojan', 'deviation', 'suspicious'])
        writer.writeheader()
        writer.writerows(results)
    print(f"[+] Results exported to {filename}")

# In main() function, after results = detector.analyze():
export_results(results)
```

---

## Troubleshooting Guide

### Problem: Vivado compilation errors

#### Error: "Syntax error near..."
**Solution:**
- Check file encoding (must be UTF-8 without BOM)
- Verify no special characters in comments
- Ensure all modules have `endmodule`

```bash
# Check file encoding (Linux)
file alu_clean.v  # Should show "ASCII text"

# Convert if needed
iconv -f ISO-8859-1 -t UTF-8 alu_clean.v > alu_clean_fixed.v
```

#### Error: "Cannot find 'alu_clean'"
**Solution:**
- Verify file paths in project
- Check module names match filenames
- Re-add sources if needed

---

### Problem: No VCD file generated

#### Vivado:
```tcl
# Check simulation settings
get_property xsim.elaborate.debug_level [get_filesets sim_1]
# Should return "all"

# Force VCD generation
set_property -name {xsim.elaborate.debug_level} -value {all} -objects [get_filesets sim_1]
```

#### Icarus Verilog:
```verilog
// Ensure these lines are in testbench
initial begin
    $dumpfile("alu_simulation.vcd");
    $dumpvars(0, alu_tb);
end
```

---

### Problem: Python script crashes

#### Error: "No module named 'numpy'"
```bash
# Verify Python environment
python -m pip list | grep numpy

# Reinstall if missing
python -m pip install --upgrade numpy matplotlib
```

#### Error: "File not found: alu_simulation.vcd"
```bash
# Check current directory
ls -la  # Linux/Mac
dir     # Windows

# Copy VCD file to script directory
cp path/to/alu_simulation.vcd .
```

---

### Problem: No suspicious signals detected

**Possible Causes:**
1. Trojan never triggered
2. Threshold too high
3. VCD parsing issue

**Solutions:**

1. **Verify trigger patterns executed:**
   - Check testbench output for "Trojan Trigger Pattern Testing"
   - Ensure simulation ran to completion

2. **Lower detection threshold:**
   ```python
   detection_threshold = 15.0  # Try lower value
   ```

3. **Debug VCD parsing:**
   ```python
   # Add debug output in VCDParser.parse()
   print(f"Parsed signal: {signal_name} = {value}")
   ```

---

## Performance Optimization

### Reduce Simulation Time

**Method 1: Reduce Test Vectors**
```verilog
// In alu_tb.v, reduce random tests
repeat(50) begin  // Instead of 100
    // ...
end
```

**Method 2: Parallel Simulation**
```bash
# For multiple designs, run in parallel
iverilog -o sim1 design1.v tb.v &
iverilog -o sim2 design2.v tb.v &
wait
```

### Optimize VCD File Size

```verilog
// In testbench, dump only specific signals
initial begin
    $dumpfile("alu_simulation.vcd");
    $dumpvars(1, uut_clean);  // Only level 1
    $dumpvars(1, uut_trojan);
end
```

### Speed Up Analysis

**Use NumPy vectorization:**
```python
# Instead of loops, use numpy operations
import numpy as np

deviations = np.abs(trojan_counts - clean_counts) / clean_counts * 100
suspicious = deviations > threshold
```

---

## Best Practices

### 1. Version Control
```bash
git init
git add *.v *.py README.md
git commit -m "Initial hardware Trojan detection project"
```

### 2. Automated Testing
**Create `run_all.sh`:**
```bash
#!/bin/bash
set -e

echo "=== Compiling Design ==="
iverilog -o alu_sim alu_clean.v alu_trojan.v alu_tb.v

echo "=== Running Simulation ==="
vvp alu_sim

echo "=== Running Analysis ==="
python trojan_detector.py

echo "=== Complete ==="
```

### 3. Documentation
- Comment complex logic blocks
- Document Trojan trigger conditions
- Keep README updated with findings

---

## Next Steps

### Enhance the Project

1. **Add More Trojan Variants**
   - Sequential trigger (requires multiple patterns)
   - Time-based trigger (after N clock cycles)
   - Combinational trigger (complex Boolean conditions)

2. **Implement Machine Learning Detection**
   - Use scikit-learn for classification
   - Train on clean vs Trojan features
   - Automatic threshold optimization

3. **FPGA Implementation**
   - Synthesize for real hardware
   - Measure actual power consumption
   - Compare FPGA resource utilization

4. **Advanced Side-Channels**
   - Power analysis (simulate with SPICE)
   - Timing analysis (path delays)
   - Electromagnetic emission simulation

---

## Resources

### Documentation
- [Vivado Design Suite User Guide](https://www.xilinx.com/support/documentation/sw_manuals/xilinx2023_2/ug835-vivado-tcl-commands.pdf)
- [Icarus Verilog Manual](http://iverilog.icarus.com/man-pages.html)
- [Python VCD Parsing](https://pypi.org/project/pyvcd/)

### Tools
- **GTKWave**: Waveform viewer
- **Verilator**: Fast simulator
- **Yosys**: Open-source synthesis

### Academic Papers
- "Hardware Trojan Horses" by Bhunia et al.
- "Detecting Hardware Trojans using Backside Optical Imaging"
- "A Survey of Hardware Trojan Detection Techniques"

---

**End of Guide**

For questions or issues, review the troubleshooting section or check simulator documentation.
