################################################################################
# File: run_all.tcl
# Description: Automated Vivado simulation script for Hardware Trojan Detection
# Author: Shivani Bhat
# Date: November 2025
#
# Usage: vivado -mode batch -source run_all.tcl
################################################################################

puts "\n================================================================================"
puts "HARDWARE TROJAN DETECTION - AUTOMATED SIMULATION"
puts "================================================================================"

# Configuration
set project_name "hardware_trojan_detection"
set project_dir "./vivado_project"
set part_name "xc7a35tcpg236-1"

# Source files
set design_files [list "alu_clean.v" "alu_trojan.v"]
set sim_files [list "alu_tb.v"]

################################################################################
# Step 1: Create Project
################################################################################
puts "\n[Step 1/5] Creating Vivado project..."

# Remove existing project if present
if {[file exists $project_dir]} {
    puts "  - Removing existing project directory"
    file delete -force $project_dir
}

# Create new project
create_project $project_name $project_dir -part $part_name -force
puts "  [OK] Project created: $project_name"

################################################################################
# Step 2: Add Design Sources
################################################################################
puts "\n[Step 2/5] Adding design sources..."

foreach file $design_files {
    if {[file exists $file]} {
        add_files -norecurse $file
        puts "  [OK] Added: $file"
    } else {
        puts "  [ERROR] File not found: $file"
        exit 1
    }
}

set_property file_type Verilog [get_files *.v]
update_compile_order -fileset sources_1

################################################################################
# Step 3: Add Simulation Sources
################################################################################
puts "\n[Step 3/5] Adding simulation sources..."

foreach file $sim_files {
    if {[file exists $file]} {
        add_files -fileset sim_1 -norecurse $file
        puts "  [OK] Added: $file"
    } else {
        puts "  [ERROR] File not found: $file"
        exit 1
    }
}

set_property file_type Verilog [get_files -of_objects [get_filesets sim_1] *.v]
set_property top alu_tb [get_filesets sim_1]
set_property top_lib xil_defaultlib [get_filesets sim_1]
update_compile_order -fileset sim_1

################################################################################
# Step 4: Configure Simulation
################################################################################
puts "\n[Step 4/5] Configuring simulation settings..."

# Set simulation runtime
set_property -name {xsim.simulate.runtime} -value {1ms} -objects [get_filesets sim_1]
puts "  [OK] Simulation runtime: 1ms"

# Enable debug for VCD generation
set_property -name {xsim.elaborate.debug_level} -value {all} -objects [get_filesets sim_1]
puts "  [OK] Debug level: all (enables VCD)"

# Enable log for all signals
set_property -name {xsim.simulate.log_all_signals} -value {true} -objects [get_filesets sim_1]
puts "  [OK] Log all signals: enabled"

################################################################################
# Step 5: Run Simulation
################################################################################
puts "\n[Step 5/5] Running behavioral simulation..."
puts "  (This may take 1-2 minutes...)\n"

# Launch simulation
set_property target_simulator XSim [current_project]
launch_simulation -mode behavioral

# Run simulation
run 1ms

puts "\n  [OK] Simulation completed successfully"

################################################################################
# Post-Simulation: Copy VCD File
################################################################################
puts "\n[Post-Processing] Locating VCD file..."

# Find VCD file in simulation directory
set sim_dir "$project_dir/$project_name.sim/sim_1/behav/xsim"
set vcd_file "$sim_dir/alu_simulation.vcd"

if {[file exists $vcd_file]} {
    # Copy to working directory for analysis
    file copy -force $vcd_file "./alu_simulation.vcd"
    puts "  [OK] VCD file copied to: ./alu_simulation.vcd"
    
    # Get file size
    set filesize [file size $vcd_file]
    set filesize_kb [expr {$filesize / 1024}]
    puts "  [INFO] VCD file size: ${filesize_kb} KB"
} else {
    puts "  [WARNING] VCD file not found at: $vcd_file"
    puts "  [INFO] Check simulation output for errors"
}

################################################################################
# Generate Report
################################################################################
puts "\n[Generating Report] Simulation summary..."

# Get simulation statistics (if available)
set log_file "$sim_dir/simulate.log"
if {[file exists $log_file]} {
    puts "  [INFO] Simulation log: $log_file"
}

################################################################################
# Cleanup and Exit
################################################################################
puts "\n================================================================================"
puts "SIMULATION COMPLETE"
puts "================================================================================"
puts ""
puts "Next Steps:"
puts "  1. Review VCD file: ./alu_simulation.vcd"
puts "  2. Run analysis: python trojan_detector.py"
puts "  3. Check results: trojan_detection_analysis.png"
puts ""
puts "================================================================================"

# Close simulation
close_sim -force

# Optionally close project (comment out to keep open)
# close_project

puts "\nScript execution complete. Exiting.\n"
exit 0
