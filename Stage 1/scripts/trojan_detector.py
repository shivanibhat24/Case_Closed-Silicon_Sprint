#!/usr/bin/env python3
"""
File: trojan_detector_gui.py
Description: Professional GUI for hardware Trojan detection
Author: Shivani Bhat 
Date: November 2025
"""

import re
import sys
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import threading

# GUI imports
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# Plotting imports
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np


class VCDParser:
    """Parse VCD files and extract switching activity data"""
    
    def __init__(self, vcd_file, log_callback=None):
        self.vcd_file = vcd_file
        self.signal_map = {}
        self.toggle_counts = defaultdict(int)
        self.signal_values = {}
        self.log_callback = log_callback
        
    def log(self, message):
        """Log message to callback if available"""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)
    
    def parse(self):
        """Parse VCD file and count toggles"""
        self.log(f"[*] Parsing VCD file: {Path(self.vcd_file).name}")
        
        try:
            with open(self.vcd_file, 'r', encoding='utf-8', errors='ignore') as f:
                in_header = True
                line_count = 0
                
                for line in f:
                    line = line.strip()
                    line_count += 1
                    
                    if in_header:
                        if line.startswith('$var'):
                            self._parse_var_declaration(line)
                        elif line.startswith('$enddefinitions'):
                            in_header = False
                            self.log(f"[+] Found {len(self.signal_map)} signals in header")
                    else:
                        self._parse_value_change(line)
            
            total_toggles = sum(self.toggle_counts.values())
            self.log(f"[+] Parsed {line_count} lines, detected {total_toggles} total toggles")
            return self.toggle_counts
            
        except FileNotFoundError:
            self.log(f"[!] ERROR: File not found: {self.vcd_file}")
            raise
        except Exception as e:
            self.log(f"[!] ERROR parsing VCD: {e}")
            raise
    
    def _parse_var_declaration(self, line):
        """Extract signal name and code from $var declaration"""
        try:
            parts = line.split()
            if len(parts) >= 5:
                signal_code = parts[3]
                signal_name = parts[4]
                self.signal_map[signal_code] = signal_name
                self.toggle_counts[signal_name] = 0
        except Exception as e:
            pass  # Skip malformed lines
    
    def _parse_value_change(self, line):
        """Detect value changes and count toggles"""
        if not line or line.startswith('$') or line.startswith('#'):
            return
        
        try:
            if line[0] in ['0', '1', 'x', 'z'] and len(line) > 1:
                value = line[0]
                code = line[1:]
                
                if code in self.signal_map:
                    signal_name = self.signal_map[code]
                    
                    if signal_name in self.signal_values:
                        if self.signal_values[signal_name] != value and value in ['0', '1']:
                            self.toggle_counts[signal_name] += 1
                    
                    self.signal_values[signal_name] = value
            
            elif line.startswith('b'):
                parts = line.split()
                if len(parts) >= 2:
                    value = parts[0][1:]
                    code = parts[1]
                    
                    if code in self.signal_map:
                        signal_name = self.signal_map[code]
                        
                        if signal_name in self.signal_values:
                            old_val = self.signal_values[signal_name]
                            if old_val != value:
                                toggles = self._count_bit_differences(old_val, value)
                                self.toggle_counts[signal_name] += toggles
                        
                        self.signal_values[signal_name] = value
        except Exception as e:
            pass  # Skip malformed lines
    
    def _count_bit_differences(self, val1, val2):
        """Count bit differences between two binary strings"""
        try:
            max_len = max(len(val1), len(val2))
            val1 = val1.zfill(max_len)
            val2 = val2.zfill(max_len)
            
            count = 0
            for b1, b2 in zip(val1, val2):
                if b1 != b2 and b1 in ['0', '1'] and b2 in ['0', '1']:
                    count += 1
            return count
        except:
            return 0


class TrojanDetector:
    """Analyze and compare toggle counts to detect hardware Trojans"""
    
    def __init__(self, clean_toggles, trojan_toggles, threshold=25.0, log_callback=None):
        self.clean_toggles = clean_toggles
        self.trojan_toggles = trojan_toggles
        self.threshold = threshold
        self.suspicious_signals = []
        self.log_callback = log_callback
        
    def log(self, message):
        """Log message to callback if available"""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)
    
    def analyze(self):
        """Compare toggle counts and identify suspicious signals"""
        self.log(f"\n[*] Analyzing deviations (threshold: {self.threshold:.1f}%)")
        
        common_signals = set(self.clean_toggles.keys()) & set(self.trojan_toggles.keys())
        self.log(f"[+] Comparing {len(common_signals)} common signals")
        
        results = []
        
        for signal in sorted(common_signals):
            clean_count = self.clean_toggles[signal]
            trojan_count = self.trojan_toggles[signal]
            
            if clean_count == 0:
                if trojan_count > 0:
                    deviation = 999.9  # Cap at 999.9% instead of infinity
                else:
                    deviation = 0.0
            else:
                deviation = abs(trojan_count - clean_count) / clean_count * 100
            
            suspicious = deviation > self.threshold
            
            results.append({
                'signal': signal,
                'clean': clean_count,
                'trojan': trojan_count,
                'deviation': deviation,
                'suspicious': suspicious
            })
            
            if suspicious:
                self.suspicious_signals.append(signal)
        
        self.log(f"[+] Found {len(self.suspicious_signals)} suspicious signals")
        return results


class ModernButton(tk.Canvas):
    """Custom modern button widget"""
    
    def __init__(self, parent, text="Button", command=None, bg="#4a9eff", fg="white", 
                 width=200, height=40, font=("Arial", 11, "bold")):
        super().__init__(parent, width=width, height=height, bg=parent.cget('bg'), 
                        highlightthickness=0, cursor="hand2")
        
        self.command = command
        self.bg = bg
        self.fg = fg
        self.text = text
        self.font = font
        self.width = width
        self.height = height
        self.enabled = True
        
        self.draw_button()
        self.bind("<Button-1>", self.on_click)
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
    
    def draw_button(self, hover=False):
        """Draw the button"""
        self.delete("all")
        
        if not self.enabled:
            color = "#666666"
        elif hover:
            color = self._brighten_color(self.bg)
        else:
            color = self.bg
        
        # Rounded rectangle
        self.create_rounded_rect(2, 2, self.width-2, self.height-2, 
                                radius=8, fill=color, outline="")
        
        # Text
        self.create_text(self.width//2, self.height//2, text=self.text, 
                        fill=self.fg, font=self.font)
    
    def create_rounded_rect(self, x1, y1, x2, y2, radius=25, **kwargs):
        """Create a rounded rectangle"""
        points = [x1+radius, y1,
                 x1+radius, y1,
                 x2-radius, y1,
                 x2-radius, y1,
                 x2, y1,
                 x2, y1+radius,
                 x2, y1+radius,
                 x2, y2-radius,
                 x2, y2-radius,
                 x2, y2,
                 x2-radius, y2,
                 x2-radius, y2,
                 x1+radius, y2,
                 x1+radius, y2,
                 x1, y2,
                 x1, y2-radius,
                 x1, y2-radius,
                 x1, y1+radius,
                 x1, y1+radius,
                 x1, y1]
        
        return self.create_polygon(points, **kwargs, smooth=True)
    
    def _brighten_color(self, hex_color):
        """Brighten a hex color"""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        rgb = tuple(min(255, int(c * 1.2)) for c in rgb)
        return '#%02x%02x%02x' % rgb
    
    def on_click(self, event):
        """Handle click event"""
        if self.enabled and self.command:
            self.command()
    
    def on_enter(self, event):
        """Handle mouse enter"""
        if self.enabled:
            self.draw_button(hover=True)
    
    def on_leave(self, event):
        """Handle mouse leave"""
        self.draw_button(hover=False)
    
    def configure_state(self, state):
        """Enable or disable button"""
        self.enabled = (state == "normal")
        self.draw_button()


class TrojanDetectorGUI:
    """Main GUI application for Trojan detection"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Hardware Trojan Detector - Advanced Side-Channel Analysis")
        self.root.geometry("1500x950")
        self.root.configure(bg='#1e1e1e')
        
        # Make window resizable
        self.root.minsize(1200, 700)
        
        # Data storage
        self.vcd_file = None
        self.results = None
        self.clean_toggles = {}
        self.trojan_toggles = {}
        
        # Colors
        self.colors = {
            'bg': '#1e1e1e',
            'fg': '#ffffff',
            'accent': '#4a9eff',
            'success': '#2ecc71',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'panel': '#2b2b2b',
            'button': '#3d3d3d',
            'border': '#4a4a4a'
        }
        
        # Create UI
        self.create_widgets()
        
        # Auto-load file if exists
        self.auto_load_file()
    
    def create_widgets(self):
        """Create all UI widgets"""
        
        # ============ HEADER ============
        header = tk.Frame(self.root, bg=self.colors['panel'], height=80)
        header.pack(fill=tk.X, padx=0, pady=0)
        header.pack_propagate(False)
        
        # Title
        title = tk.Label(header, text="🛡️ Hardware Trojan Detector", 
                        font=("Arial", 24, "bold"), 
                        bg=self.colors['panel'], fg=self.colors['accent'])
        title.pack(side=tk.LEFT, padx=30, pady=20)
        
        # Subtitle
        subtitle = tk.Label(header, text="Advanced Side-Channel Analysis via Toggle Counting", 
                           font=("Arial", 11), 
                           bg=self.colors['panel'], fg='#aaaaaa')
        subtitle.pack(side=tk.LEFT, padx=0, pady=20)
        
        # ============ MAIN CONTAINER ============
        main_container = tk.Frame(self.root, bg=self.colors['bg'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel (controls)
        left_panel = tk.Frame(main_container, bg=self.colors['panel'], width=380)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Right panel (visualization)
        right_panel = tk.Frame(main_container, bg=self.colors['panel'])
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # === LEFT PANEL CONTENTS ===
        
        # File selection section
        file_section = self.create_section(left_panel, "📁 VCD File Selection")
        
        self.file_status = tk.Label(file_section, text="No file loaded", 
                                   font=("Arial", 10), 
                                   bg=self.colors['panel'], fg='#888888',
                                   wraplength=340, justify=tk.LEFT)
        self.file_status.pack(pady=(0, 10), padx=10, anchor=tk.W)
        
        browse_btn = ModernButton(file_section, text="📂 Browse VCD File", 
                                 command=self.load_vcd_file,
                                 bg=self.colors['accent'], width=340, height=45)
        browse_btn.pack(pady=5, padx=10)
        
        # Settings section
        settings_section = self.create_section(left_panel, "⚙️ Analysis Settings")
        
        tk.Label(settings_section, text="Detection Threshold:", 
                font=("Arial", 10, "bold"),
                bg=self.colors['panel'], fg=self.colors['fg']).pack(anchor=tk.W, padx=10, pady=(5, 5))
        
        threshold_frame = tk.Frame(settings_section, bg=self.colors['panel'])
        threshold_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.threshold_var = tk.DoubleVar(value=25.0)
        
        threshold_slider = tk.Scale(threshold_frame, from_=5, to=100, 
                                   orient=tk.HORIZONTAL, 
                                   variable=self.threshold_var,
                                   bg=self.colors['panel'], fg=self.colors['fg'],
                                   highlightthickness=0, troughcolor=self.colors['button'],
                                   activebackground=self.colors['accent'],
                                   command=self.update_threshold_display)
        threshold_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.threshold_display = tk.Label(threshold_frame, text="25.0%", 
                                         font=("Arial", 11, "bold"),
                                         bg=self.colors['panel'], fg=self.colors['accent'],
                                         width=8)
        self.threshold_display.pack(side=tk.LEFT, padx=10)
        
        # Run analysis button
        self.analyze_btn = ModernButton(settings_section, text="🔍 Run Analysis", 
                                       command=self.run_analysis,
                                       bg=self.colors['success'], width=340, height=50)
        self.analyze_btn.pack(pady=15, padx=10)
        self.analyze_btn.configure_state("disabled")
        
        # Results section
        results_section = self.create_section(left_panel, "📊 Detection Results")
        
        self.status_display = tk.Label(results_section, 
                                      text="⏳ Waiting for analysis...",
                                      font=("Arial", 12, "bold"),
                                      bg=self.colors['panel'], fg='#888888',
                                      wraplength=340, pady=10)
        self.status_display.pack(pady=10, padx=10)
        
        # Stats display
        stats_frame = tk.Frame(results_section, bg=self.colors['button'], 
                              highlightbackground=self.colors['border'], highlightthickness=1)
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=10, wrap=tk.WORD, 
                                 font=("Consolas", 9),
                                 bg=self.colors['button'], fg=self.colors['fg'],
                                 relief=tk.FLAT, padx=10, pady=10)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        self.stats_text.insert(1.0, "No analysis data yet.\n\nLoad a VCD file and run analysis to see results.")
        self.stats_text.config(state=tk.DISABLED)
        
        # Log section
        log_section = self.create_section(left_panel, "📝 Analysis Log")
        
        log_frame = tk.Frame(log_section, bg=self.colors['button'],
                            highlightbackground=self.colors['border'], highlightthickness=1)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=12, wrap=tk.WORD,
                                                  font=("Consolas", 9),
                                                  bg=self.colors['button'], fg='#00ff00',
                                                  relief=tk.FLAT, padx=10, pady=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Export buttons
        export_frame = tk.Frame(left_panel, bg=self.colors['panel'])
        export_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ModernButton(export_frame, text="💾 Export Report", 
                    command=self.export_report,
                    bg='#9b59b6', width=165, height=40).pack(side=tk.LEFT, padx=(0, 5))
        
        ModernButton(export_frame, text="📊 Export Chart", 
                    command=self.export_chart,
                    bg='#e67e22', width=165, height=40).pack(side=tk.RIGHT, padx=(5, 0))
        
        # === RIGHT PANEL CONTENTS ===
        
        # Notebook for tabs
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background=self.colors['panel'], borderwidth=0)
        style.configure('TNotebook.Tab', background=self.colors['button'], 
                       foreground=self.colors['fg'], padding=[20, 10],
                       font=("Arial", 10, "bold"))
        style.map('TNotebook.Tab', background=[('selected', self.colors['accent'])],
                 foreground=[('selected', 'white')])
        
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.create_chart_tabs()
        
        self.log("[*] Hardware Trojan Detector initialized")
        self.log("[*] Ready to load VCD file...")
    
    def create_section(self, parent, title):
        """Create a styled section frame"""
        section = tk.Frame(parent, bg=self.colors['panel'])
        section.pack(fill=tk.X, padx=10, pady=10)
        
        header = tk.Label(section, text=title, font=("Arial", 12, "bold"),
                         bg=self.colors['panel'], fg=self.colors['fg'])
        header.pack(anchor=tk.W, pady=(0, 10))
        
        return section
    
    def create_chart_tabs(self):
        """Create visualization tabs"""
        
        # Tab 1: Toggle Comparison
        self.tab1 = tk.Frame(self.notebook, bg=self.colors['button'])
        self.notebook.add(self.tab1, text="  📊 Toggle Comparison  ")
        
        # Tab 2: Deviation Analysis
        self.tab2 = tk.Frame(self.notebook, bg=self.colors['button'])
        self.notebook.add(self.tab2, text="  📈 Deviation Analysis  ")
        
        # Tab 3: Signal Table
        self.tab3 = tk.Frame(self.notebook, bg=self.colors['button'])
        self.notebook.add(self.tab3, text="  📋 Signal Details  ")
        
        # Create initial empty charts
        self.create_empty_chart(self.tab1, "Load VCD file and run analysis\nto view toggle comparison")
        self.create_empty_chart(self.tab2, "Load VCD file and run analysis\nto view deviation analysis")
        
        # Create table in tab 3
        self.create_signal_table()
    
    def create_empty_chart(self, parent, message):
        """Create empty placeholder chart"""
        fig = Figure(figsize=(10, 6), facecolor=self.colors['button'])
        ax = fig.add_subplot(111, facecolor=self.colors['button'])
        ax.text(0.5, 0.5, message, ha='center', va='center', 
               fontsize=16, color=self.colors['fg'], weight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        return canvas
    
    def create_signal_table(self):
        """Create signal details table"""
        # Style
        style = ttk.Style()
        style.configure("Treeview", background=self.colors['button'],
                       foreground=self.colors['fg'], fieldbackground=self.colors['button'],
                       font=("Arial", 10))
        style.configure("Treeview.Heading", background=self.colors['panel'],
                       foreground=self.colors['fg'], font=("Arial", 10, "bold"))
        style.map('Treeview', background=[('selected', self.colors['accent'])])
        
        # Container
        table_container = tk.Frame(self.tab3, bg=self.colors['button'])
        table_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Scrollbars
        vsb = ttk.Scrollbar(table_container, orient="vertical")
        hsb = ttk.Scrollbar(table_container, orient="horizontal")
        
        # Treeview
        self.tree = ttk.Treeview(table_container,
                                 columns=("rank", "signal", "clean", "trojan", "deviation", "status"),
                                 show="headings",
                                 yscrollcommand=vsb.set,
                                 xscrollcommand=hsb.set)
        
        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)
        
        # Headings
        self.tree.heading("rank", text="#")
        self.tree.heading("signal", text="Signal Name")
        self.tree.heading("clean", text="Clean Toggles")
        self.tree.heading("trojan", text="Trojan Toggles")
        self.tree.heading("deviation", text="Deviation %")
        self.tree.heading("status", text="Status")
        
        # Column widths
        self.tree.column("rank", width=50, anchor=tk.CENTER)
        self.tree.column("signal", width=250)
        self.tree.column("clean", width=120, anchor=tk.CENTER)
        self.tree.column("trojan", width=120, anchor=tk.CENTER)
        self.tree.column("deviation", width=120, anchor=tk.CENTER)
        self.tree.column("status", width=150, anchor=tk.CENTER)
        
        # Grid
        self.tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        
        table_container.grid_rowconfigure(0, weight=1)
        table_container.grid_columnconfigure(0, weight=1)
    
    def auto_load_file(self):
        """Automatically load VCD file if it exists"""
        default_path = r"C:\Users\sg78b\Case_CLosed-Silicon_Sprint\Case_CLosed-Silicon_Sprint.sim\sim_1\behav\xsim\alu_simulation.vcd"
        
        if Path(default_path).exists():
            self.vcd_file = default_path
            filename = Path(default_path).name
            self.file_status.config(text=f"✅ {filename}", fg=self.colors['success'])
            self.analyze_btn.configure_state("normal")
            self.log(f"[+] Auto-loaded: {filename}")
            self.log("[*] Click 'Run Analysis' to begin detection")
        else:
            self.log("[*] Default VCD file not found. Please browse to select file.")
    
    def load_vcd_file(self):
        """Browse and load VCD file"""
        filename = filedialog.askopenfilename(
            title="Select VCD File",
            filetypes=[("VCD Files", "*.vcd"), ("All Files", "*.*")]
        )
        
        if filename:
            self.vcd_file = filename
            file_name = Path(filename).name
            self.file_status.config(text=f"✅ {file_name}", fg=self.colors['success'])
            self.analyze_btn.configure_state("normal")
            self.log(f"[+] Loaded: {file_name}")
    
    def update_threshold_display(self, value):
        """Update threshold display label"""
        self.threshold_display.config(text=f"{float(value):.1f}%")
    
    def run_analysis(self):
        """Run Trojan detection analysis"""
        if not self.vcd_file:
            messagebox.showwarning("No File", "Please select a VCD file first.")
            return
        
        self.analyze_btn.configure_state("disabled")
        self.status_display.config(text="⏳ Analysis in progress...", fg='#f39c12')
        
        self.log("\n" + "="*50)
        self.log("Starting Hardware Trojan Detection...")
        self.log("="*50)
        
        # Run in thread
        thread = threading.Thread(target=self._run_analysis_thread, daemon=True)
        thread.start()
    
    def _run_analysis_thread(self):
        """Analysis worker thread"""
        try:
            # Parse VCD
            parser = VCDParser(self.vcd_file, log_callback=self.log)
            all_toggles = parser.parse()
            
            if not all_toggles:
                raise Exception("No signals found in VCD file")
            
            # Separate signals
            self.clean_toggles, self.trojan_toggles = self.separate_signals(all_toggles)
            
            if not self.clean_toggles or not self.trojan_toggles:
                raise Exception("Could not separate clean and trojan signals. Check VCD signal naming.")
            
            # Run detection
            threshold = self.threshold_var.get()
            detector = TrojanDetector(self.clean_toggles, self.trojan_toggles, 
                                    threshold, log_callback=self.log)
            
            self.results = detector.analyze()
            
            # Update GUI
            self.root.after(0, self.display_results)
            
        except Exception as e:
            self.log(f"\n[!] ERROR: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("Analysis Error", 
                                                            f"An error occurred:\n\n{str(e)}"))
            self.root.after(0, lambda: self.status_display.config(
                text="❌ Analysis failed", fg=self.colors['danger']))
        finally:
            self.root.after(0, lambda: self.analyze_btn.configure_state("normal"))
    
    def separate_signals(self, all_toggles):
        """Separate signals into clean and trojan categories"""
        clean = {}
        trojan = {}
        
        self.log("\n[*] Separating clean and trojan signals...")
        
        # Try multiple naming patterns
        for sig, count in all_toggles.items():
            sig_lower = sig.lower()
            
            # Pattern 1: Direct module instance names
            if 'uut_clean' in sig_lower:
                name = sig.split('.')[-1] if '.' in sig else sig
                clean[name] = count
            elif 'uut_trojan' in sig_lower:
                name = sig.split('.')[-1] if '.' in sig else sig
                trojan[name] = count
            # Pattern 2: Contains clean/trojan keywords
            elif 'clean' in sig_lower and 'trojan' not in sig_lower:
                name = sig.replace('clean', '').replace('_', '').split('.')[-1]
                if name:
                    clean[name] = count
            elif 'trojan' in sig_lower:
                name = sig.replace('trojan', '').replace('_', '').split('.')[-1]
                if name:
                    trojan[name] = count
        
        # Fallback: group by basename
        if len(clean) == 0 and len(trojan) == 0:
            self.log("[*] Using fallback: grouping by signal basename")
            signal_groups = defaultdict(list)
            for sig, count in all_toggles.items():
                basename = sig.split('.')[-1]
                signal_groups[basename].append((sig, count))
            
            for basename, instances in signal_groups.items():
                if len(instances) >= 2:
                    clean[basename] = instances[0][1]
                    trojan[basename] = instances[1][1]
        
        self.log(f"[+] Found {len(clean)} clean signals, {len(trojan)} trojan signals")
        return clean, trojan
    
    def display_results(self):
        """Display analysis results"""
        if not self.results:
            return
        
        suspicious = [r for r in self.results if r['suspicious']]
        
        # Update status
        if len(suspicious) == 0:
            status_text = "✅ NO TROJAN DETECTED"
            status_color = self.colors['success']
        elif len(suspicious) <= 3:
            status_text = "⚠️ SUSPICIOUS ACTIVITY"
            status_color = self.colors['warning']
        else:
            status_text = "🚨 TROJAN CONFIRMED"
            status_color = self.colors['danger']
        
        self.status_display.config(text=status_text, fg=status_color)
        
        # Update stats
        total_clean = sum(r['clean'] for r in self.results)
        total_trojan = sum(r['trojan'] for r in self.results)
        diff = total_trojan - total_clean
        
        if total_clean > 0:
            percent_diff = (diff / total_clean) * 100
        else:
            percent_diff = 0
        
        stats = f"""
╔══════════════════════════════════════╗
║      DETECTION STATISTICS            ║
╚══════════════════════════════════════╝

Signals Analyzed:      {len(self.results)}
Suspicious Signals:    {len(suspicious)}
Detection Threshold:   {self.threshold_var.get():.1f}%

╔══════════════════════════════════════╗
║      TOGGLE ACTIVITY SUMMARY         ║
╚══════════════════════════════════════╝

Clean Design:          {total_clean:,} toggles
Trojan Design:         {total_trojan:,} toggles
Difference:            {diff:+,} toggles
Relative Change:       {percent_diff:+.2f}%

╔══════════════════════════════════════╗
║      THREAT ASSESSMENT               ║
╚══════════════════════════════════════╝

Status: {status_text}
"""
        
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats)
        self.stats_text.config(state=tk.DISABLED)
        
        # Update visualizations
        self.update_charts()
        self.update_table()
        
        self.log(f"\n[✓] Analysis Complete!")
        self.log(f"[✓] Found {len(suspicious)} suspicious signals")
    
    def update_charts(self):
        """Update all charts"""
        if not self.results:
            return
        
        # Get significant results
        significant = [r for r in self.results if r['clean'] > 5 or r['trojan'] > 5]
        significant.sort(key=lambda x: x['deviation'], reverse=True)
        plot_data = significant[:25]  # Top 25
        
        if not plot_data:
            self.log("[!] No significant signals to plot")
            return
        
        # Extract data
        signals = [r['signal'][:20] for r in plot_data]
        clean_counts = [r['clean'] for r in plot_data]
        trojan_counts = [r['trojan'] for r in plot_data]
        deviations = [min(r['deviation'], 200) for r in plot_data]  # Cap at 200%
        suspicious = [r['suspicious'] for r in plot_data]
        
        # Clear old charts
        for widget in self.tab1.winfo_children():
            widget.destroy()
        for widget in self.tab2.winfo_children():
            widget.destroy()
        
        # Chart 1: Toggle Comparison
        self.create_toggle_chart(self.tab1, signals, clean_counts, trojan_counts, suspicious)
        
        # Chart 2: Deviation Analysis
        self.create_deviation_chart(self.tab2, signals, deviations, suspicious)
    
    def create_toggle_chart(self, parent, signals, clean, trojan, suspicious):
        """Create toggle comparison chart"""
        fig = Figure(figsize=(12, 7), facecolor=self.colors['button'])
        ax = fig.add_subplot(111, facecolor=self.colors['button'])
        
        x = np.arange(len(signals))
        width = 0.35
        
        # Bars
        bars1 = ax.bar(x - width/2, clean, width, label='Clean Design',
                      color='#2ecc71', alpha=0.9, edgecolor='white', linewidth=0.8)
        bars2 = ax.bar(x + width/2, trojan, width, label='Trojan Design',
                      color='#e74c3c', alpha=0.9, edgecolor='white', linewidth=0.8)
        
        # Highlight suspicious
        for i, susp in enumerate(suspicious):
            if susp:
                ax.axvspan(i - 0.5, i + 0.5, alpha=0.15, color='yellow', zorder=0)
        
        # Labels and styling
        ax.set_xlabel('Signal Name', color=self.colors['fg'], fontsize=12, fontweight='bold')
        ax.set_ylabel('Toggle Count', color=self.colors['fg'], fontsize=12, fontweight='bold')
        ax.set_title('Toggle Activity Comparison: Clean vs Trojan Design',
                    color=self.colors['fg'], fontsize=14, fontweight='bold', pad=15)
        
        ax.set_xticks(x)
        ax.set_xticklabels(signals, rotation=45, ha='right', fontsize=9, color=self.colors['fg'])
        ax.tick_params(colors=self.colors['fg'])
        
        ax.legend(facecolor=self.colors['panel'], edgecolor=self.colors['fg'],
                 labelcolor=self.colors['fg'], fontsize=11)
        ax.grid(True, alpha=0.2, color='white', linestyle='--', linewidth=0.5)
        
        for spine in ax.spines.values():
            spine.set_color(self.colors['fg'])
        
        fig.tight_layout()
        
        # Add to GUI
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        
        return canvas
    
    def create_deviation_chart(self, parent, signals, deviations, suspicious):
        """Create deviation analysis chart"""
        fig = Figure(figsize=(12, 7), facecolor=self.colors['button'])
        ax = fig.add_subplot(111, facecolor=self.colors['button'])
        
        x = np.arange(len(signals))
        
        # Color bars based on suspicion
        colors = [self.colors['danger'] if s else '#95a5a6' for s in suspicious]
        
        bars = ax.bar(x, deviations, color=colors, alpha=0.9, 
                     edgecolor='white', linewidth=0.8)
        
        # Threshold line
        threshold = self.threshold_var.get()
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2.5,
                  label=f'Detection Threshold ({threshold:.0f}%)', zorder=10)
        
        # Labels and styling
        ax.set_xlabel('Signal Name', color=self.colors['fg'], fontsize=12, fontweight='bold')
        ax.set_ylabel('Deviation (%)', color=self.colors['fg'], fontsize=12, fontweight='bold')
        ax.set_title('Toggle Count Deviation Analysis - Trojan Detection Metric',
                    color=self.colors['fg'], fontsize=14, fontweight='bold', pad=15)
        
        ax.set_xticks(x)
        ax.set_xticklabels(signals, rotation=45, ha='right', fontsize=9, color=self.colors['fg'])
        ax.tick_params(colors=self.colors['fg'])
        
        ax.legend(facecolor=self.colors['panel'], edgecolor=self.colors['fg'],
                 labelcolor=self.colors['fg'], fontsize=11)
        ax.grid(True, alpha=0.2, color='white', linestyle='--', linewidth=0.5)
        
        for spine in ax.spines.values():
            spine.set_color(self.colors['fg'])
        
        fig.tight_layout()
        
        # Add to GUI
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        
        return canvas
    
    def update_table(self):
        """Update signal details table"""
        # Clear existing
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Sort by deviation
        sorted_results = sorted(self.results, key=lambda x: x['deviation'], reverse=True)
        
        # Add data
        for rank, result in enumerate(sorted_results, 1):
            dev_str = f"{result['deviation']:.1f}%" if result['deviation'] < 999 else "999.9%"
            status = "⚠️ SUSPICIOUS" if result['suspicious'] else "✓ Normal"
            
            # Color coding
            tag = 'suspicious' if result['suspicious'] else 'normal'
            
            self.tree.insert("", tk.END, values=(
                rank,
                result['signal'],
                result['clean'],
                result['trojan'],
                dev_str,
                status
            ), tags=(tag,))
        
        # Configure tags
        self.tree.tag_configure('suspicious', background='#3d1f1f', foreground='#ff6b6b')
        self.tree.tag_configure('normal', background=self.colors['button'], foreground=self.colors['fg'])
    
    def export_report(self):
        """Export detailed text report"""
        if not self.results:
            messagebox.showwarning("No Data", "Please run analysis first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            initialfile=f"trojan_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("="*80 + "\n")
                    f.write("HARDWARE TROJAN DETECTION REPORT\n")
                    f.write("Advanced Side-Channel Analysis via Toggle Counting\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("Author: Shivani Bhat\n")
                    f.write("="*80 + "\n\n")
                    
                    # Summary
                    suspicious = [r for r in self.results if r['suspicious']]
                    total_clean = sum(r['clean'] for r in self.results)
                    total_trojan = sum(r['trojan'] for r in self.results)
                    
                    f.write("EXECUTIVE SUMMARY\n")
                    f.write("-"*80 + "\n")
                    f.write(f"VCD File: {Path(self.vcd_file).name}\n")
                    f.write(f"Detection Threshold: {self.threshold_var.get():.1f}%\n")
                    f.write(f"Total Signals Analyzed: {len(self.results)}\n")
                    f.write(f"Suspicious Signals Found: {len(suspicious)}\n\n")
                    
                    f.write(f"Total Toggle Activity:\n")
                    f.write(f"  Clean Design:  {total_clean:,} toggles\n")
                    f.write(f"  Trojan Design: {total_trojan:,} toggles\n")
                    f.write(f"  Difference:    {total_trojan - total_clean:+,} toggles\n\n")
                    
                    # Threat level
                    if len(suspicious) == 0:
                        threat = "LOW - No anomalies detected"
                    elif len(suspicious) <= 3:
                        threat = "MEDIUM - Possible Trojan activity"
                    else:
                        threat = "HIGH - Trojan confirmed"
                    
                    f.write(f"THREAT LEVEL: {threat}\n\n")
                    
                    # Suspicious signals
                    if suspicious:
                        f.write("="*80 + "\n")
                        f.write("SUSPICIOUS SIGNALS DETAILS\n")
                        f.write("="*80 + "\n\n")
                        
                        f.write(f"{'Rank':<6}{'Signal':<35}{'Clean':<10}{'Trojan':<10}{'Deviation':<12}\n")
                        f.write("-"*80 + "\n")
                        
                        for i, sig in enumerate(suspicious, 1):
                            dev = f"{sig['deviation']:.1f}%" if sig['deviation'] < 999 else "999.9%"
                            f.write(f"{i:<6}{sig['signal']:<35}{sig['clean']:<10}{sig['trojan']:<10}{dev:<12}\n")
                        
                        f.write("\n")
                    
                    # All signals
                    f.write("="*80 + "\n")
                    f.write("COMPLETE SIGNAL ANALYSIS (Top 50 by Deviation)\n")
                    f.write("="*80 + "\n\n")
                    
                    sorted_results = sorted(self.results, key=lambda x: x['deviation'], reverse=True)[:50]
                    
                    f.write(f"{'Signal':<35}{'Clean':<10}{'Trojan':<10}{'Deviation':<12}{'Status':<12}\n")
                    f.write("-"*80 + "\n")
                    
                    for sig in sorted_results:
                        dev = f"{sig['deviation']:.1f}%" if sig['deviation'] < 999 else "999.9%"
                        status = "SUSPICIOUS" if sig['suspicious'] else "Normal"
                        f.write(f"{sig['signal']:<35}{sig['clean']:<10}{sig['trojan']:<10}{dev:<12}{status:<12}\n")
                    
                    f.write("\n" + "="*80 + "\n")
                    f.write("END OF REPORT\n")
                    f.write("="*80 + "\n")
                
                self.log(f"[+] Report exported: {Path(filename).name}")
                messagebox.showinfo("Export Successful", f"Report saved to:\n{filename}")
                
            except Exception as e:
                messagebox.showerror("Export Failed", f"Failed to export report:\n{str(e)}")
    
    def export_chart(self):
        """Export current chart as image"""
        if not self.results:
            messagebox.showwarning("No Data", "Please run analysis first.")
            return
        
        # Get current tab
        current_tab = self.notebook.index(self.notebook.select())
        
        if current_tab >= 2:
            messagebox.showinfo("Info", "Cannot export table view. Switch to a chart tab.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("PDF Document", "*.pdf"), ("All Files", "*.*")],
            initialfile=f"trojan_detection_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        
        if filename:
            try:
                # Get the canvas from current tab
                if current_tab == 0:
                    tab = self.tab1
                else:
                    tab = self.tab2
                
                # Find the canvas widget
                for widget in tab.winfo_children():
                    if isinstance(widget, tk.Canvas):
                        # This is the FigureCanvasTkAgg widget
                        canvas = widget.master
                        if hasattr(canvas, 'figure'):
                            canvas.figure.savefig(filename, dpi=300, bbox_inches='tight',
                                                facecolor=self.colors['button'])
                            self.log(f"[+] Chart exported: {Path(filename).name}")
                            messagebox.showinfo("Export Successful", f"Chart saved to:\n{filename}")
                            return
                
                messagebox.showerror("Export Failed", "Could not find chart to export.")
                
            except Exception as e:
                messagebox.showerror("Export Failed", f"Failed to export chart:\n{str(e)}")
    
    def log(self, message):
        """Add message to log console"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.update_idletasks()


def main():
    """Main application entry point"""
    root = tk.Tk()
    
    # Set icon (if available)
    try:
        root.iconbitmap('icon.ico')
    except:
        pass
    
    app = TrojanDetectorGUI(root)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()


if __name__ == "__main__":
    main()
