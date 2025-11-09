"""
File: integrated_trojan_detector.py
Description: Integrated ML-based Hardware Trojan Detection System with Professional GUI
Author: Shivani Bhat (Enhanced & Integrated)
Date: November 2025

Features:
- Professional GUI with dark theme
- Multiple ML classifiers (Random Forest, SVM, Neural Network, Gradient Boosting)
- Real-time analysis with progress indicators
- Advanced feature extraction from VCD files
- Comprehensive visualization dashboard
- Model training and testing capabilities
- Export functionality for reports and charts
"""

import re
import sys
import argparse
from collections import defaultdict
from pathlib import Path
import pickle
import json
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
import seaborn as sns

# Scientific computing
import numpy as np
import pandas as pd

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_curve, auc, precision_recall_curve,
                            accuracy_score, f1_score)
from sklearn.decomposition import PCA


# ============================================================================
# VCD PARSER & FEATURE EXTRACTION
# ============================================================================

class VCDParser:
    """Enhanced VCD parser with feature extraction and ML support"""
    
    def __init__(self, vcd_file, log_callback=None):
        self.vcd_file = vcd_file
        self.signal_map = {}
        self.toggle_counts = defaultdict(int)
        self.signal_values = {}
        self.timestamps = []
        self.value_changes = defaultdict(list)
        self.log_callback = log_callback
        
    def log(self, message):
        """Log message to callback if available"""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)
    
    def parse(self):
        """Parse VCD and extract features"""
        self.log(f"[*] Parsing {Path(self.vcd_file).name}")
        
        try:
            with open(self.vcd_file, 'r', encoding='utf-8', errors='ignore') as f:
                in_header = True
                current_time = 0
                line_count = 0
                
                for line in f:
                    line = line.strip()
                    line_count += 1
                    
                    if in_header:
                        if line.startswith('$var'):
                            self._parse_var_declaration(line)
                        elif line.startswith('$enddefinitions'):
                            in_header = False
                            self.log(f"[+] Found {len(self.signal_map)} signals")
                    else:
                        if line.startswith('#'):
                            current_time = int(line[1:])
                            self.timestamps.append(current_time)
                        else:
                            self._parse_value_change(line, current_time)
            
            total_toggles = sum(self.toggle_counts.values())
            self.log(f"[+] Total toggles: {total_toggles}")
            return self.toggle_counts
            
        except Exception as e:
            self.log(f"[!] ERROR: {e}")
            raise
    
    def _parse_var_declaration(self, line):
        parts = line.split()
        if len(parts) >= 5:
            signal_code = parts[3]
            signal_name = parts[4]
            self.signal_map[signal_code] = signal_name
            self.toggle_counts[signal_name] = 0
    
    def _parse_value_change(self, line, timestamp):
        if not line or line.startswith('$'):
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
                            self.value_changes[signal_name].append(timestamp)
                    
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
        except:
            pass
    
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
    
    def extract_advanced_features(self):
        """Extract advanced temporal features for ML"""
        features = {}
        
        for signal, count in self.toggle_counts.items():
            features[f"{signal}_toggle_count"] = count
            
            if signal in self.value_changes and len(self.value_changes[signal]) > 0:
                changes = self.value_changes[signal]
                features[f"{signal}_avg_interval"] = np.mean(np.diff(changes)) if len(changes) > 1 else 0
                features[f"{signal}_variance"] = np.var(np.diff(changes)) if len(changes) > 1 else 0
                features[f"{signal}_first_toggle"] = changes[0]
                features[f"{signal}_last_toggle"] = changes[-1]
            else:
                features[f"{signal}_avg_interval"] = 0
                features[f"{signal}_variance"] = 0
                features[f"{signal}_first_toggle"] = 0
                features[f"{signal}_last_toggle"] = 0
        
        return features


# ============================================================================
# FEATURE EXTRACTION & ML DETECTOR
# ============================================================================

class FeatureExtractor:
    """Extract ML features from signal data"""
    
    @staticmethod
    def extract_statistical_features(clean_toggles, trojan_toggles):
        """Extract statistical features comparing clean vs trojan"""
        features = {}
        
        common_signals = set(clean_toggles.keys()) & set(trojan_toggles.keys())
        
        if len(common_signals) == 0:
            return features
        
        total_clean = sum(clean_toggles.values())
        total_trojan = sum(trojan_toggles.values())
        
        features['total_toggle_ratio'] = total_trojan / (total_clean + 1)
        features['total_toggle_diff'] = total_trojan - total_clean
        features['avg_clean_toggles'] = np.mean(list(clean_toggles.values()))
        features['avg_trojan_toggles'] = np.mean(list(trojan_toggles.values()))
        features['std_clean_toggles'] = np.std(list(clean_toggles.values()))
        features['std_trojan_toggles'] = np.std(list(trojan_toggles.values()))
        
        deviations = []
        ratios = []
        abs_diffs = []
        
        for signal in common_signals:
            clean = clean_toggles.get(signal, 0)
            trojan = trojan_toggles.get(signal, 0)
            
            if clean > 0:
                deviation = abs(trojan - clean) / clean * 100
                ratio = trojan / clean
            else:
                deviation = 999.9 if trojan > 0 else 0
                ratio = trojan
            
            deviations.append(deviation)
            ratios.append(ratio)
            abs_diffs.append(abs(trojan - clean))
        
        features['max_deviation'] = np.max(deviations) if deviations else 0
        features['mean_deviation'] = np.mean(deviations) if deviations else 0
        features['median_deviation'] = np.median(deviations) if deviations else 0
        features['std_deviation'] = np.std(deviations) if deviations else 0
        features['skew_deviation'] = FeatureExtractor._skewness(deviations) if deviations else 0
        features['kurtosis_deviation'] = FeatureExtractor._kurtosis(deviations) if deviations else 0
        
        features['max_ratio'] = np.max(ratios) if ratios else 0
        features['mean_ratio'] = np.mean(ratios) if ratios else 0
        features['median_ratio'] = np.median(ratios) if ratios else 0
        
        features['max_abs_diff'] = np.max(abs_diffs) if abs_diffs else 0
        features['mean_abs_diff'] = np.mean(abs_diffs) if abs_diffs else 0
        
        features['suspicious_10pct'] = sum(1 for d in deviations if d > 10)
        features['suspicious_25pct'] = sum(1 for d in deviations if d > 25)
        features['suspicious_50pct'] = sum(1 for d in deviations if d > 50)
        features['suspicious_100pct'] = sum(1 for d in deviations if d > 100)
        
        n_signals = len(common_signals)
        features['pct_suspicious_25'] = features['suspicious_25pct'] / n_signals * 100
        features['pct_suspicious_50'] = features['suspicious_50pct'] / n_signals * 100
        
        return features
    
    @staticmethod
    def _skewness(data):
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    @staticmethod
    def _kurtosis(data):
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3


class MLTrojanDetector:
    """Machine Learning based Trojan detector"""
    
    def __init__(self, log_callback=None):
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', probability=True, random_state=42),
            'neural_net': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        }
        
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        self.threshold = 0.5
        self.log_callback = log_callback
        
    def log(self, message):
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)
    
    def train(self, X_train, y_train, optimize=True):
        """Train multiple models and select best"""
        self.log("\n[*] Training ML models...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        best_score = 0
        results = {}
        
        for name, model in self.models.items():
            self.log(f"[*] Training {name}...")
            
            if optimize and name in ['random_forest', 'gradient_boost']:
                model = self._optimize_hyperparameters(model, X_train_scaled, y_train, name)
            
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
            mean_score = cv_scores.mean()
            
            self.log(f"    CV F1: {mean_score:.4f} (+/- {cv_scores.std():.4f})")
            
            model.fit(X_train_scaled, y_train)
            
            results[name] = {
                'model': model,
                'cv_score': mean_score,
                'cv_std': cv_scores.std()
            }
            
            if mean_score > best_score:
                best_score = mean_score
                self.best_model = model
                self.best_model_name = name
        
        self.log(f"[+] Best model: {self.best_model_name} (F1: {best_score:.4f})")
        
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = self.best_model.feature_importances_
        
        return results
    
    def _optimize_hyperparameters(self, model, X, y, model_name):
        self.log(f"    Optimizing hyperparameters...")
        
        if model_name == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        elif model_name == 'gradient_boost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        else:
            return model
        
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1', n_jobs=-1)
        grid_search.fit(X, y)
        
        self.log(f"    Best params: {grid_search.best_params_}")
        return grid_search.best_estimator_
    
    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.best_model.predict(X_test_scaled)
        probabilities = self.best_model.predict_proba(X_test_scaled)[:, 1]
        return predictions, probabilities
    
    def evaluate(self, X_test, y_test):
        predictions, probabilities = self.predict(X_test)
        
        self.log("\n[*] Model Evaluation")
        self.log("="*60)
        
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        
        self.log(f"Accuracy: {accuracy:.4f}")
        self.log(f"F1-Score: {f1:.4f}")
        
        cm = confusion_matrix(y_test, predictions)
        self.log(f"\nConfusion Matrix:\n{cm}")
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': cm,
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    def save_model(self, filepath):
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'model_name': self.best_model_name,
            'threshold': self.threshold,
            'feature_importance': self.feature_importance
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.log(f"[+] Model saved to {filepath}")
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_model = model_data['model']
        self.scaler = model_data['scaler']
        self.best_model_name = model_data['model_name']
        self.threshold = model_data['threshold']
        self.feature_importance = model_data.get('feature_importance')
        
        self.log(f"[+] Model loaded from {filepath}")


class TrojanDetector:
    """Traditional threshold-based detector"""
    
    def __init__(self, clean_toggles, trojan_toggles, threshold=25.0, log_callback=None):
        self.clean_toggles = clean_toggles
        self.trojan_toggles = trojan_toggles
        self.threshold = threshold
        self.suspicious_signals = []
        self.log_callback = log_callback
        
    def log(self, message):
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)
    
    def analyze(self):
        self.log(f"\n[*] Analyzing deviations (threshold: {self.threshold:.1f}%)")
        
        common_signals = set(self.clean_toggles.keys()) & set(self.trojan_toggles.keys())
        self.log(f"[+] Comparing {len(common_signals)} common signals")
        
        results = []
        
        for signal in sorted(common_signals):
            clean_count = self.clean_toggles[signal]
            trojan_count = self.trojan_toggles[signal]
            
            if clean_count == 0:
                deviation = 999.9 if trojan_count > 0 else 0.0
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


# ============================================================================
# GUI COMPONENTS
# ============================================================================

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
        self.delete("all")
        
        if not self.enabled:
            color = "#666666"
        elif hover:
            color = self._brighten_color(self.bg)
        else:
            color = self.bg
        
        self.create_rounded_rect(2, 2, self.width-2, self.height-2, 
                                radius=8, fill=color, outline="")
        
        self.create_text(self.width//2, self.height//2, text=self.text, 
                        fill=self.fg, font=self.font)
    
    def create_rounded_rect(self, x1, y1, x2, y2, radius=25, **kwargs):
        points = [x1+radius, y1, x1+radius, y1, x2-radius, y1, x2-radius, y1,
                 x2, y1, x2, y1+radius, x2, y1+radius, x2, y2-radius, x2, y2-radius,
                 x2, y2, x2-radius, y2, x2-radius, y2, x1+radius, y2, x1+radius, y2,
                 x1, y2, x1, y2-radius, x1, y2-radius, x1, y1+radius, x1, y1+radius,
                 x1, y1]
        return self.create_polygon(points, **kwargs, smooth=True)
    
    def _brighten_color(self, hex_color):
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        rgb = tuple(min(255, int(c * 1.2)) for c in rgb)
        return '#%02x%02x%02x' % rgb
    
    def on_click(self, event):
        if self.enabled and self.command:
            self.command()
    
    def on_enter(self, event):
        if self.enabled:
            self.draw_button(hover=True)
    
    def on_leave(self, event):
        self.draw_button(hover=False)
    
    def configure_state(self, state):
        self.enabled = (state == "normal")
        self.draw_button()


# ============================================================================
# MAIN GUI APPLICATION
# ============================================================================

class IntegratedTrojanDetectorGUI:
    """Integrated GUI with ML and traditional detection"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Hardware Trojan Detector - ML-Enhanced Analysis")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#1e1e1e')
        self.root.minsize(1400, 800)
        
        # Data storage
        self.vcd_file = None
        self.results = None
        self.clean_toggles = {}
        self.trojan_toggles = {}
        self.ml_detector = MLTrojanDetector(log_callback=self.log)
        self.use_ml = tk.BooleanVar(value=False)
        self.ml_model_loaded = False
        
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
        
        self.create_widgets()
        self.auto_load_file()
    
    def create_widgets(self):
        """Create all UI widgets"""
        
        # Header
        header = tk.Frame(self.root, bg=self.colors['panel'], height=80)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        title = tk.Label(header, text="🛡️ ML-Enhanced Trojan Detector", 
                        font=("Arial", 24, "bold"), 
                        bg=self.colors['panel'], fg=self.colors['accent'])
        title.pack(side=tk.LEFT, padx=30, pady=20)
        
        subtitle = tk.Label(header, text="Machine Learning + Side-Channel Analysis", 
                           font=("Arial", 11), 
                           bg=self.colors['panel'], fg='#aaaaaa')
        subtitle.pack(side=tk.LEFT, pady=20)
        
        # Main container
        main_container = tk.Frame(self.root, bg=self.colors['bg'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel
        left_panel = tk.Frame(main_container, bg=self.colors['panel'], width=420)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Right panel
        right_panel = tk.Frame(main_container, bg=self.colors['panel'])
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Left panel contents
        self.create_left_panel(left_panel)
        
        # Right panel contents
        self.create_right_panel(right_panel)
        
        self.log("[*] ML-Enhanced Hardware Trojan Detector initialized")
        self.log("[*] Ready to load VCD file...")
    
    def create_left_panel(self, parent):
        """Create left control panel"""
        
        # File selection
        file_section = self.create_section(parent, "📁 VCD File Selection")
        
        self.file_status = tk.Label(file_section, text="No file loaded", 
                                   font=("Arial", 10), 
                                   bg=self.colors['panel'], fg='#888888',
                                   wraplength=380, justify=tk.LEFT)
        self.file_status.pack(pady=(0, 10), padx=10, anchor=tk.W)
        
        ModernButton(file_section, text="📂 Browse VCD File", 
                    command=self.load_vcd_file,
                    bg=self.colors['accent'], width=380, height=45).pack(pady=5, padx=10)
        
        # Detection mode
        mode_section = self.create_section(parent, "🤖 Detection Mode")
        
        mode_frame = tk.Frame(mode_section, bg=self.colors['panel'])
        mode_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Radiobutton(mode_frame, text="Traditional Threshold-Based", 
                      variable=self.use_ml, value=False,
                      bg=self.colors['panel'], fg=self.colors['fg'],
                      selectcolor=self.colors['button'], font=("Arial", 10),
                      activebackground=self.colors['panel'],
                      command=self.update_mode).pack(anchor=tk.W, pady=5)
        
        tk.Radiobutton(mode_frame, text="ML-Enhanced Detection", 
                      variable=self.use_ml, value=True,
                      bg=self.colors['panel'], fg=self.colors['fg'],
                      selectcolor=self.colors['button'], font=("Arial", 10),
                      activebackground=self.colors['panel'],
                      command=self.update_mode).pack(anchor=tk.W, pady=5)
        
        # ML Controls
        ml_controls = tk.Frame(mode_section, bg=self.colors['panel'])
        ml_controls.pack(fill=tk.X, padx=10, pady=5)
        
        ModernButton(ml_controls, text="📥 Load ML Model", 
                    command=self.load_ml_model,
                    bg='#9b59b6', width=185, height=38).pack(side=tk.LEFT, padx=(0, 5))
        
        ModernButton(ml_controls, text="🎓 Train New Model", 
                    command=self.train_ml_model,
                    bg='#e67e22', width=185, height=38).pack(side=tk.RIGHT)
        
        # Settings
        settings_section = self.create_section(parent, "⚙️ Analysis Settings")
        
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
        
        # Run analysis
        self.analyze_btn = ModernButton(settings_section, text="🔍 Run Analysis", 
                                       command=self.run_analysis,
                                       bg=self.colors['success'], width=380, height=50)
        self.analyze_btn.pack(pady=15, padx=10)
        self.analyze_btn.configure_state("disabled")
        
        # Results
        results_section = self.create_section(parent, "📊 Detection Results")
        
        self.status_display = tk.Label(results_section, 
                                      text="⏳ Waiting for analysis...",
                                      font=("Arial", 12, "bold"),
                                      bg=self.colors['panel'], fg='#888888',
                                      wraplength=380, pady=10)
        self.status_display.pack(pady=10, padx=10)
        
        # Stats display
        stats_frame = tk.Frame(results_section, bg=self.colors['button'], 
                              highlightbackground=self.colors['border'], highlightthickness=1)
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=8, wrap=tk.WORD, 
                                 font=("Consolas", 9),
                                 bg=self.colors['button'], fg=self.colors['fg'],
                                 relief=tk.FLAT, padx=10, pady=10)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        self.stats_text.insert(1.0, "No analysis data yet.\n\nLoad VCD and run analysis.")
        self.stats_text.config(state=tk.DISABLED)
        
        # Log section
        log_section = self.create_section(parent, "📝 Analysis Log")
        
        log_frame = tk.Frame(log_section, bg=self.colors['button'],
                            highlightbackground=self.colors['border'], highlightthickness=1)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD,
                                                  font=("Consolas", 9),
                                                  bg=self.colors['button'], fg='#00ff00',
                                                  relief=tk.FLAT, padx=10, pady=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Export buttons
        export_frame = tk.Frame(parent, bg=self.colors['panel'])
        export_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ModernButton(export_frame, text="💾 Export Report", 
                    command=self.export_report,
                    bg='#9b59b6', width=185, height=40).pack(side=tk.LEFT, padx=(0, 5))
        
        ModernButton(export_frame, text="📊 Save Chart", 
                    command=self.export_chart,
                    bg='#e67e22', width=185, height=40).pack(side=tk.RIGHT)
    
    def create_right_panel(self, parent):
        """Create right visualization panel"""
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background=self.colors['panel'], borderwidth=0)
        style.configure('TNotebook.Tab', background=self.colors['button'], 
                       foreground=self.colors['fg'], padding=[20, 10],
                       font=("Arial", 10, "bold"))
        style.map('TNotebook.Tab', background=[('selected', self.colors['accent'])],
                 foreground=[('selected', 'white')])
        
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.tab1 = tk.Frame(self.notebook, bg=self.colors['button'])
        self.notebook.add(self.tab1, text="  📊 Toggle Comparison  ")
        
        self.tab2 = tk.Frame(self.notebook, bg=self.colors['button'])
        self.notebook.add(self.tab2, text="  📈 Deviation Analysis  ")
        
        self.tab3 = tk.Frame(self.notebook, bg=self.colors['button'])
        self.notebook.add(self.tab3, text="  🤖 ML Metrics  ")
        
        self.tab4 = tk.Frame(self.notebook, bg=self.colors['button'])
        self.notebook.add(self.tab4, text="  📋 Signal Details  ")
        
        # Initialize tabs
        self.create_empty_chart(self.tab1, "Load VCD and run analysis")
        self.create_empty_chart(self.tab2, "Load VCD and run analysis")
        self.create_empty_chart(self.tab3, "ML metrics available after ML analysis")
        self.create_signal_table()
    
    def create_section(self, parent, title):
        section = tk.Frame(parent, bg=self.colors['panel'])
        section.pack(fill=tk.X, padx=10, pady=10)
        
        header = tk.Label(section, text=title, font=("Arial", 12, "bold"),
                         bg=self.colors['panel'], fg=self.colors['fg'])
        header.pack(anchor=tk.W, pady=(0, 10))
        
        return section
    
    def create_empty_chart(self, parent, message):
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
        style = ttk.Style()
        style.configure("Treeview", background=self.colors['button'],
                       foreground=self.colors['fg'], fieldbackground=self.colors['button'],
                       font=("Arial", 10))
        style.configure("Treeview.Heading", background=self.colors['panel'],
                       foreground=self.colors['fg'], font=("Arial", 10, "bold"))
        style.map('Treeview', background=[('selected', self.colors['accent'])])
        
        table_container = tk.Frame(self.tab4, bg=self.colors['button'])
        table_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        vsb = ttk.Scrollbar(table_container, orient="vertical")
        hsb = ttk.Scrollbar(table_container, orient="horizontal")
        
        self.tree = ttk.Treeview(table_container,
                                 columns=("rank", "signal", "clean", "trojan", "deviation", "status"),
                                 show="headings",
                                 yscrollcommand=vsb.set,
                                 xscrollcommand=hsb.set)
        
        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)
        
        self.tree.heading("rank", text="#")
        self.tree.heading("signal", text="Signal Name")
        self.tree.heading("clean", text="Clean Toggles")
        self.tree.heading("trojan", text="Trojan Toggles")
        self.tree.heading("deviation", text="Deviation %")
        self.tree.heading("status", text="Status")
        
        self.tree.column("rank", width=50, anchor=tk.CENTER)
        self.tree.column("signal", width=250)
        self.tree.column("clean", width=120, anchor=tk.CENTER)
        self.tree.column("trojan", width=120, anchor=tk.CENTER)
        self.tree.column("deviation", width=120, anchor=tk.CENTER)
        self.tree.column("status", width=150, anchor=tk.CENTER)
        
        self.tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        
        table_container.grid_rowconfigure(0, weight=1)
        table_container.grid_columnconfigure(0, weight=1)
    
    def auto_load_file(self):
        # Try multiple possible paths
        possible_paths = [
            r"C:\Users\sg78b\Case_CLosed-Silicon_Sprint\Case_CLosed-Silicon_Sprint.sim\sim_1\behav\xsim\alu_simulation.vcd",
            "alu_simulation.vcd",
            "./alu_simulation.vcd",
            "../alu_simulation.vcd"
        ]
        
        for default_path in possible_paths:
            if Path(default_path).exists():
                self.vcd_file = default_path
                filename = Path(default_path).name
                self.file_status.config(text=f"✅ {filename}", fg=self.colors['success'])
                self.analyze_btn.configure_state("normal")
                self.log(f"[+] Auto-loaded: {filename}")
                self.log(f"[+] Path: {default_path}")
                return
        
        self.log("[*] Default VCD file not found. Please browse to select file.")
    
    def load_vcd_file(self):
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
        self.threshold_display.config(text=f"{float(value):.1f}%")
    
    def update_mode(self):
        if self.use_ml.get():
            self.log("[*] Switched to ML-Enhanced mode")
            if not self.ml_model_loaded:
                self.log("[!] No ML model loaded. Please load or train a model.")
        else:
            self.log("[*] Switched to Traditional threshold mode")
    
    def load_ml_model(self):
        filename = filedialog.askopenfilename(
            title="Select ML Model",
            filetypes=[("Pickle Files", "*.pkl"), ("All Files", "*.*")]
        )
        
        if filename:
            try:
                self.log(f"[*] Loading ML model from {filename}...")
                self.ml_detector.load_model(filename)
                self.ml_model_loaded = True
                
                # Display model info
                if hasattr(self.ml_detector.scaler, 'feature_names_in_'):
                    n_features = len(self.ml_detector.scaler.feature_names_in_)
                    self.log(f"[+] Model expects {n_features} features")
                
                self.log(f"[+] Model type: {self.ml_detector.best_model_name}")
                self.log(f"[+] Threshold: {self.ml_detector.threshold:.3f}")
                
                messagebox.showinfo("Success", 
                    f"ML Model loaded successfully!\n\n"
                    f"Model: {self.ml_detector.best_model_name}\n"
                    f"Threshold: {self.ml_detector.threshold:.3f}")
                
            except Exception as e:
                self.log(f"[!] Failed to load model: {e}")
                import traceback
                self.log(f"[!] Traceback: {traceback.format_exc()}")
                messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
    
    def train_ml_model(self):
        """Train new ML model with synthetic data"""
        self.log("\n[*] Starting ML model training...")
        
        def train_thread():
            try:
                # Generate synthetic training data
                self.log("[*] Generating synthetic training dataset...")
                df = self.generate_synthetic_dataset(n_clean=100, n_trojan=100)
                
                X = df.drop('label', axis=1)
                y = df['label']
                X = X.fillna(0)
                
                # Store feature names for later use
                feature_names = X.columns.tolist()
                self.log(f"[+] Training with {len(feature_names)} features")
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                self.log(f"[+] Training set: {len(X_train)} samples")
                self.log(f"[+] Test set: {len(X_test)} samples")
                
                # Train
                self.ml_detector.train(X_train, y_train, optimize=True)
                
                # Store feature names in the scaler
                self.ml_detector.scaler.feature_names_in_ = np.array(feature_names)
                
                # Evaluate
                results = self.ml_detector.evaluate(X_test, y_test)
                
                # Auto-save model
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                default_filename = f"trojan_ml_model_{timestamp}.pkl"
                
                model_path = filedialog.asksaveasfilename(
                    defaultextension=".pkl",
                    filetypes=[("Pickle Files", "*.pkl")],
                    initialfile=default_filename
                )
                
                if model_path:
                    self.ml_detector.save_model(model_path)
                    self.ml_model_loaded = True
                    self.root.after(0, lambda: messagebox.showinfo("Success", 
                        f"Model trained successfully!\n\n"
                        f"Model: {self.ml_detector.best_model_name}\n"
                        f"Accuracy: {results['accuracy']:.4f}\n"
                        f"F1-Score: {results['f1_score']:.4f}\n\n"
                        f"Model saved to:\n{Path(model_path).name}"))
                else:
                    self.log("[!] Model not saved - user cancelled")
                
            except Exception as e:
                self.log(f"[!] Training error: {e}")
                import traceback
                self.log(f"[!] Traceback: {traceback.format_exc()}")
                self.root.after(0, lambda: messagebox.showerror("Error", 
                    f"Training failed:\n{str(e)}"))
        
        threading.Thread(target=train_thread, daemon=True).start()
    
    def generate_synthetic_dataset(self, n_clean=50, n_trojan=50):
        """Generate synthetic dataset for ML training"""
        np.random.seed(42)
        
        data = []
        labels = []
        
        # Clean samples
        for _ in range(n_clean):
            sample = {
                'total_toggle_ratio': np.random.uniform(0.95, 1.05),
                'total_toggle_diff': np.random.randint(-10, 10),
                'avg_clean_toggles': np.random.uniform(50, 150),
                'avg_trojan_toggles': np.random.uniform(50, 150),
                'max_deviation': np.random.uniform(0, 20),
                'mean_deviation': np.random.uniform(0, 10),
                'median_deviation': np.random.uniform(0, 8),
                'std_deviation': np.random.uniform(0, 15),
                'pct_suspicious_25': np.random.uniform(0, 5),
                'pct_suspicious_50': np.random.uniform(0, 2),
                'suspicious_25pct': np.random.randint(0, 3),
                'suspicious_50pct': np.random.randint(0, 2),
            }
            data.append(sample)
            labels.append(0)
        
        # Trojan samples
        for _ in range(n_trojan):
            sample = {
                'total_toggle_ratio': np.random.uniform(1.1, 1.5),
                'total_toggle_diff': np.random.randint(20, 100),
                'avg_clean_toggles': np.random.uniform(50, 150),
                'avg_trojan_toggles': np.random.uniform(80, 200),
                'max_deviation': np.random.uniform(30, 150),
                'mean_deviation': np.random.uniform(20, 80),
                'median_deviation': np.random.uniform(15, 60),
                'std_deviation': np.random.uniform(20, 100),
                'pct_suspicious_25': np.random.uniform(15, 50),
                'pct_suspicious_50': np.random.uniform(10, 40),
                'suspicious_25pct': np.random.randint(5, 20),
                'suspicious_50pct': np.random.randint(3, 15),
            }
            data.append(sample)
            labels.append(1)
        
        df = pd.DataFrame(data)
        df['label'] = labels
        
        return df
    
    def run_analysis(self):
        if not self.vcd_file:
            messagebox.showwarning("No File", "Please select a VCD file first.")
            return
        
        self.analyze_btn.configure_state("disabled")
        self.status_display.config(text="⏳ Analysis in progress...", fg='#f39c12')
        
        self.log("\n" + "="*60)
        self.log("Starting Hardware Trojan Detection...")
        self.log("="*60)
        
        threading.Thread(target=self._run_analysis_thread, daemon=True).start()
    
    def _run_analysis_thread(self):
        try:
            # Parse VCD
            parser = VCDParser(self.vcd_file, log_callback=self.log)
            all_toggles = parser.parse()
            
            if not all_toggles:
                raise Exception("No signals found in VCD file")
            
            # Separate signals
            self.clean_toggles, self.trojan_toggles = self.separate_signals(all_toggles)
            
            if not self.clean_toggles or not self.trojan_toggles:
                raise Exception("Could not separate clean and trojan signals")
            
            # Run detection
            threshold = self.threshold_var.get()
            detector = TrojanDetector(self.clean_toggles, self.trojan_toggles, 
                                    threshold, log_callback=self.log)
            
            self.results = detector.analyze()
            
            # ML detection if enabled
            if self.use_ml.get() and self.ml_model_loaded:
                self.log("\n[*] Running ML-based detection...")
                self.run_ml_detection()
            
            # Update GUI
            self.root.after(0, self.display_results)
            
        except Exception as e:
            self.log(f"\n[!] ERROR: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("Analysis Error", str(e)))
            self.root.after(0, lambda: self.status_display.config(
                text="❌ Analysis failed", fg=self.colors['danger']))
        finally:
            self.root.after(0, lambda: self.analyze_btn.configure_state("normal"))
    
    def run_ml_detection(self):
        """Run ML-based detection"""
        try:
            # Extract features
            extractor = FeatureExtractor()
            features = extractor.extract_statistical_features(
                self.clean_toggles, self.trojan_toggles
            )
            
            if not features:
                self.log("[!] Could not extract features for ML detection")
                return
            
            self.log(f"[+] Extracted {len(features)} features")
            
            # Create dataframe with proper column handling
            X = pd.DataFrame([features])
            X = X.fillna(0)
            
            # Check if model has the required features
            if hasattr(self.ml_detector.scaler, 'feature_names_in_'):
                model_features = self.ml_detector.scaler.feature_names_in_
                self.log(f"[+] Model expects {len(model_features)} features")
                
                # Align features with model expectations
                missing_features = set(model_features) - set(X.columns)
                extra_features = set(X.columns) - set(model_features)
                
                if missing_features:
                    self.log(f"[!] Adding {len(missing_features)} missing features")
                    for feat in missing_features:
                        X[feat] = 0
                
                if extra_features:
                    self.log(f"[!] Removing {len(extra_features)} extra features")
                    X = X.drop(columns=list(extra_features))
                
                # Ensure correct order
                X = X[model_features]
            
            # Predict
            predictions, probabilities = self.ml_detector.predict(X)
            
            ml_result = "TROJAN DETECTED" if predictions[0] == 1 else "CLEAN"
            confidence = probabilities[0] * 100 if predictions[0] == 1 else (1 - probabilities[0]) * 100
            
            self.log(f"[+] ML Prediction: {ml_result}")
            self.log(f"[+] Confidence: {confidence:.2f}%")
            self.log(f"[+] Model: {self.ml_detector.best_model_name}")
            
            # Store for visualization
            self.ml_prediction = predictions[0]
            self.ml_confidence = confidence
            
        except Exception as e:
            self.log(f"[!] ML detection error: {e}")
            import traceback
            self.log(f"[!] Traceback: {traceback.format_exc()}")
    
    def separate_signals(self, all_toggles):
        clean = {}
        trojan = {}
        
        self.log("\n[*] Separating clean and trojan signals...")
        
        for sig, count in all_toggles.items():
            sig_lower = sig.lower()
            
            if 'uut_clean' in sig_lower:
                name = sig.split('.')[-1] if '.' in sig else sig
                clean[name] = count
            elif 'uut_trojan' in sig_lower:
                name = sig.split('.')[-1] if '.' in sig else sig
                trojan[name] = count
            elif 'clean' in sig_lower and 'trojan' not in sig_lower:
                name = sig.replace('clean', '').replace('_', '').split('.')[-1]
                if name:
                    clean[name] = count
            elif 'trojan' in sig_lower:
                name = sig.replace('trojan', '').replace('_', '').split('.')[-1]
                if name:
                    trojan[name] = count
        
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
        
        self.log(f"[+] Found {len(clean)} clean, {len(trojan)} trojan signals")
        return clean, trojan
    
    def display_results(self):
        if not self.results:
            return
        
        suspicious = [r for r in self.results if r['suspicious']]
        
        # Determine status
        if self.use_ml.get() and self.ml_model_loaded and hasattr(self, 'ml_prediction'):
            if self.ml_prediction == 1:
                status_text = f"🚨 ML: TROJAN ({self.ml_confidence:.1f}%)"
                status_color = self.colors['danger']
            else:
                status_text = f"✅ ML: CLEAN ({self.ml_confidence:.1f}%)"
                status_color = self.colors['success']
        else:
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
        percent_diff = (diff / total_clean) * 100 if total_clean > 0 else 0
        
        stats = f"""
╔════════════════════════════════╗
║   DETECTION STATISTICS         ║
╚════════════════════════════════╝

Mode: {"ML-Enhanced" if self.use_ml.get() else "Traditional"}
Signals: {len(self.results)}
Suspicious: {len(suspicious)}
Threshold: {self.threshold_var.get():.1f}%

╔════════════════════════════════╗
║   TOGGLE ACTIVITY              ║
╚════════════════════════════════╝

Clean:  {total_clean:,} toggles
Trojan: {total_trojan:,} toggles
Diff:   {diff:+,} ({percent_diff:+.2f}%)

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
        if not self.results:
            return
        
        significant = [r for r in self.results if r['clean'] > 5 or r['trojan'] > 5]
        significant.sort(key=lambda x: x['deviation'], reverse=True)
        plot_data = significant[:25]
        
        if not plot_data:
            return
        
        signals = [r['signal'][:20] for r in plot_data]
        clean_counts = [r['clean'] for r in plot_data]
        trojan_counts = [r['trojan'] for r in plot_data]
        deviations = [min(r['deviation'], 200) for r in plot_data]
        suspicious = [r['suspicious'] for r in plot_data]
        
        # Clear old charts
        for widget in self.tab1.winfo_children():
            widget.destroy()
        for widget in self.tab2.winfo_children():
            widget.destroy()
        
        # Create charts
        self.create_toggle_chart(self.tab1, signals, clean_counts, trojan_counts, suspicious)
        self.create_deviation_chart(self.tab2, signals, deviations, suspicious)
        
        # ML metrics if available
        if self.use_ml.get() and hasattr(self, 'ml_prediction'):
            self.create_ml_metrics_chart()
    
    def create_toggle_chart(self, parent, signals, clean, trojan, suspicious):
        fig = Figure(figsize=(12, 7), facecolor=self.colors['button'])
        ax = fig.add_subplot(111, facecolor=self.colors['button'])
        
        x = np.arange(len(signals))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, clean, width, label='Clean Design',
                      color='#2ecc71', alpha=0.9, edgecolor='white', linewidth=0.8)
        bars2 = ax.bar(x + width/2, trojan, width, label='Trojan Design',
                      color='#e74c3c', alpha=0.9, edgecolor='white', linewidth=0.8)
        
        for i, susp in enumerate(suspicious):
            if susp:
                ax.axvspan(i - 0.5, i + 0.5, alpha=0.15, color='yellow', zorder=0)
        
        ax.set_xlabel('Signal Name', color=self.colors['fg'], fontsize=12, fontweight='bold')
        ax.set_ylabel('Toggle Count', color=self.colors['fg'], fontsize=12, fontweight='bold')
        ax.set_title('Toggle Activity Comparison', color=self.colors['fg'], 
                    fontsize=14, fontweight='bold', pad=15)
        
        ax.set_xticks(x)
        ax.set_xticklabels(signals, rotation=45, ha='right', fontsize=9, color=self.colors['fg'])
        ax.tick_params(colors=self.colors['fg'])
        ax.legend(facecolor=self.colors['panel'], edgecolor=self.colors['fg'],
                 labelcolor=self.colors['fg'])
        ax.grid(True, alpha=0.2, color='white', linestyle='--', linewidth=0.5)
        
        for spine in ax.spines.values():
            spine.set_color(self.colors['fg'])
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
    
    def create_deviation_chart(self, parent, signals, deviations, suspicious):
        fig = Figure(figsize=(12, 7), facecolor=self.colors['button'])
        ax = fig.add_subplot(111, facecolor=self.colors['button'])
        
        x = np.arange(len(signals))
        colors = [self.colors['danger'] if s else '#95a5a6' for s in suspicious]
        
        bars = ax.bar(x, deviations, color=colors, alpha=0.9, 
                     edgecolor='white', linewidth=0.8)
        
        threshold = self.threshold_var.get()
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2.5,
                  label=f'Threshold ({threshold:.0f}%)', zorder=10)
        
        ax.set_xlabel('Signal Name', color=self.colors['fg'], fontsize=12, fontweight='bold')
        ax.set_ylabel('Deviation (%)', color=self.colors['fg'], fontsize=12, fontweight='bold')
        ax.set_title('Deviation Analysis', color=self.colors['fg'], 
                    fontsize=14, fontweight='bold', pad=15)
        
        ax.set_xticks(x)
        ax.set_xticklabels(signals, rotation=45, ha='right', fontsize=9, color=self.colors['fg'])
        ax.tick_params(colors=self.colors['fg'])
        ax.legend(facecolor=self.colors['panel'], edgecolor=self.colors['fg'],
                 labelcolor=self.colors['fg'])
        ax.grid(True, alpha=0.2, color='white', linestyle='--', linewidth=0.5)
        
        for spine in ax.spines.values():
            spine.set_color(self.colors['fg'])
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
    
    def create_ml_metrics_chart(self):
        for widget in self.tab3.winfo_children():
            widget.destroy()
        
        fig = Figure(figsize=(12, 7), facecolor=self.colors['button'])
        
        # Prediction confidence gauge
        ax1 = fig.add_subplot(121, facecolor=self.colors['button'])
        
        prediction_text = "TROJAN" if self.ml_prediction == 1 else "CLEAN"
        color = self.colors['danger'] if self.ml_prediction == 1 else self.colors['success']
        
        ax1.barh([0], [self.ml_confidence], color=color, alpha=0.8)
        ax1.set_xlim(0, 100)
        ax1.set_ylim(-0.5, 0.5)
        ax1.set_xlabel('Confidence (%)', color=self.colors['fg'], fontweight='bold')
        ax1.set_title(f'ML Prediction: {prediction_text}', 
                     color=self.colors['fg'], fontweight='bold', fontsize=14)
        ax1.set_yticks([])
        ax1.tick_params(colors=self.colors['fg'])
        ax1.grid(True, alpha=0.2, axis='x')
        
        for spine in ax1.spines.values():
            spine.set_color(self.colors['fg'])
        
        # Model info
        ax2 = fig.add_subplot(122, facecolor=self.colors['button'])
        ax2.axis('off')
        
        info_text = f"""
        ML DETECTION RESULTS
        {'='*40}
        
        Model: {self.ml_detector.best_model_name}
        
        Prediction: {prediction_text}
        Confidence: {self.ml_confidence:.2f}%
        
        Threshold: {self.ml_detector.threshold:.3f}
        
        Status: {'⚠️ Trojan Detected' if self.ml_prediction == 1 else '✅ Clean Design'}
        """
        
        ax2.text(0.1, 0.5, info_text, fontsize=12, 
                verticalalignment='center', fontfamily='monospace',
                color=self.colors['fg'],
                bbox=dict(boxstyle='round', facecolor=self.colors['panel'], 
                         alpha=0.5, edgecolor=self.colors['border']))
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.tab3)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        toolbar = NavigationToolbar2Tk(canvas, self.tab3)
        toolbar.update()
    
    def update_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        sorted_results = sorted(self.results, key=lambda x: x['deviation'], reverse=True)
        
        for rank, result in enumerate(sorted_results, 1):
            dev_str = f"{result['deviation']:.1f}%" if result['deviation'] < 999 else "999.9%"
            status = "⚠️ SUSPICIOUS" if result['suspicious'] else "✓ Normal"
            tag = 'suspicious' if result['suspicious'] else 'normal'
            
            self.tree.insert("", tk.END, values=(
                rank,
                result['signal'],
                result['clean'],
                result['trojan'],
                dev_str,
                status
            ), tags=(tag,))
        
        self.tree.tag_configure('suspicious', background='#3d1f1f', foreground='#ff6b6b')
        self.tree.tag_configure('normal', background=self.colors['button'], 
                               foreground=self.colors['fg'])
    
    def export_report(self):
        if not self.results:
            messagebox.showwarning("No Data", "Please run analysis first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            initialfile=f"trojan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("="*80 + "\n")
                    f.write("HARDWARE TROJAN DETECTION REPORT\n")
                    f.write("ML-Enhanced Side-Channel Analysis\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("="*80 + "\n\n")
                    
                    # Summary
                    suspicious = [r for r in self.results if r['suspicious']]
                    total_clean = sum(r['clean'] for r in self.results)
                    total_trojan = sum(r['trojan'] for r in self.results)
                    
                    f.write("EXECUTIVE SUMMARY\n")
                    f.write("-"*80 + "\n")
                    f.write(f"VCD File: {Path(self.vcd_file).name}\n")
                    f.write(f"Detection Mode: {'ML-Enhanced' if self.use_ml.get() else 'Traditional'}\n")
                    f.write(f"Threshold: {self.threshold_var.get():.1f}%\n")
                    f.write(f"Signals Analyzed: {len(self.results)}\n")
                    f.write(f"Suspicious Signals: {len(suspicious)}\n\n")
                    
                    # ML Results
                    if self.use_ml.get() and hasattr(self, 'ml_prediction'):
                        f.write("ML DETECTION RESULTS\n")
                        f.write("-"*80 + "\n")
                        f.write(f"Model: {self.ml_detector.best_model_name}\n")
                        f.write(f"Prediction: {'TROJAN' if self.ml_prediction == 1 else 'CLEAN'}\n")
                        f.write(f"Confidence: {self.ml_confidence:.2f}%\n\n")
                    
                    f.write(f"Toggle Activity:\n")
                    f.write(f"  Clean:  {total_clean:,} toggles\n")
                    f.write(f"  Trojan: {total_trojan:,} toggles\n")
                    f.write(f"  Diff:   {total_trojan - total_clean:+,} toggles\n\n")
                    
                    # Threat assessment
                    if len(suspicious) == 0:
                        threat = "LOW - No anomalies detected"
                    elif len(suspicious) <= 3:
                        threat = "MEDIUM - Suspicious activity"
                    else:
                        threat = "HIGH - Trojan confirmed"
                    
                    f.write(f"THREAT LEVEL: {threat}\n\n")
                    
                    # Suspicious signals
                    if suspicious:
                        f.write("="*80 + "\n")
                        f.write("SUSPICIOUS SIGNALS\n")
                        f.write("="*80 + "\n\n")
                        f.write(f"{'Rank':<6}{'Signal':<35}{'Clean':<10}{'Trojan':<10}{'Deviation':<12}\n")
                        f.write("-"*80 + "\n")
                        
                        for i, sig in enumerate(suspicious, 1):
                            dev = f"{sig['deviation']:.1f}%" if sig['deviation'] < 999 else "999.9%"
                            f.write(f"{i:<6}{sig['signal']:<35}{sig['clean']:<10}{sig['trojan']:<10}{dev:<12}\n")
                        f.write("\n")
                    
                    # All signals
                    f.write("="*80 + "\n")
                    f.write("COMPLETE ANALYSIS (Top 50)\n")
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
                messagebox.showinfo("Success", f"Report saved to:\n{filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export:\n{str(e)}")
    
    def export_chart(self):
        if not self.results:
            messagebox.showwarning("No Data", "Please run analysis first.")
            return
        
        current_tab = self.notebook.index(self.notebook.select())
        
        if current_tab >= 3:
            messagebox.showinfo("Info", "Cannot export table view. Switch to chart tab.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("PDF Document", "*.pdf")],
            initialfile=f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        
        if filename:
            try:
                if current_tab == 0:
                    tab = self.tab1
                elif current_tab == 1:
                    tab = self.tab2
                else:
                    tab = self.tab3
                
                for widget in tab.winfo_children():
                    if isinstance(widget, tk.Canvas):
                        canvas = widget.master
                        if hasattr(canvas, 'figure'):
                            canvas.figure.savefig(filename, dpi=300, bbox_inches='tight',
                                                facecolor=self.colors['button'])
                            self.log(f"[+] Chart exported: {Path(filename).name}")
                            messagebox.showinfo("Success", f"Chart saved to:\n{filename}")
                            return
                
                messagebox.showerror("Error", "Could not find chart to export.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export:\n{str(e)}")
    
    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.update_idletasks()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main application entry point"""
    
    parser = argparse.ArgumentParser(description='Integrated ML Hardware Trojan Detector')
    parser.add_argument('--mode', choices=['gui', 'cli', 'train'], default='gui',
                       help='Operation mode (default: gui)')
    parser.add_argument('--vcd', type=str, help='VCD file path for CLI mode')
    parser.add_argument('--threshold', type=float, default=25.0,
                       help='Detection threshold (default: 25.0)')
    parser.add_argument('--train-model', action='store_true',
                       help='Train new ML model')
    parser.add_argument('--model', type=str, default='trojan_model.pkl',
                       help='Model file path')
    
    args = parser.parse_args()
    
    if args.mode == 'gui':
        # Launch GUI
        root = tk.Tk()
        
        try:
            root.iconbitmap('icon.ico')
        except:
            pass
        
        app = IntegratedTrojanDetectorGUI(root)
        
        # Center window
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f'{width}x{height}+{x}+{y}')
        
        root.mainloop()
        
    elif args.mode == 'cli':
        # CLI mode
        if not args.vcd:
            print("[!] Error: --vcd required for CLI mode")
            return
        
        print("="*60)
        print("HARDWARE TROJAN DETECTOR - CLI MODE")
        print("="*60)
        
        # Parse VCD
        parser = VCDParser(args.vcd)
        all_toggles = parser.parse()
        
        # Separate signals (simplified for CLI)
        signals = list(all_toggles.keys())
        mid = len(signals) // 2
        clean_toggles = {s: all_toggles[s] for s in signals[:mid]}
        trojan_toggles = {s: all_toggles[s] for s in signals[mid:]}
        
        # Detect
        detector = TrojanDetector(clean_toggles, trojan_toggles, args.threshold)
        results = detector.analyze()
        
        # Display results
        suspicious = [r for r in results if r['suspicious']]
        print(f"\n[+] Found {len(suspicious)} suspicious signals")
        
        if suspicious:
            print("\nSuspicious Signals:")
            print(f"{'Signal':<40} {'Clean':<10} {'Trojan':<10} {'Deviation':<10}")
            print("-"*70)
            for sig in suspicious:
                print(f"{sig['signal']:<40} {sig['clean']:<10} {sig['trojan']:<10} {sig['deviation']:.1f}%")
    
    elif args.mode == 'train':
        # Training mode
        print("="*60)
        print("TRAINING ML MODEL")
        print("="*60)
        
        # Generate synthetic data
        print("[*] Generating synthetic dataset...")
        np.random.seed(42)
        
        def generate_sample(is_trojan):
            if is_trojan:
                return {
                    'total_toggle_ratio': np.random.uniform(1.1, 1.5),
                    'total_toggle_diff': np.random.randint(20, 100),
                    'max_deviation': np.random.uniform(30, 150),
                    'mean_deviation': np.random.uniform(20, 80),
                    'pct_suspicious_25': np.random.uniform(15, 50),
                    'pct_suspicious_50': np.random.uniform(10, 40),
                }
            else:
                return {
                    'total_toggle_ratio': np.random.uniform(0.95, 1.05),
                    'total_toggle_diff': np.random.randint(-10, 10),
                    'max_deviation': np.random.uniform(0, 20),
                    'mean_deviation': np.random.uniform(0, 10),
                    'pct_suspicious_25': np.random.uniform(0, 5),
                    'pct_suspicious_50': np.random.uniform(0, 2),
                }
        
        data = []
        labels = []
        
        for _ in range(100):
            data.append(generate_sample(False))
            labels.append(0)
        
        for _ in range(100):
            data.append(generate_sample(True))
            labels.append(1)
        
        df = pd.DataFrame(data)
        df['label'] = labels
        
        X = df.drop('label', axis=1)
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train
        detector = MLTrojanDetector()
        detector.train(X_train, y_train, optimize=True)
        
        # Evaluate
        results = detector.evaluate(X_test, y_test)
        
        # Save
        detector.save_model(args.model)
        
        print(f"\n[✓] Model saved to {args.model}")
        print(f"[✓] Accuracy: {results['accuracy']:.4f}")
        print(f"[✓] F1-Score: {results['f1_score']:.4f}")


if __name__ == "__main__":
    main()
