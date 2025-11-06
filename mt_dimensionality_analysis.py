"""
Magnetotelluric (MT) Data Dimensionality Analysis
Analyzes EDI files to determine 1D, 2D, or 3D subsurface structure
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path


class EDIParser:
    """Parser for EDI (Electrical Data Interchange) format files"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = {}
        
    def parse(self):
        """Parse EDI file and extract impedance tensor data"""
        with open(self.filepath, 'r') as f:
            lines = f.readlines()
        
        # Extract station ID
        for line in lines:
            if 'DATAID' in line:
                self.data['station_id'] = line.split('=')[1].strip().replace('"', '')
                break
        
        # Parse frequency and impedance data
        self.data['freq'] = self._extract_data_block(lines, '>FREQ')
        self.data['zxxr'] = self._extract_data_block(lines, '>ZXXR')
        self.data['zxxi'] = self._extract_data_block(lines, '>ZXXI')
        self.data['zxyr'] = self._extract_data_block(lines, '>ZXYR')
        self.data['zxyi'] = self._extract_data_block(lines, '>ZXYI')
        self.data['zyxr'] = self._extract_data_block(lines, '>ZYXR')
        self.data['zyxi'] = self._extract_data_block(lines, '>ZYXI')
        self.data['zyyr'] = self._extract_data_block(lines, '>ZYYR')
        self.data['zyyi'] = self._extract_data_block(lines, '>ZYYI')
        
        # Build complex impedance tensor
        self.data['Zxx'] = np.array(self.data['zxxr']) + 1j * np.array(self.data['zxxi'])
        self.data['Zxy'] = np.array(self.data['zxyr']) + 1j * np.array(self.data['zxyi'])
        self.data['Zyx'] = np.array(self.data['zyxr']) + 1j * np.array(self.data['zyxi'])
        self.data['Zyy'] = np.array(self.data['zyyr']) + 1j * np.array(self.data['zyyi'])
        
        return self.data
    
    def _extract_data_block(self, lines, marker):
        """Extract numerical data following a marker line"""
        data = []
        capture = False
        
        for line in lines:
            if marker in line:
                capture = True
                continue
            
            if capture:
                # Stop at next section marker
                if line.strip().startswith('>'):
                    break
                
                # Extract numbers from line
                values = line.strip().split()
                for val in values:
                    try:
                        data.append(float(val))
                    except ValueError:
                        continue
        
        return np.array(data)


class MTDimensionalityAnalyzer:
    """Analyzes MT data to determine subsurface dimensionality"""
    
    def __init__(self, edi_data):
        self.data = edi_data
        self.results = {}
    
    def calculate_strike_angle(self):
        """
        Calculate impedance strike angle using Swift's method
        Strike angle is the rotation angle that minimizes diagonal elements
        Returns angle in degrees (0-180)
        """
        Zxx = self.data['Zxx']
        Zxy = self.data['Zxy']
        Zyx = self.data['Zyx']
        Zyy = self.data['Zyy']
        
        # Calculate strike angle using arctan formula
        numerator = (Zxx - Zyy).real * (Zxy + Zyx).real + (Zxx - Zyy).imag * (Zxy + Zyx).imag
        denominator = (Zxx - Zyy).real * (Zxy + Zyx).imag - (Zxx - Zyy).imag * (Zxy + Zyx).real
        
        with np.errstate(divide='ignore', invalid='ignore'):
            strike_angle = (np.arctan2(numerator, denominator) * 180 / np.pi) / 2
            # Normalize to 0-180 degrees
            strike_angle = strike_angle % 180
        
        self.results['strike_angle'] = strike_angle
        return strike_angle
    
    def calculate_phase_tensor(self):
        """
        Calculate phase tensor parameters
        Phase tensor Φ = X^(-1) * Y where Z = X + iY
        """
        X = np.array([[self.data['Zxx'].real, self.data['Zxy'].real],
                      [self.data['Zyx'].real, self.data['Zyy'].real]])
        Y = np.array([[self.data['Zxx'].imag, self.data['Zxy'].imag],
                      [self.data['Zyx'].imag, self.data['Zyy'].imag]])
        
        n_freq = len(self.data['freq'])
        beta = np.zeros(n_freq)  # Phase tensor skew angle
        alpha = np.zeros(n_freq)  # Phase tensor principal direction
        phi_max = np.zeros(n_freq)  # Maximum phase
        phi_min = np.zeros(n_freq)  # Minimum phase
        
        for i in range(n_freq):
            Xi = np.array([[X[0, 0][i], X[0, 1][i]], [X[1, 0][i], X[1, 1][i]]])
            Yi = np.array([[Y[0, 0][i], Y[0, 1][i]], [Y[1, 0][i], Y[1, 1][i]]])
            
            try:
                Phi = np.linalg.inv(Xi) @ Yi
                
                # Calculate phase tensor parameters
                phi_11 = Phi[0, 0]
                phi_12 = Phi[0, 1]
                phi_21 = Phi[1, 0]
                phi_22 = Phi[1, 1]
                
                # Beta: phase tensor skew angle
                beta[i] = 0.5 * np.arctan2((phi_12 - phi_21), (phi_11 + phi_22)) * 180 / np.pi
                
                # Alpha: principal direction
                alpha[i] = 0.5 * np.arctan2((phi_12 + phi_21), (phi_11 - phi_22)) * 180 / np.pi
                
                # Principal phases
                trace = phi_11 + phi_22
                det = phi_11 * phi_22 - phi_12 * phi_21
                discriminant = trace**2 - 4*det
                
                if discriminant >= 0:
                    phi_max[i] = np.arctan((trace + np.sqrt(discriminant)) / 2) * 180 / np.pi
                    phi_min[i] = np.arctan((trace - np.sqrt(discriminant)) / 2) * 180 / np.pi
                else:
                    phi_max[i] = np.nan
                    phi_min[i] = np.nan
                    
            except np.linalg.LinAlgError:
                beta[i] = np.nan
                alpha[i] = np.nan
                phi_max[i] = np.nan
                phi_min[i] = np.nan
        
        self.results['phase_tensor_beta'] = beta
        self.results['phase_tensor_alpha'] = alpha
        self.results['phi_max'] = phi_max
        self.results['phi_min'] = phi_min
        
        return beta, alpha, phi_max, phi_min
        
    def calculate_swift_skew(self):
        """
        Calculate Swift Skew parameter
        K = |Zxx + Zyy| / |Zxy - Zyx|
        """
        S1 = self.data['Zxx'] + self.data['Zyy']
        D2 = self.data['Zxy'] - self.data['Zyx']
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            swift_skew = np.abs(S1) / np.abs(D2)
            swift_skew = np.where(np.abs(D2) < 1e-10, np.nan, swift_skew)
        
        self.results['swift_skew'] = swift_skew
        return swift_skew
    
    def calculate_bahr_skew(self):
        """
        Calculate Bahr Phase Sensitive Skew
        η = |(S1·D1* - S2·D2*)| / |D2|²
        where * denotes complex conjugate
        """
        S1 = self.data['Zxx'] + self.data['Zyy']
        S2 = self.data['Zxy'] + self.data['Zyx']
        D1 = self.data['Zxx'] - self.data['Zyy']
        D2 = self.data['Zxy'] - self.data['Zyx']
        
        numerator = S1 * np.conj(D1) - S2 * np.conj(D2)
        denominator = np.abs(D2) ** 2
        
        with np.errstate(divide='ignore', invalid='ignore'):
            bahr_skew = np.abs(numerator) / denominator
            bahr_skew = np.where(denominator < 1e-10, np.nan, bahr_skew)
        
        self.results['bahr_skew'] = bahr_skew
        return bahr_skew
    
    def check_1d_condition(self, threshold=0.1):
        """
        Check if data satisfies 1D condition
        For 1D: |Zxx| ≈ 0 and |Zyy| ≈ 0 compared to off-diagonal elements
        """
        zxx_mag = np.abs(self.data['Zxx'])
        zyy_mag = np.abs(self.data['Zyy'])
        zxy_mag = np.abs(self.data['Zxy'])
        zyx_mag = np.abs(self.data['Zyx'])
        
        # Calculate ratio of diagonal to off-diagonal elements
        avg_offdiag = (zxy_mag + zyx_mag) / 2
        
        with np.errstate(divide='ignore', invalid='ignore'):
            zxx_ratio = zxx_mag / avg_offdiag
            zyy_ratio = zyy_mag / avg_offdiag
            zxx_ratio = np.where(avg_offdiag < 1e-10, np.nan, zxx_ratio)
            zyy_ratio = np.where(avg_offdiag < 1e-10, np.nan, zyy_ratio)
        
        is_1d = (zxx_ratio < threshold) & (zyy_ratio < threshold)
        
        self.results['zxx_ratio'] = zxx_ratio
        self.results['zyy_ratio'] = zyy_ratio
        self.results['is_1d'] = is_1d
        
        return is_1d
    
    def classify_dimensionality(self, swift_threshold=0.15, bahr_threshold=0.15, diag_threshold=0.1):
        """
        Classify each frequency point as 1D, 2D, or 3D
        
        Logic:
        - 1D: |Zxx| and |Zyy| << |Zxy| and |Zyx|
        - 2D: Swift Skew < 0.15 AND Bahr Skew < 0.15
        - 3D: Otherwise
        """
        is_1d = self.check_1d_condition(threshold=diag_threshold)
        swift_skew = self.results['swift_skew']
        bahr_skew = self.results['bahr_skew']
        
        classification = np.empty(len(is_1d), dtype=object)
        
        for i in range(len(is_1d)):
            if is_1d[i]:
                classification[i] = '1D'
            elif (not np.isnan(swift_skew[i]) and not np.isnan(bahr_skew[i]) and
                  swift_skew[i] < swift_threshold and bahr_skew[i] < bahr_threshold):
                classification[i] = '2D'
            else:
                classification[i] = '3D'
        
        self.results['classification'] = classification
        return classification
    
    def analyze(self):
        """Run complete dimensionality analysis"""
        self.calculate_swift_skew()
        self.calculate_bahr_skew()
        self.calculate_strike_angle()
        self.calculate_phase_tensor()
        self.classify_dimensionality()
        return self.results


class MTVisualizer:
    """Create visualization plots for MT dimensionality analysis"""
    
    def __init__(self, edi_data, results, output_dir='results'):
        self.data = edi_data
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def plot_skew_parameters(self):
        """Plot Swift and Bahr skew vs frequency"""
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        freq = self.data['freq']
        swift_skew = self.results['swift_skew']
        bahr_skew = self.results['bahr_skew']
        
        # Swift Skew plot
        axes[0].loglog(freq, swift_skew, 'bo-', markersize=4, linewidth=1.5, label='Swift Skew')
        axes[0].axhline(y=0.15, color='r', linestyle='--', linewidth=2, label='2D/3D Threshold (0.15)')
        axes[0].set_xlabel('Frequency (Hz)', fontsize=11)
        axes[0].set_ylabel('Swift Skew (K)', fontsize=11)
        axes[0].set_title(f'Swift Skew vs Frequency - {self.data["station_id"]}', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Bahr Skew plot
        axes[1].loglog(freq, bahr_skew, 'go-', markersize=4, linewidth=1.5, label='Bahr Skew')
        axes[1].axhline(y=0.15, color='r', linestyle='--', linewidth=2, label='2D/3D Threshold (0.15)')
        axes[1].set_xlabel('Frequency (Hz)', fontsize=11)
        axes[1].set_ylabel('Bahr Phase Sensitive Skew (η)', fontsize=11)
        axes[1].set_title(f'Bahr Phase Sensitive Skew vs Frequency - {self.data["station_id"]}', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        filename = self.output_dir / f'{self.data["station_id"]}_skew_parameters.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    def plot_impedance_components(self):
        """Plot impedance tensor component magnitudes"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        freq = self.data['freq']
        
        ax.loglog(freq, np.abs(self.data['Zxx']), 'r^-', markersize=4, label='|Zxx|', alpha=0.7)
        ax.loglog(freq, np.abs(self.data['Zxy']), 'bo-', markersize=4, label='|Zxy|', alpha=0.7)
        ax.loglog(freq, np.abs(self.data['Zyx']), 'gs-', markersize=4, label='|Zyx|', alpha=0.7)
        ax.loglog(freq, np.abs(self.data['Zyy']), 'mv-', markersize=4, label='|Zyy|', alpha=0.7)
        
        ax.set_xlabel('Frequency (Hz)', fontsize=11)
        ax.set_ylabel('Impedance Magnitude (Ω)', fontsize=11)
        ax.set_title(f'Impedance Tensor Components - {self.data["station_id"]}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        plt.tight_layout()
        filename = self.output_dir / f'{self.data["station_id"]}_impedance_components.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    def plot_dimensionality_classification(self):
        """Plot dimensionality classification with frequency"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        freq = self.data['freq']
        classification = self.results['classification']
        
        # Create numerical representation for plotting
        class_num = np.array([{'1D': 1, '2D': 2, '3D': 3}[c] for c in classification])
        
        colors = {'1D': 'green', '2D': 'blue', '3D': 'red'}
        
        for dim_type in ['1D', '2D', '3D']:
            mask = classification == dim_type
            if np.any(mask):
                ax.semilogx(freq[mask], class_num[mask], 'o', markersize=8, 
                           color=colors[dim_type], label=dim_type, alpha=0.7)
        
        ax.set_xlabel('Frequency (Hz)', fontsize=11)
        ax.set_ylabel('Dimensionality', fontsize=11)
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(['1D', '2D', '3D'])
        ax.set_ylim([0.5, 3.5])
        ax.set_title(f'Dimensionality Classification - {self.data["station_id"]}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        filename = self.output_dir / f'{self.data["station_id"]}_dimensionality.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    def plot_strike_rose_diagram(self):
        """Plot rose diagram showing strike directions"""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
        
        strike_angles = self.results['strike_angle']
        freq = self.data['freq']
        
        # Remove NaN values
        valid_mask = ~np.isnan(strike_angles)
        strike_valid = strike_angles[valid_mask]
        freq_valid = freq[valid_mask]
        
        # Convert to radians
        strike_rad = np.deg2rad(strike_valid)
        
        # Create histogram of strike directions
        n_bins = 36  # 5-degree bins
        hist, bin_edges = np.histogram(strike_valid, bins=n_bins, range=(0, 180))
        
        # Convert bin edges to radians (double for full circle)
        theta = np.deg2rad(np.concatenate([bin_edges[:-1], bin_edges[:-1] + 180]))
        radii = np.concatenate([hist, hist])
        width = np.deg2rad(180 / n_bins)
        
        # Plot rose diagram
        bars = ax.bar(theta, radii, width=width, bottom=0.0, alpha=0.7, edgecolor='black', linewidth=1)
        
        # Color bars by frequency (log scale)
        norm = plt.Normalize(vmin=np.log10(freq_valid.min()), vmax=np.log10(freq_valid.max()))
        for bar in bars:
            bar.set_facecolor(plt.cm.viridis(0.5))
        
        # Also plot individual strike directions
        for i, (angle, f) in enumerate(zip(strike_valid, freq_valid)):
            # Plot on both sides (bidirectional)
            angle_rad = np.deg2rad(angle)
            color = plt.cm.plasma(norm(np.log10(f)))
            ax.plot([angle_rad, angle_rad], [0, max(radii)*1.1], 
                   'o-', markersize=3, alpha=0.3, color=color)
            ax.plot([angle_rad + np.pi, angle_rad + np.pi], [0, max(radii)*1.1], 
                   'o-', markersize=3, alpha=0.3, color=color)
        
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title(f'Strike Direction Rose Diagram - {self.data["station_id"]}', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('Frequency Count', labelpad=30)
        
        # Add legend
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('Log10(Frequency Hz)', rotation=270, labelpad=20)
        
        plt.tight_layout()
        filename = self.output_dir / f'{self.data["station_id"]}_strike_rose.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    def plot_phase_tensor_rose(self):
        """Plot phase tensor ellipses as rose diagram"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='polar')
        
        alpha = self.results['phase_tensor_alpha']
        beta = self.results['phase_tensor_beta']
        phi_max = self.results['phi_max']
        phi_min = self.results['phi_min']
        freq = self.data['freq']
        
        # Plot every Nth frequency to avoid overcrowding
        step = max(1, len(freq) // 20)
        
        for i in range(0, len(freq), step):
            if not (np.isnan(alpha[i]) or np.isnan(beta[i])):
                # Convert angles to radians
                alpha_rad = np.deg2rad(alpha[i])
                
                # Create ellipse data
                theta_ellipse = np.linspace(0, 2*np.pi, 50)
                
                # Ellipse parameters (simplified)
                a = max(abs(phi_max[i]), 1)  # Semi-major axis
                b = max(abs(phi_min[i]), 0.5)  # Semi-minor axis
                
                # Parametric ellipse
                x_ellipse = a * np.cos(theta_ellipse)
                y_ellipse = b * np.sin(theta_ellipse)
                
                # Rotate by alpha
                x_rot = x_ellipse * np.cos(alpha_rad) - y_ellipse * np.sin(alpha_rad)
                y_rot = x_ellipse * np.sin(alpha_rad) + y_ellipse * np.cos(alpha_rad)
                
                # Convert to polar
                r_ellipse = np.sqrt(x_rot**2 + y_rot**2)
                theta_pol = np.arctan2(y_rot, x_rot)
                
                # Color by frequency
                color = plt.cm.viridis(i / len(freq))
                ax.plot(theta_pol, r_ellipse, alpha=0.6, linewidth=1.5, color=color)
                ax.fill(theta_pol, r_ellipse, alpha=0.2, color=color)
        
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title(f'Phase Tensor Ellipses - {self.data["station_id"]}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add colorbar for frequency
        norm = plt.Normalize(vmin=np.log10(freq.min()), vmax=np.log10(freq.max()))
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('Log10(Frequency Hz)', rotation=270, labelpad=20)
        
        plt.tight_layout()
        filename = self.output_dir / f'{self.data["station_id"]}_phase_tensor_rose.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    def plot_impedance_polar(self):
        """Plot impedance magnitude as polar plot"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 14), subplot_kw=dict(projection='polar'))
        
        freq = self.data['freq']
        components = [
            ('Zxx', self.data['Zxx'], axes[0, 0], 'red'),
            ('Zxy', self.data['Zxy'], axes[0, 1], 'blue'),
            ('Zyx', self.data['Zyx'], axes[1, 0], 'green'),
            ('Zyy', self.data['Zyy'], axes[1, 1], 'purple')
        ]
        
        for name, Z, ax, color in components:
            magnitude = np.abs(Z)
            phase = np.angle(Z)  # radians
            
            # Sort by phase for smoother plotting
            sorted_indices = np.argsort(phase)
            phase_sorted = phase[sorted_indices]
            mag_sorted = magnitude[sorted_indices]
            freq_sorted = freq[sorted_indices]
            
            # Create color map based on frequency
            norm = plt.Normalize(vmin=np.log10(freq.min()), vmax=np.log10(freq.max()))
            colors = plt.cm.plasma(norm(np.log10(freq_sorted)))
            
            # Plot as scatter
            scatter = ax.scatter(phase_sorted, mag_sorted, c=np.log10(freq_sorted), 
                               s=50, alpha=0.7, cmap='plasma', edgecolors='black', linewidth=0.5)
            
            ax.set_theta_zero_location('E')
            ax.set_title(f'{name} Impedance', fontsize=12, fontweight='bold', pad=10)
            ax.set_ylabel('Magnitude (Ω)', labelpad=30)
            ax.grid(True, alpha=0.3)
        
        # Add single colorbar for all subplots
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
        norm = plt.Normalize(vmin=np.log10(freq.min()), vmax=np.log10(freq.max()))
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Log10(Frequency Hz)', rotation=270, labelpad=20)
        
        fig.suptitle(f'Impedance Polar Plots - {self.data["station_id"]}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        filename = self.output_dir / f'{self.data["station_id"]}_impedance_polar.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    def create_all_plots(self):
        """Generate all visualization plots"""
        self.plot_skew_parameters()
        self.plot_impedance_components()
        self.plot_dimensionality_classification()
        self.plot_strike_rose_diagram()
        self.plot_phase_tensor_rose()
        self.plot_impedance_polar()


class MTReportGenerator:
    """Generate text and CSV reports for MT analysis"""
    
    def __init__(self, edi_data, results, output_dir='results'):
        self.data = edi_data
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def save_csv(self):
        """Save detailed results to CSV file"""
        df = pd.DataFrame({
            'Frequency_Hz': self.data['freq'],
            'Swift_Skew': self.results['swift_skew'],
            'Bahr_Skew': self.results['bahr_skew'],
            'Zxx_Magnitude': np.abs(self.data['Zxx']),
            'Zxy_Magnitude': np.abs(self.data['Zxy']),
            'Zyx_Magnitude': np.abs(self.data['Zyx']),
            'Zyy_Magnitude': np.abs(self.data['Zyy']),
            'Zxx_Ratio': self.results['zxx_ratio'],
            'Zyy_Ratio': self.results['zyy_ratio'],
            'Dimensionality': self.results['classification']
        })
        
        filename = self.output_dir / f'{self.data["station_id"]}_analysis_results.csv'
        df.to_csv(filename, index=False, float_format='%.6e')
        print(f"Saved: {filename}")
        return df
    
    def generate_text_report(self):
        """Generate comprehensive text report"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"MAGNETOTELLURIC DIMENSIONALITY ANALYSIS REPORT")
        report_lines.append(f"Station: {self.data['station_id']}")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Summary statistics
        report_lines.append("SUMMARY STATISTICS")
        report_lines.append("-" * 80)
        report_lines.append(f"Number of frequencies analyzed: {len(self.data['freq'])}")
        report_lines.append(f"Frequency range: {self.data['freq'].min():.4e} - {self.data['freq'].max():.4e} Hz")
        report_lines.append("")
        
        # Swift Skew statistics
        swift_valid = self.results['swift_skew'][~np.isnan(self.results['swift_skew'])]
        if len(swift_valid) > 0:
            report_lines.append("Swift Skew Statistics:")
            report_lines.append(f"  Mean: {np.mean(swift_valid):.4f}")
            report_lines.append(f"  Median: {np.median(swift_valid):.4f}")
            report_lines.append(f"  Min: {np.min(swift_valid):.4f}")
            report_lines.append(f"  Max: {np.max(swift_valid):.4f}")
            report_lines.append("")
        
        # Bahr Skew statistics
        bahr_valid = self.results['bahr_skew'][~np.isnan(self.results['bahr_skew'])]
        if len(bahr_valid) > 0:
            report_lines.append("Bahr Phase Sensitive Skew Statistics:")
            report_lines.append(f"  Mean: {np.mean(bahr_valid):.4f}")
            report_lines.append(f"  Median: {np.median(bahr_valid):.4f}")
            report_lines.append(f"  Min: {np.min(bahr_valid):.4f}")
            report_lines.append(f"  Max: {np.max(bahr_valid):.4f}")
            report_lines.append("")
        
        # Dimensionality classification summary
        report_lines.append("DIMENSIONALITY CLASSIFICATION SUMMARY")
        report_lines.append("-" * 80)
        classification = self.results['classification']
        
        for dim_type in ['1D', '2D', '3D']:
            count = np.sum(classification == dim_type)
            percentage = (count / len(classification)) * 100
            report_lines.append(f"{dim_type}: {count} frequencies ({percentage:.1f}%)")
        
        report_lines.append("")
        
        # Overall classification
        unique, counts = np.unique(classification, return_counts=True)
        dominant = unique[np.argmax(counts)]
        report_lines.append(f"OVERALL CLASSIFICATION: {dominant}")
        report_lines.append("")
        
        # Detailed frequency-by-frequency results
        report_lines.append("DETAILED FREQUENCY-BY-FREQUENCY RESULTS")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Freq (Hz)':<12} {'Swift Skew':<12} {'Bahr Skew':<12} {'Classification':<15}")
        report_lines.append("-" * 80)
        
        for i in range(len(self.data['freq'])):
            freq_str = f"{self.data['freq'][i]:.4e}"
            swift_str = f"{self.results['swift_skew'][i]:.4f}" if not np.isnan(self.results['swift_skew'][i]) else "N/A"
            bahr_str = f"{self.results['bahr_skew'][i]:.4f}" if not np.isnan(self.results['bahr_skew'][i]) else "N/A"
            class_str = classification[i]
            report_lines.append(f"{freq_str:<12} {swift_str:<12} {bahr_str:<12} {class_str:<15}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        # Save report
        filename = self.output_dir / f'{self.data["station_id"]}_analysis_report.txt'
        with open(filename, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Saved: {filename}")
        
        # Also print to console
        print("\n" + '\n'.join(report_lines))
        
        return '\n'.join(report_lines)
    
    def generate_all_reports(self):
        """Generate all reports"""
        self.save_csv()
        self.generate_text_report()


def main():
    """Main analysis workflow"""
    print("=" * 80)
    print("MAGNETOTELLURIC DIMENSIONALITY ANALYSIS")
    print("=" * 80)
    print()
    
    # Define EDI file paths
    edi_files = [
        'Run1_free_31_ed.edi'
    ]
    
    # Process each EDI file
    for edi_file in edi_files:
        print(f"\nProcessing: {edi_file}")
        print("-" * 80)
        
        # Parse EDI file
        parser = EDIParser(edi_file)
        edi_data = parser.parse()
        print(f"Station: {edi_data['station_id']}")
        print(f"Frequencies: {len(edi_data['freq'])}")
        
        # Analyze dimensionality
        analyzer = MTDimensionalityAnalyzer(edi_data)
        results = analyzer.analyze()
        
        # Create visualizations
        visualizer = MTVisualizer(edi_data, results)
        visualizer.create_all_plots()
        
        # Generate reports
        report_gen = MTReportGenerator(edi_data, results)
        report_gen.generate_all_reports()
        
        print()
    
    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print("All results saved to 'results/' directory")
    print("=" * 80)


if __name__ == "__main__":
    main()

