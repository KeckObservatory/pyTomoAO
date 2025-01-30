#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Optics Reconstructor System

This module provides classes for wavefront reconstruction in adaptive optics systems,
including matrix loading, covariance calculations, and reconstructor assembly.
"""

import os
import logging
import configparser
import dataclasses
import struct
import numpy as np
from scipy.sparse import issparse, csr_matrix
from scipy.sparse.linalg import svds
from scipy.linalg import svd
from scipy.special import gamma, kv
from scipy.signal import convolve2d

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('AOReconstructor')

@dataclasses.dataclass
class ReconMatrices:
    """Dataclass to hold reconstruction matrices and parameters"""
    weight0: np.ndarray
    KL_matrix: np.ndarray
    invcov: np.ndarray
    ttp: np.ndarray
    sub_actuator_map: np.ndarray
    actuator_actuator_map: np.ndarray
    foccents: np.ndarray
    sub_aperture_mask: np.ndarray
    act_mask: np.ndarray

class MatrixLoader:
    """Handles loading of configuration files and reconstruction matrices"""
    
    @staticmethod
    def load_config(config_file: str, ao_mode: str = 'default', config_dir: str = None) -> dict:
        """
        Load configuration from INI file
        Args:
            config_file: Name of configuration file
            ao_mode: AO mode section to load
            config_dir: Directory containing config file
        Returns:
            Dictionary of configuration parameters
        """
        config = configparser.ConfigParser()
        config.optionxform = str
        config_path = os.path.join(config_dir or os.path.dirname(__file__), config_file)
        config.read(config_path)
        return MatrixLoader._safe_parse_config(config[ao_mode])

    @staticmethod
    def _safe_parse_config(config_section: configparser.SectionProxy) -> dict:
        """Safely parse configuration values avoiding eval()"""
        parsed = {}
        for key, value in config_section.items():
            try:
                parsed[key] = int(value)
            except ValueError:
                try:
                    parsed[key] = float(value)
                except ValueError:
                    if value.lower() in ('true', 'false'):
                        parsed[key] = value.lower() == 'true'
                    else:
                        parsed[key] = value
        return parsed

    @staticmethod
    def load_recon_matrices(config: dict) -> ReconMatrices:
        """Load all reconstruction matrices from configured paths"""
        base_path = config['path2parms']
        return ReconMatrices(
            weight0=MatrixLoader._load_matrix(base_path + config['weight0'], np.int8),
            KL_matrix=MatrixLoader._load_binary(base_path + config['KL_matrix'], 
                                              (config['nb_actuators'], config['nb_actuators'])),
            invcov=MatrixLoader._load_binary(base_path + config['invcov'], 
                                           (config['nb_actuators'], config['nb_actuators'])),
            ttp=MatrixLoader._load_binary(base_path + config['ttp'], 
                                        (config['nb_actuators'], config['nb_actuators'])),
            sub_actuator_map=MatrixLoader._load_sub_actuator_map(config),
            actuator_actuator_map=MatrixLoader._load_actuator_map(config),
            foccents=MatrixLoader._load_foccents(config),
            sub_aperture_mask=MatrixLoader._load_mask(config, 'sub_aperture_mask'),
            act_mask=MatrixLoader._load_mask(config, 'actuator_mask')
        )

    @staticmethod
    def _load_binary(filename: str, shape: tuple, fmt: str = ">f") -> np.ndarray:
        """Load binary file with given data format"""
        try:
            with open(filename, "rb") as f:
                data = np.array(struct.unpack(f">{shape[0]*shape[1]}f", f.read()))
            return data.reshape(shape)
        except FileNotFoundError:
            log.error(f"Binary file not found: {filename}")
            raise

    @staticmethod
    def _load_mask(config: dict, mask_key: str) -> np.ndarray:
        """Load and convert mask file to boolean array"""
        path = os.path.join(config['path2parms'], config[mask_key])
        return np.loadtxt(path).astype(bool)

    @staticmethod
    def _load_sub_actuator_map(config: dict) -> np.ndarray:
        """Load and reshape sub-actuator map"""
        data = np.fromfile(config['path2parms'] + config['sub_actuator_map'], dtype=np.int8)
        return data.reshape((config['nb_sub_apertures'], config['nb_actuators'])).T

    @staticmethod
    def _load_actuator_map(config: dict) -> np.ndarray:
        """Load actuator-actuator map"""
        data = np.fromfile(config['path2parms'] + config['actuator_actuator_map'], dtype=np.int8)
        return data.reshape((config['nb_actuators'], config['nb_actuators']))

    @staticmethod
    def _load_foccents(config: dict) -> np.ndarray:
        """Load and process focus centroids"""
        path = os.path.join(config['path2moreparms'], config['zernToCent'])
        data = MatrixLoader._load_binary(path, (2*config['nb_sub_apertures'], 10))
        return data[:, 3]

class CovarianceCalculator:
    """Handles atmospheric covariance matrix calculations"""
    
    @staticmethod
    def phase_covariance(rho: np.ndarray, r0: float, L0: float) -> np.ndarray:
        """
        Compute phase covariance matrix using Von Karman model
        Args:
            rho: Matrix of separation distances
            r0: Fried parameter
            L0: Outer scale
        Returns:
            Covariance matrix
        """
        L0r0_ratio = (L0 / r0) ** (5/3)
        constant = (24 * gamma(6/5)/5) ** (5/6) * gamma(11/6) / (2**(5/6) * np.pi**(8/3)) * L0r0_ratio
        
        cov = np.full_like(rho, constant * gamma(5/6)/(2**(1/6)*np.pi**(8/3)) * L0r0_ratio)
        non_zero = rho != 0
        u = 2 * np.pi * rho[non_zero] / L0
        cov[non_zero] = constant * u**(5/6) * kv(5/6, u)
        return cov

    @staticmethod
    def spatial_angular_covariance(telescope, atmosphere, source1, source2, mask, oversampling=2, dm_space=False):
        """
        Compute cross-covariance matrix between two guide stars
        Args:
            telescope: Telescope parameters
            atmosphere: Atmospheric parameters
            source1: First guide star
            source2: Second guide star
            mask: Subaperture mask
            oversampling: Oversampling factor
            dm_space: Flag for DM space calculation
        Returns:
            Cross-covariance matrix
        """
        grid = ReconstructionGrid(mask, oversampling, dm_space)
        arcsec_to_rad = np.pi / (180 * 3600)
        
        covariance = np.zeros((grid.size, grid.size, len(atmosphere.layers)))
        for layer_idx, layer in enumerate(atmosphere.layers):
            # Calculate compressed coordinates for each source
            rho1 = CovarianceCalculator._compressed_coords(source1, layer, grid, arcsec_to_rad)
            rho2 = CovarianceCalculator._compressed_coords(source2, layer, grid, arcsec_to_rad)
            
            distances = CovarianceCalculator.pairwise_distances(rho1.T, rho2.T)
            covariance[:, :, layer_idx] = CovarianceCalculator.phase_covariance(
                distances, atmosphere.r0, atmosphere.L0) * layer.fractional_r0
        
        return np.sum(covariance, axis=2)[grid.mask.flatten()][:, grid.mask.flatten()]

    @staticmethod
    def _compressed_coords(source, layer, grid, conversion_factor):
        """Calculate compressed coordinates for a source"""
        compression = 1 - layer.altitude / source.altitude
        x = source.x * layer.altitude * conversion_factor * np.cos(np.radians(source.y))
        y = source.y * layer.altitude * conversion_factor * np.sin(np.radians(source.x))
        return ReconstructionGrid.meshgrid(grid.size, grid.D, x, y, compression, compression)

    @staticmethod
    def pairwise_distances(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between two sets of points"""
        return np.abs(np.subtract.outer(points1.flatten(), points2.flatten()))

class ReconstructionGrid:
    """Handles reconstruction grid geometry calculations"""
    
    def __init__(self, mask: np.ndarray, oversampling: int, dm_space: bool):
        self.mask = self._compute_grid(mask, oversampling, dm_space)
        self.size = self.mask.shape[0]
        self.D = 1.0  # Normalized diameter

    @staticmethod
    def _compute_grid(mask: np.ndarray, os: int, dm_space: bool) -> np.ndarray:
        """Compute valid grid points based on mask and oversampling"""
        if os == 1 and not dm_space:
            return convolve2d(np.ones((2, 2)), mask, mode='same').astype(bool)
        elif os == 2:
            n = os * mask.shape[0] + 1
            valid = np.zeros((n, n), dtype=bool)
            valid[1::2, 1::2] = mask
            return convolve2d(valid, np.ones((3, 3)), mode='same').astype(bool)
        return mask

    @staticmethod
    def meshgrid(n_pts: int, diameter: float, offset_x: float, offset_y: float,
                 stretch_x: float = 1, stretch_y: float = 1) -> tuple:
        """Generate stretched and offset meshgrid"""
        x = np.linspace(-diameter/2, diameter/2, n_pts)
        X, Y = np.meshgrid(x*stretch_x + offset_x, x*stretch_y + offset_y)
        return X + 1j*Y  # Return complex coordinates for compact storage

class Reconstructor:
    """Main reconstructor class for wavefront reconstruction"""
    
    def __init__(self, config_file: str, ao_mode: str = 'default'):
        self.config = MatrixLoader.load_config(config_file, ao_mode)
        self.matrices = MatrixLoader.load_recon_matrices(self.config)
        self.covariance_calculator = CovarianceCalculator()

    def assemble_reconstructor(self, valid_subap_mask: np.ndarray, valid_act_mask: np.ndarray) -> np.ndarray:
        """
        Assemble complete reconstructor matrix
        Args:
            valid_subap_mask: Valid subapertures mask
            valid_act_mask: Valid actuators mask
        Returns:
            Reconstruction matrix
        """
        # 1. Load base matrices
        R_base = self.matrices.KL_matrix @ self.matrices.invcov
        
        # 2. Apply modal filtering
        R_filtered = self._apply_modal_filtering(R_base)
        
        # 3. Apply actuator masking
        R_masked = self._apply_actuator_masking(R_filtered, valid_act_mask)
        
        # 4. Apply subaperture masking
        return self._apply_subap_masking(R_masked, valid_subap_mask)

    def _apply_modal_filtering(self, R: np.ndarray) -> np.ndarray:
        """Apply tip/tilt/piston removal"""
        return R @ self.matrices.ttp

    def _apply_actuator_masking(self, R: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply valid actuator mask"""
        return R[:, mask]

    def _apply_subap_masking(self, R: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply valid subaperture mask"""
        return R[mask.flatten(), :]

class SVDHandler:
    """Handles Singular Value Decomposition operations"""
    
    @staticmethod
    def truncated_svd(matrix: np.ndarray, n_modes: int = 0) -> tuple:
        """
        Perform truncated SVD
        Args:
            matrix: Input matrix
            n_modes: Number of modes to keep (0 = keep all)
        Returns:
            U, S, V matrices
        """
        if issparse(matrix):
            U, S, Vt = svds(matrix, k=matrix.shape[0]-n_modes if n_modes else None)
        else:
            U, S, Vt = svd(matrix, full_matrices=False)
            if n_modes:
                U = U[:, :-n_modes]
                S = S[:-n_modes]
                Vt = Vt[:-n_modes, :]
        return U, S, Vt

    @staticmethod
    def tsvd_reconstruct(U: np.ndarray, S: np.ndarray, Vt: np.ndarray) -> np.ndarray:
        """Reconstruct matrix from truncated SVD components"""
        return (Vt.T @ np.diag(1/S)) @ U.T

# Example usage
if __name__ == "__main__":
    reconstructor = Reconstructor("ao_config.ini", "LGS")
    valid_subap = np.loadtxt("valid_subap.txt").astype(bool)
    valid_act = np.loadtxt("valid_act.txt").astype(bool)
    R = reconstructor.assemble_reconstructor(valid_subap, valid_act)
    print("Reconstructor matrix shape:", R.shape)