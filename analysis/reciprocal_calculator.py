from typing import Sequence, Tuple
import numpy as np
import scipy.spatial.transform as transform
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
from ipywidgets import FloatSlider, IntSlider, Checkbox, HBox, VBox, interactive_output
from IPython.display import display
import mplcursors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class ReciprocalCalculator:
    def __init__(self, a_len, b_len, c_len, alpha_deg, beta_deg, gamma_deg):
        self.set_lattice(a_len, b_len, c_len, alpha_deg, beta_deg, gamma_deg)

    def set_lattice(self, a_len, b_len, c_len, alpha_deg, beta_deg, gamma_deg):
        self.a_len, self.b_len, self.c_len = a_len, b_len, c_len
        self.alpha_deg, self.beta_deg, self.gamma_deg = alpha_deg, beta_deg, gamma_deg
        self.a_vec, self.b_vec, self.c_vec = self._calc_real_space_abc(
            a_len, b_len, c_len, alpha_deg, beta_deg, gamma_deg
        )
        self.a_star, self.b_star, self.c_star = self._calc_reciprocal_space(
            self.a_vec, self.b_vec, self.c_vec
        )

    @staticmethod
    def _calc_real_space_abc(a_len, b_len, c_len, alpha_deg, beta_deg, gamma_deg):
        α = np.deg2rad(alpha_deg)
        β = np.deg2rad(beta_deg)
        γ = np.deg2rad(gamma_deg)
        V = (a_len * b_len * c_len *
             np.sqrt(1 - np.cos(α)**2 - np.cos(β)**2 - np.cos(γ)**2
                     + 2*np.cos(α)*np.cos(β)*np.cos(γ)))
        a = np.array([a_len, 0.0, 0.0])
        b = np.array([b_len * np.cos(γ), b_len * np.sin(γ), 0.0])
        c = np.array([
            c_len * np.cos(β),
            c_len * (np.cos(α) - np.cos(β)*np.cos(γ)) / np.sin(γ),
            V / (a_len * b_len * np.sin(γ))
        ])
        return a, b, c

    @staticmethod
    def _calc_reciprocal_space(a, b, c):
        vol = np.dot(a, np.cross(b, c))
        factor = 2 * np.pi / vol
        return (factor * np.cross(b, c),
                factor * np.cross(c, a),
                factor * np.cross(a, b))

    @staticmethod
    def rotate_vector(v, axis, angle_deg):
        """
        Rotates a vector 'v' around the 'axis' by 'angle_degree' degrees using quaternions.
        """
        angle_rad = np.deg2rad(angle_deg)
        axis = axis / np.linalg.norm(axis)
        q = transform.Rotation.from_rotvec(axis * angle_rad)
        return q.apply(v)

    def calculate_q_chi(self, h, k, l):
        q_vec = h*self.a_star + k*self.b_star + l*self.c_star
        q_mag = np.linalg.norm(q_vec)
        if q_mag > 0:
            cos_chi = np.clip(q_vec[2]/q_mag, -1, 1)
            chi_deg = np.degrees(np.arccos(cos_chi))
        else:
            chi_deg = 0.0
        return q_mag, chi_deg

    def find_peaks(self, hkl_range=range(-4,10), target_q=None, tol=0.1):
        peaks = [((h, k, l), *self.calculate_q_chi(h,k,l))
                 for h in hkl_range for k in hkl_range for l in hkl_range]
        if target_q is not None:
            peaks = [p for p in peaks if abs(p[1] - target_q) <= tol]
        return peaks

    def update_reciprocal(self, a_vec, b_vec, c_vec):
        self.a_star, self.b_star, self.c_star = self._calc_reciprocal_space(
            a_vec, b_vec, c_vec)
