# File: ewald/analysis/bragg_calculator.py
"""
BraggCalculator: compute Bragg peak positions (q_xy, q_z) and Miller indices for a given unit cell and sample rotation.
"""
import numpy as np
import math

class BraggCalculator:
    def __init__(self, a, b, c, alpha, beta, gamma):
        """
        Initialize with lattice parameters:
          a, b, c (Å) and angles alpha, beta, gamma (degrees).
        Precompute direct lattice vectors a1,a2,a3.
        """
        # Convert angles to radians
        alpha_r = math.radians(alpha)
        beta_r  = math.radians(beta)
        gamma_r = math.radians(gamma)
        # Direct lattice vectors in Cartesian
        self.a1 = np.array([a, 0.0, 0.0])
        self.a2 = np.array([
            b * math.cos(gamma_r),
            b * math.sin(gamma_r),
            0.0
        ])
        # c vector with components c_x, c_y, c_z
        c_x = c * math.cos(beta_r)
        c_y = c * (math.cos(alpha_r) - math.cos(beta_r)*math.cos(gamma_r)) / math.sin(gamma_r)
        c_z = math.sqrt(max(0.0, c**2 - c_x**2 - c_y**2))
        self.a3 = np.array([c_x, c_y, c_z])

    def compute_peaks(self, orientation=(0.0, 0.0, 0.0), hkl_range=(1, 1, 1)):
        """
        Compute Bragg peak positions for Miller indices in [-h..h],[-k..k],[-l..l],
        applying sample rotations thetax (X-axis) and thetay (Y-axis) in degrees.
        Returns:
          q_xy: numpy array of in-plane magnitudes
          q_z : numpy array of out-of-plane components
          hkl : numpy array of shape (N,3) of Miller indices
        """
        # Unpack
        thetax, thetay, _ = orientation
        hmax, kmax, lmax = hkl_range
        # Rotation matrices
        tx = math.radians(thetax)
        ty = math.radians(thetay)
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(tx), -math.sin(tx)],
            [0, math.sin(tx),  math.cos(tx)]
        ])
        Ry = np.array([
            [ math.cos(ty), 0, -math.sin(ty)],
            [0, 1, 0],
            [ math.sin(ty), 0,  math.cos(ty)]
        ])

        # Rotate direct cell
        M = np.vstack([self.a1, self.a2, self.a3])  # shape (3,3)
        M = M @ Rx
        M = M @ Ry
        aa1, aa2, aa3 = M[0], M[1], M[2]

        # Reciprocal lattice
        V = np.dot(aa3, np.cross(aa1, aa2))
        b1 = 2*math.pi * np.cross(aa2, aa3) / V
        b2 = 2*math.pi * np.cross(aa3, aa1) / V
        b3 = 2*math.pi * np.cross(aa1, aa2) / V

        # Generate Miller index grid
        i = np.arange(-hmax, hmax+1)
        H, K, L = np.meshgrid(i, i, i, indexing='ij')

        # Compute reciprocal space vectors
        G1 = H * b1[0] + K * b2[0] + L * b3[0]
        G2 = H * b1[1] + K * b2[1] + L * b3[1]
        G3 = H * b1[2] + K * b2[2] + L * b3[2]

        # Flatten and compute q_xy, q_z
        q_xy = np.hypot(G1, G2).ravel()
        q_z  = G3.ravel()
        hkl  = np.vstack((H.ravel(), K.ravel(), L.ravel())).T

        # Exclude the (0,0,0) reflection
        mask = ~((hkl[:,0]==0) & (hkl[:,1]==0) & (hkl[:,2]==0))
        return q_xy[mask], q_z[mask], hkl[mask]
