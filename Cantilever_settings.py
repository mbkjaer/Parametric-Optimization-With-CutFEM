import numpy as np
Settings = {
    "Material_settings": {
        "E": 1,          # Young's Modulus (Pa)
        "nu": 0.3,          # Poisson's Ratio
    },
    "Geometric_settings_rectangle": {
        "thickness": 0.01,  # Thickness of the beam (m)
        "length": 1,      # Length of the cantilever (m)
        "height": 0.1       # Height of the cantilever (m)
    },
    "Geometric_settings_spline": {
        "thickness": 0.01,  # Thickness of the beam (m)
        "height": 20,
        "length": 17,  # Length of the cantilever (m)
        "x_spline_pos": np.linspace(0,17,6),  # X position of knotpoints (m)
        "y_spline_pos": np.ones(6)*15,  # Y position of knotpoints (m)
        "y_lower_lim": 1.5,
        "y_upper_lim": 20,
        "p_order": 3,
        "num_gauss_segments_in_partial":10
    },
    "Mesh_settings": {
        "nx": 17,           # Number of elements along length
        "ny": 20,            # Number of elements along height
        "mesh_px": 3,
        "mesh_py": 3
    },
    "mode": {
        "convex_constraints": 0,
        "lagrange_penalty": 0,
        "gradient_penalty": 0,
        "gamma_BC": 0,
        "link_right_node_to_next": 0,
        "BC_only_top": 1,
        "basis_removal": 1
    },
    "Load_settings": {
        "point_load": -1 # Point load at the tip (N)
    },
    "Plotting_settings": {
        "scale_factor": 1e2,  # Amplify displacements for visualization
        "figsize": (10, 4)    # Figure size
    }
}