import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from Cantilever_settings import Settings
from time import time
from B_spline_class import BSplineInterpolator
from B_spline_surface_class import BSplineSurfaceInterpolator
from scipy.optimize import minimize
from numba import njit

def indices_to_keep_basis_removal(A, threshold):
    # Find indices where the diagonal elements are less than or equal to the threshold
    indices_to_keep = np.where(np.diag(A) > threshold)[0]
    indices_to_remove = np.where(np.diag(A) <= threshold)[0]
    return indices_to_keep,indices_to_remove

def principal_stress_components(sigma_1, sigma_2, theta_p):
    # Components of sigma_1
    sigma_1_x = sigma_1 * np.cos(theta_p)
    sigma_1_y = sigma_1 * np.sin(theta_p)

    # Components of sigma_2 (which is orthogonal to sigma_1)
    sigma_2_x = sigma_2 * np.cos(theta_p + np.pi / 2)
    sigma_2_y = sigma_2 * np.sin(theta_p + np.pi / 2)

    return sigma_1_x, sigma_1_y, sigma_2_x, sigma_2_y


def principal_stresses_and_angle(sigma_x, sigma_y, tau_xy):
    # Calculate the average stress
    sigma_avg = (sigma_x + sigma_y) / 2

    # Calculate the radius of Mohr's Circle
    R = np.sqrt((sigma_x - sigma_y) ** 2 / 4 + tau_xy ** 2)

    # Calculate the principal stresses
    sigma_1 = sigma_avg + R
    sigma_2 = sigma_avg - R

    theta_p = np.where(sigma_x == sigma_y,
                       np.where(tau_xy != 0, np.pi / 4, 0),
                       np.arctan2(2 * tau_xy, sigma_x - sigma_y) / 2)

    return sigma_1, sigma_2, theta_p

def fill_values_axis_last(inp, n):
    *dims, c = inp.shape
    reshaped_inp = inp.reshape(-1, c)
    result_reshaped = np.repeat(reshaped_inp[:, n-1::n], n, axis=1)[:, :c]
    return result_reshaped.reshape(*dims, c)


def include_lagrange_in_K(K, G):
    top = np.hstack((K, G.T))
    bottom = np.hstack((G, np.zeros((G.shape[0], G.shape[0]))))
    K_with_lagrange = np.vstack((top, bottom))
    return K_with_lagrange

@njit(cache=True, fastmath=True)
def K_BC_inner_builder(dN_dx_partials, dN_dy_partials, normal_x, normal_y, gauss_weight, n):
    K_BC_penalty = np.zeros((dN_dx_partials.shape[1] * 2, dN_dx_partials.shape[1] * 2))
    nu,E=0.3,1
    for p_index in range(1, 2):
        a = (dN_dx_partials[:, :, p_index] * (normal_x[:, np.newaxis] ** 2+nu*normal_y[:, np.newaxis] ** 2)
             +dN_dy_partials[:, :, p_index] * (1 - nu) * normal_x[:,np.newaxis] * normal_y[:,np.newaxis])
        b = (dN_dy_partials[:, :, p_index] * (nu*normal_x[:, np.newaxis] ** 2+normal_y[:, np.newaxis] ** 2)
             +dN_dx_partials[:, :, p_index] * (1 - nu) * normal_x[:,np.newaxis] * normal_y[:,np.newaxis])


        D_gauss_all = np.zeros((a.shape[0], 2 * a.shape[1]), dtype=a.dtype)
        D_gauss_all[:, ::2] = a
        D_gauss_all[:, 1::2] = b
        D_gauss_all_w_gp = D_gauss_all * gauss_weight[:,np.newaxis]
        for arg_1 in range(dN_dx_partials.shape[1] * 2):
            for arg_2 in range(arg_1 + 1):  # Iterate only up to arg_1 to exploit symmetry
                D_gauss = D_gauss_all_w_gp[:, arg_1] * D_gauss_all_w_gp[:, arg_2]
                D = np.sum(D_gauss)
                K_BC_penalty[arg_1, arg_2] += D
                if arg_1 != arg_2:  # If not on the diagonal, mirror the value
                    K_BC_penalty[arg_2, arg_1] += K_BC_penalty[arg_1, arg_2]
    return K_BC_penalty*E

@njit(cache=True, fastmath=True)
def K_BC_inner_builder_deriv(dN_dx_partials,dN_dy_partials,dN_dx_partials_da,dN_dy_partials_da,normal_x,normal_y,normal_x_da,normal_y_da,gauss_weight,gauss_weight_deriv,n):
    nu,E=0.3,1
    K_BC_penalty_deriv=np.zeros((dN_dx_partials_da.shape[0],dN_dx_partials.shape[1]*2,dN_dx_partials.shape[1]*2))
    for alpha_index in range(dN_dx_partials_da.shape[0]):
        for p_index in range(1, 2):
            #"a is Nx displacements b is Ny displacement
            da_1_1 = dN_dx_partials_da[alpha_index,:, :, p_index] * (normal_x[:, np.newaxis] ** 2 + nu * normal_y[:, np.newaxis] ** 2)
            da_1_2 = dN_dx_partials[:, :, p_index]*(2*normal_x[:, np.newaxis]*normal_x_da[alpha_index,:, np.newaxis]+nu*2*normal_y[:, np.newaxis]*normal_y_da[alpha_index,:, np.newaxis])
            da_2_1 = dN_dy_partials_da[alpha_index,:, :, p_index] * (1 - nu) * normal_x[:, np.newaxis] * normal_y[:, np.newaxis]
            da_2_2 = dN_dy_partials[:, :, p_index] * (1 - nu) * normal_x_da[alpha_index,:, np.newaxis] * normal_y[:, np.newaxis]
            da_2_3 = dN_dy_partials[:, :, p_index] * (1 - nu) * normal_x[:, np.newaxis] * normal_y_da[alpha_index, :, np.newaxis]
            da=da_1_1+da_1_2+da_2_1+da_2_2+da_2_3

            db_1_1 = dN_dy_partials_da[alpha_index,:, :, p_index] * (nu * normal_x[:, np.newaxis] ** 2 + normal_y[:, np.newaxis] ** 2)
            db_1_2 = dN_dy_partials[:, :, p_index]*(nu*2*normal_x[:, np.newaxis]*normal_x_da[alpha_index,:, np.newaxis]+2*normal_y[:, np.newaxis]*normal_y_da[alpha_index,:, np.newaxis])
            db_2_1 = dN_dx_partials_da[alpha_index,:, :, p_index] * (1 - nu) * normal_x[:, np.newaxis] * normal_y[:, np.newaxis]
            db_2_2 = dN_dx_partials[:, :, p_index] * (1 - nu) * normal_x_da[alpha_index,:, np.newaxis] * normal_y[:, np.newaxis]
            db_2_3 = dN_dx_partials[:, :, p_index] * (1 - nu) * normal_x[:, np.newaxis] * normal_y_da[alpha_index, :, np.newaxis]
            db=db_1_1+db_1_2+db_2_1+db_2_2+db_2_3

            b = (dN_dy_partials[:, :, p_index] * (nu*normal_x[:, np.newaxis] ** 2+normal_y[:, np.newaxis] ** 2)
                 +dN_dx_partials[:, :, p_index] * (1 - nu) * normal_x[:,np.newaxis] * normal_y[:,np.newaxis])

            a = (dN_dx_partials[:, :, p_index] * (normal_x[:, np.newaxis] ** 2+nu*normal_y[:, np.newaxis] ** 2)
                 +dN_dy_partials[:, :, p_index] * (1 - nu) * normal_x[:,np.newaxis] * normal_y[:,np.newaxis])


            D_gauss_all = np.zeros((a.shape[0], 2 * a.shape[1]), dtype=a.dtype)
            dD_gauss_all = np.zeros((a.shape[0], 2 * a.shape[1]), dtype=a.dtype)
            D_gauss_all[:, ::2] = a
            D_gauss_all[:, 1::2] = b
            dD_gauss_all[:, ::2] = da
            dD_gauss_all[:, 1::2] = db
            D_gauss_all_w_gp = D_gauss_all * gauss_weight[:,np.newaxis]
            dD_gauss_all_w_gp = dD_gauss_all * gauss_weight[:, np.newaxis]+D_gauss_all * gauss_weight_deriv[alpha_index,:, np.newaxis]
            for arg_1 in range(dN_dx_partials.shape[1] * 2):
                for arg_2 in range(arg_1 + 1):  # Iterate only up to arg_1 to exploit symmetry
                    #D_gauss = D_gauss_all_w_gp[:, arg_1] * D_gauss_all_w_gp[:, arg_2]
                    dD_gauss = dD_gauss_all_w_gp[:, arg_1] * D_gauss_all_w_gp[:, arg_2]+D_gauss_all_w_gp[:, arg_1] * dD_gauss_all_w_gp[:, arg_2]
                    dD = np.sum(dD_gauss)
                    K_BC_penalty_deriv[alpha_index,arg_1, arg_2] = dD
                    if arg_1 != arg_2:  # If not on the diagonal, mirror the value
                        K_BC_penalty_deriv[alpha_index,arg_2, arg_1] = K_BC_penalty_deriv[alpha_index,arg_1, arg_2]
    return K_BC_penalty_deriv*E


@njit(cache=True, fastmath=True)
def G_BC_inner_builder(dN_dx_partials,dN_dy_partials,normal_x,normal_y,gauss_weight,n):
    nu,E=0.3,1
    n_gp=len(normal_x)
    n_ele=n_gp//n
    G = np.zeros((n_ele,dN_dx_partials.shape[1]*2))

    p_index = 1
    a = (dN_dx_partials[:, :, p_index] * (normal_x[:, np.newaxis] ** 2 + nu * normal_y[:, np.newaxis] ** 2)
         + dN_dy_partials[:, :, p_index] * (1 - nu) * normal_x[:, np.newaxis] * normal_y[:, np.newaxis])
    b = (dN_dy_partials[:, :, p_index] * (nu * normal_x[:, np.newaxis] ** 2 + normal_y[:, np.newaxis] ** 2)
         + dN_dx_partials[:, :, p_index] * (1 - nu) * normal_x[:, np.newaxis] * normal_y[:, np.newaxis])
    D_gauss_all = np.zeros((a.shape[0], 2 * a.shape[1]), dtype=a.dtype)
    D_gauss_all[:, ::2] = a
    D_gauss_all[:, 1::2] = b
    for arg_dof in range(dN_dx_partials.shape[1]*2):
        for arg_ele,arg_gp in enumerate(range(0,n_gp,n)):
            G[arg_ele, arg_dof] = np.sum(D_gauss_all[arg_gp:(arg_gp+n), arg_dof] * gauss_weight[arg_gp:(arg_gp+n)])
    return G*E

def G_BC_inner_builder_deriv(dN_dx_partials, dN_dy_partials, dN_dx_partials_da, dN_dy_partials_da, normal_x, normal_y, normal_x_da, normal_y_da, gauss_weight, gauss_weight_deriv, n):
    p_index = 1
    nu,E=0.3,1
    n_gp=len(normal_x)
    n_ele=n_gp//n
    G_deriv = np.zeros((dN_dx_partials_da.shape[0], n_ele, dN_dx_partials.shape[1] * 2))
    for alpha_index in range(dN_dx_partials_da.shape[0]):
        # a is Nx displacements b is Ny displacement
        a = (dN_dx_partials[:, :, p_index] * (normal_x[:, np.newaxis] ** 2 + nu * normal_y[:, np.newaxis] ** 2)
             + dN_dy_partials[:, :, p_index] * (1 - nu) * normal_x[:, np.newaxis] * normal_y[:, np.newaxis])
        b = (dN_dy_partials[:, :, p_index] * (nu * normal_x[:, np.newaxis] ** 2 + normal_y[:, np.newaxis] ** 2)
             + dN_dx_partials[:, :, p_index] * (1 - nu) * normal_x[:, np.newaxis] * normal_y[:, np.newaxis])

        da_1_1 = dN_dx_partials_da[alpha_index,:, :, p_index] * (normal_x[:, np.newaxis] ** 2 + nu * normal_y[:, np.newaxis] ** 2)
        da_1_2 = dN_dx_partials[:, :, p_index]*(2*normal_x[:, np.newaxis]*normal_x_da[alpha_index,:, np.newaxis]+nu*2*normal_y[:, np.newaxis]*normal_y_da[alpha_index,:, np.newaxis])
        da_2_1 = dN_dy_partials_da[alpha_index,:, :, p_index] * (1 - nu) * normal_x[:, np.newaxis] * normal_y[:, np.newaxis]
        da_2_2 = dN_dy_partials[:, :, p_index] * (1 - nu) * normal_x_da[alpha_index,:, np.newaxis] * normal_y[:, np.newaxis]
        da_2_3 = dN_dy_partials[:, :, p_index] * (1 - nu) * normal_x[:, np.newaxis] * normal_y_da[alpha_index, :, np.newaxis]
        da=da_1_1+da_1_2+da_2_1+da_2_2+da_2_3

        db_1_1 = dN_dy_partials_da[alpha_index,:, :, p_index] * (nu * normal_x[:, np.newaxis] ** 2 + normal_y[:, np.newaxis] ** 2)
        db_1_2 = dN_dy_partials[:, :, p_index]*(nu*2*normal_x[:, np.newaxis]*normal_x_da[alpha_index,:, np.newaxis]+2*normal_y[:, np.newaxis]*normal_y_da[alpha_index,:, np.newaxis])
        db_2_1 = dN_dx_partials_da[alpha_index,:, :, p_index] * (1 - nu) * normal_x[:, np.newaxis] * normal_y[:, np.newaxis]
        db_2_2 = dN_dx_partials[:, :, p_index] * (1 - nu) * normal_x_da[alpha_index,:, np.newaxis] * normal_y[:, np.newaxis]
        db_2_3 = dN_dx_partials[:, :, p_index] * (1 - nu) * normal_x[:, np.newaxis] * normal_y_da[alpha_index, :, np.newaxis]
        db=db_1_1+db_1_2+db_2_1+db_2_2+db_2_3

        D_gauss_all = np.zeros((a.shape[0], 2 * a.shape[1]), dtype=a.dtype)
        D_gauss_all[:, ::2] = a
        D_gauss_all[:, 1::2] = b
        dD_gauss_all = np.zeros((a.shape[0], 2 * a.shape[1]), dtype=a.dtype)
        dD_gauss_all[:, ::2] = da
        dD_gauss_all[:, 1::2] = db
        for arg_dof in range(dN_dx_partials.shape[1]*2):
            for arg_ele,arg_gp in enumerate(range(0,n_gp,n)):
                G_deriv[alpha_index,arg_ele, arg_dof] = np.sum(dD_gauss_all[arg_gp:(arg_gp+n), arg_dof] * gauss_weight[arg_gp:(arg_gp+n)]+
                                                         D_gauss_all[arg_gp:(arg_gp+n), arg_dof] * gauss_weight_deriv[alpha_index,arg_gp:(arg_gp+n)])
    return G_deriv*E


@njit(cache=True, fastmath=True)
def dK_dalpha_inner_builder(dN_dx, dN_dy, ddN_dax, ddN_day, detJ, d_detJ_d_alphas, D_with_t):
    n = dN_dx.shape[0]
    m = len(detJ)

    B = np.zeros((3, 2 * n, m))
    dB = np.zeros_like(B)

    B[0, ::2, :] = dN_dx
    B[1, 1::2, :] = dN_dy
    B[2, ::2, :] = dN_dy
    B[2, 1::2, :] = dN_dx

    dB[0, ::2, :] = ddN_dax
    dB[1, 1::2, :] = ddN_day
    dB[2, ::2, :] = ddN_day
    dB[2, 1::2, :] = ddN_dax

    K = np.zeros((2 * n, 2 * n))

    BtD = np.zeros((2 * n, 3))
    dBtD = np.zeros_like(BtD)
    D_with_t = np.ascontiguousarray(D_with_t)
    for gp in range(m):
        if detJ[gp] != 0:
            # Compute B' * D and dB' * D once for each gp
            BT_Forder = np.asfortranarray(B[:, :, gp].T)
            np.dot(BT_Forder, D_with_t, out=BtD)

            # Ensure dB is Fortran-contiguous for transpose
            dB_T_Forder = np.asfortranarray(dB[:, :, gp].T)
            np.dot(dB_T_Forder, D_with_t, out=dBtD)

            B_Corder = np.ascontiguousarray(B[:, :, gp])
            step_A = np.dot(dBtD, B_Corder) * detJ[gp]

            # Ensure dB is C-contiguous
            dB_Corder = np.ascontiguousarray(dB[:, :, gp])
            step_B = np.dot(BtD, dB_Corder) * detJ[gp]
            step_C = np.dot(BtD, B_Corder) * d_detJ_d_alphas[gp]

            K += step_C + step_A + step_B

    return K


@njit(cache=True, fastmath=True)
def K_mat_inner_builder(dN_dx, dN_dy, detJ, D_with_t):
    n = dN_dx.shape[0]
    m = len(detJ)

    # Create B and K matrices
    B = np.zeros((3, 2 * n, m))
    K = np.zeros((2 * n, 2 * n))

    # Fill B matrix
    B[0, ::2, :] = dN_dx
    B[1, 1::2, :] = dN_dy
    B[2, ::2, :] = dN_dy
    B[2, 1::2, :] = dN_dx

    # Initialize intermediary result
    BtD = np.zeros((2 * n, 3))

    for gp in range(m):
        if detJ[gp] != 0:
            # Convert B[:, :, gp].T to F-order
            BT_Forder = np.asfortranarray(B[:, :, gp].T)
            # Compute B' * D once for each gp
            np.dot(BT_Forder, D_with_t, out=BtD)
            # Update K with the intermediary result
            K += detJ[gp] * np.dot(BtD, np.ascontiguousarray(B[:, :, gp]))

    return K

class CantileverBeam:
    def __init__(self, Settings_nested_dict, alphas = None,also_solve=False):
        #Import Settings

        self.gamma_BC = None #If spesified in settings, this gets another value

        for dict_name, Settings_spessific_dict in Settings_nested_dict.items():
            if dict_name != "Geometric_settings_rectangle":
                for settings_parameter_name, value in Settings_spessific_dict.items():
                    setattr(self, settings_parameter_name, value)

        if alphas is None: #From settings if not spesified
            alphas = self.y_spline_pos

        self.n_alphas=len(alphas)

        self.itteration_number = 0
        self.Area_of_each_basis_fun,self.dArea_dalpha = None,None
        self.Spline_basis_vals_at_nodes = None
        self.alphas_that_was_last_formulated = None
        self.alphas_that_was_last_solved = None


        self._stiffness_intialisation()

        self.formulate_problem(alphas)
        if also_solve:
            self.solve()


    def update_alphas(self,alphas,also_solve=False,also_grad=False):
        self.formulate_problem(alphas,also_grad)
        if also_solve:
            self.solve(also_grad=also_grad)

    def _stiffness_intialisation(self):
        self.D = self.E / (1 - self.nu ** 2) * np.array([[1, self.nu, 0],
                                                         [self.nu, 1, 0],
                                                         [0, 0, (1 - self.nu) / 2]])
        self.D_with_t = self.thickness*self.D
        self.gauss_weights = np.array([1, 1, 1, 1])
        self.gauss_points = np.array([(-1 / np.sqrt(3), -1 / np.sqrt(3)),
                        (-1 / np.sqrt(3), 1 / np.sqrt(3)),
                        (1 / np.sqrt(3), -1 / np.sqrt(3)),
                        (1 / np.sqrt(3), 1 / np.sqrt(3))])

        self.gauss_weights_flat = np.array([1, 1])
        self.gauss_points_flat = np.array((-1 / np.sqrt(3), 1 / np.sqrt(3)))
        self.gauss_points_per_element = 4
        self.gauss_points_per_slice = 2
        self.detJ_area_off_grid_cell = self.length / self.nx * self.height / self.ny

    def formulate_problem(self,optimization_alphas,also_grad=False):
        if 1:#(not np.all(optimization_alphas == self.alphas_that_was_last_formulated)):

            #self.alphas_that_was_last_formulated
            self._generate_mesh(optimization_alphas)
            self._generate_elements()
            self._prepare_gauss_points()
            self._generate_elements_categories()

            self._make_K_global(also_grad=also_grad)
            self._apply_BC()
            self._make_F_global()
            self._find_dArea_dalpha()
            self.alphas_that_was_last_formulated = optimization_alphas




    def find_area_quick(self,optimization_alphas): #!NB updates the object only sometimes
        if self.Area_of_each_basis_fun is None:
            self.formulate_problem(optimization_alphas)

        return np.dot(self.Area_of_each_basis_fun,optimization_alphas)

    def find_dArea_alpha_quick(self,optimization_alphas): #!NB updates the object only sometimes
        if self.dArea_dalpha is None:
            self.formulate_problem(optimization_alphas)

        return self.dArea_dalpha

    def _generate_elements_categories(self):

        self.completly_active_elements=np.zeros((self.nx+1)*(self.ny+1),dtype=int)
        self.partal_elements=np.zeros((self.nx+1)*(self.ny+1),dtype=int)
        self.all_active_elements=np.zeros((self.nx+1)*(self.ny+1),dtype=int)
        gauss_weight_standard_mat = np.zeros((self.nx, self.ny))
        for x_arg in range(self.nx):
            y_top=self.topSpline.evaluate(self.u_vals_for_top_splines[x_arg:x_arg + 2], input_is_u=True)
            y_top_max = np.max(y_top)
            y_top_min = np.min(y_top)
            for y_arg in range(self.ny):

                element_top = self.dy * (y_arg + 1)
                element_bot = self.dy * (y_arg)
                element_left = self.dx * (x_arg)
                element_right = self.dx * (x_arg + 1)

                arg = y_arg * (self.nx+1) + x_arg
                if (element_top+1e-6)<=(y_top_min): #Completly used
                    self.completly_active_elements[arg] = 1
                    self.partal_elements[arg] = 0
                    self.all_active_elements[arg] = 1
                    gauss_weight_standard_mat[x_arg,y_arg] = 0.25 #In normal FEM integration is from -1 to 1 (area 4)

                elif (element_bot+1e-6) <= y_top_max and (element_top+1e-6) >= y_top_min: #Partial
                    self.completly_active_elements[arg] = 0
                    self.partal_elements[arg] = 1
                    self.all_active_elements[arg] = 1
                    gauss_weight_standard_mat[x_arg, y_arg] = 0
                    self._generate_gauss_points_single_parial_element(element_left,element_right,element_bot,element_top)

                else:  #Empty
                    self.completly_active_elements[arg] = 0
                    self.partal_elements[arg] = 0
                    self.all_active_elements[arg] = 0
                    if self.basis_removal:
                        gauss_weight_standard_mat[x_arg, y_arg] = 0
                    else:
                        gauss_weight_standard_mat[x_arg, y_arg] = 1e-8

        self.gauss_weight_standard = np.vstack((gauss_weight_standard_mat, gauss_weight_standard_mat)).flatten('F').repeat(2)

        self._last_step_dense_gauss_preperation()

    def get_index_from_ij(self,i, j, dof):
        return j * (self.nx + 1) * 2 + i * 2 + dof

    def _last_step_dense_gauss_preperation(self):
        self.u_pos_gauss_dense_partial = self.u_pos_gauss_dense_partial[:self.current_partial_arg]
        self.v_pos_gauss_dense_partial = self.v_pos_gauss_dense_partial[:self.current_partial_arg]
        self.gauss_weight_dense_partial = self.gauss_weight_dense_partial[:self.current_partial_arg]

        self.u_pos_gauss_dense_filled = self.u_pos_gauss_dense_filled[:self.current_filled_arg]
        self.v_pos_gauss_dense_filled = self.v_pos_gauss_dense_filled[:self.current_filled_arg]
        self.gauss_weight_dense_filled = np.full(self.current_filled_arg,self.gauss_weight_dense_filled_single)

        #plt.scatter(self.u_pos_gauss_partial,self.v_pos_gauss_partial)

        #plt.show()

    def plot_gauss_points(self):
        # Grid-like plotting
        for i, x in enumerate(self.x_pos_gauss_standard):
            for j, y in enumerate(self.y_pos_gauss_standard):
                # Get the activity from the flattened array
                x_element_arg = int(x // self.dx)
                y_element_arg = int(y // self.dy)
                active = self.completly_active_elements[x_element_arg+y_element_arg*(self.nx+1)]
                if active == 1:
                    plt.scatter(x, y, c='green', marker='x')
                else:
                    plt.scatter(x, y, c='grey', marker='x')

        x_pos_gauss_dense = self.u_pos_gauss_dense*self.length
        y_pos_gauss_dense = self.v_pos_gauss_dense*self.height

        # Scatter plot for the non-grid-like points
        plt.scatter(x_pos_gauss_dense, y_pos_gauss_dense, c='blue', marker='x')
        x = np.linspace(np.min(self.x_mesh), np.max(self.x_mesh), 20)
        plt.plot(x, self.topSpline.evaluate(x, input_is_u=False))
        # Show the plot
        plt.show()


    def _generate_gauss_points_single_parial_element(self,x_l,x_r,y_b,y_t):
        N=self.num_gauss_segments_in_partial
        logical_right_dense_position = np.logical_and((self.x_vals_for_dense_gauss >= x_l) , (self.x_vals_for_dense_gauss <= x_r))
        u_dense = self.all_u_pos_for_dense_gauss[logical_right_dense_position]
        y_uncliped = self.top_pos_for_y_dense[logical_right_dense_position]

        y=np.clip(y_uncliped, y_b, y_t)

        y_valid = y!=y_b
        y_logical_filled = np.logical_and(y_valid,y==y_t)
        y_logical_partial = np.logical_and(y_valid,y!=y_t)

        y_filled = y[y_logical_filled]
        y_partial = y[y_logical_partial]

        filled_arg=self.current_filled_arg
        filled_arg_end=self.current_filled_arg+np.sum(y_logical_filled)*self.gauss_points_per_slice

        partial_arg=self.current_partial_arg
        partial_arg_end=self.current_partial_arg+np.sum(y_logical_partial)*self.gauss_points_per_slice

        y_gauss_filled_pos = (np.outer(y_filled - y_b, (self.gauss_points_flat + 1) / 2) + y_b).flatten()
        y_gauss_partial_pos = (np.outer(y_partial - y_b, (self.gauss_points_flat + 1) / 2) + y_b).flatten()

        self.u_pos_gauss_dense_filled[filled_arg:filled_arg_end] = np.repeat(u_dense[y_logical_filled], self.gauss_points_per_slice)
        self.v_pos_gauss_dense_filled[filled_arg:filled_arg_end] = y_gauss_filled_pos/self.height

        self.u_pos_gauss_dense_partial[partial_arg:partial_arg_end] = np.repeat(u_dense[y_logical_partial], self.gauss_points_per_slice)
        self.v_pos_gauss_dense_partial[partial_arg:partial_arg_end] = y_gauss_partial_pos/self.height



        self.gauss_weight_dense_partial[partial_arg:partial_arg_end] = np.repeat(0.5*(self.dx_dense/self.dx*
                                                  (y_partial-y_b)/self.dy), self.gauss_points_per_slice)


        self.gauss_weight_dense_partial_deriv_per_dy = 0.5*(self.dx_dense/self.dx*1/self.dy)

        self.gauss_weight_dense_filled_single = 0.5 * (self.dx_dense / self.dx)



        self.current_filled_arg = filled_arg_end
        self.current_partial_arg = partial_arg_end



    def _generate_mesh(self,ghost_points_y_pos):
        if self.itteration_number==0:
            x = np.linspace(0, self.length, self.nx + 1)
            y = np.linspace(0, self.height, self.ny + 1)

            self.dx = x[1] - x[0]
            self.dy = y[1] - y[0]


            self.X, self.Y = np.meshgrid(x, y)

            self.nodes = np.vstack([self.X.ravel(), self.Y.ravel()]).T

            self.num_nodes = len(x) * len(y)
            self.x_mesh,self.y_mesh = x,y
            self.topSpline = BSplineInterpolator(self.x_spline_pos, ghost_points_y_pos, self.p_order)
            self.u_vals_for_top_splines = self.topSpline.find_u_for_x(x)
            self._generate_all_u_pos_for_dense_gauss()
            self.alphas_to_Y_mat_for_dense_gauss = self.topSpline.find_basis(self.all_u_pos_for_dense_gauss).T

        else:
            self.topSpline.update_y_data(ghost_points_y_pos)

    def _generate_all_u_pos_for_dense_gauss(self):
        N=self.num_gauss_segments_in_partial
        x = np.zeros(N*self.nx)
        for i in range(self.nx):
            x_l,x_r = i*self.dx,(i+1)*self.dx
            x[i*N:(i+1)*N] = x_l + (x_r - x_l) * (np.arange(1, N + 1) - 0.5) / N
        self.x_vals_for_dense_gauss = x
        self.all_u_pos_for_dense_gauss = self.topSpline.find_u_for_x(x)
        self.top_pos_for_y_dense = self.topSpline.evaluate(self.all_u_pos_for_dense_gauss,input_is_u=True)
        self.dx_dense= x[1]-x[0]

        self.gauss_weight_dense_partial_deriv_per_dy = 0.5*(self.dx_dense/self.dx*1/self.dy)

        self.gauss_weight_dense_filled_single = 0.5 * (self.dx_dense / self.dx)

    def _generate_elements(self):
        if self.itteration_number==0:
            elements = []
            for i in range(self.ny):
                for j in range(self.nx):
                    elements.append([
                        i * (self.nx + 1) + j,
                        i * (self.nx + 1) + j + 1,
                        (i + 1) * (self.nx + 1) + j + 1,
                        (i + 1) * (self.nx + 1) + j
                    ])

            self.elements = np.array(elements)

    def _make_dK_dalphas_global(self):
        self.dK_dalphas_global = np.zeros((self.n_alphas, 2 * self.num_nodes, 2 * self.num_nodes))
        if np.sum(self.partal_elements)!=0:
            u = self.u_pos_gauss_dense_partial
            v = self.v_pos_gauss_dense_partial
            gauss_weights = self.gauss_weight_dense_partial

            my_args = np.searchsorted(self.all_u_pos_for_dense_gauss, u)
            self.d_top_pos_dense_d_alphas = self.alphas_to_Y_mat_for_dense_gauss[:,my_args]
            d_detJ_d_alphas = self.detJ_area_off_grid_cell*self.d_top_pos_dense_d_alphas*self.gauss_weight_dense_partial_deriv_per_dy

            ddN_dyx, ddN_dyy = self.flatten(self.Surface.evaluate_basises(u, v,return_double_derivs_in_y=True,input_is_u=True))

            y = v*np.max(self.y_mesh)
            top_from_element_bot=np.mod(self.top_pos_for_y_dense[my_args],self.dy)
            d_mesh_pos_d_top_pos_y = (top_from_element_bot-(self.top_pos_for_y_dense[my_args]-y))/top_from_element_bot
            d_mesh_pos_d_alphas=d_mesh_pos_d_top_pos_y[np.newaxis, :] * self.d_top_pos_dense_d_alphas

            ddN_dax = np.zeros((self.n_alphas,ddN_dyx.shape[0],ddN_dyx.shape[1]))
            ddN_day = np.zeros((self.n_alphas,ddN_dyy.shape[0],ddN_dyy.shape[1]))

            ddN_dax += ddN_dyx[np.newaxis,:,:]*d_mesh_pos_d_alphas[:,np.newaxis,:]
            ddN_day += ddN_dyy[np.newaxis,:,:]*(d_mesh_pos_d_alphas)[:,np.newaxis,:]

            dN_dx, dN_dy = self.flatten(self.Surface.evaluate_basises(u, v, input_is_u=True, return_derivs=True))

            detJ = gauss_weights * self.detJ_area_off_grid_cell
            for alpha_index in range(self.n_alphas):
                self.dK_dalphas_global[alpha_index] = dK_dalpha_inner_builder(dN_dx,dN_dy,ddN_dax[alpha_index],ddN_day[alpha_index],detJ,d_detJ_d_alphas[alpha_index],self.D_with_t)

            if self.gradient_penalty:
                self.dK_dalphas_global += self.K_BC_penalty_deriv

            if len(self.G) != 0:
                self.K_lagrange_deriv = np.array([include_lagrange_in_K(self.dK_dalphas_global[i], self.G_deriv[i]) for i in range(self.n_alphas)])










    def _generate_K_standard(self):
        dN_dx,dN_dy = (
            self.flatten(self.Surface.evaluate_basises(self.u_pos_gauss_standard,self.v_pos_gauss_standard,return_derivs=True,gridlike=True),gridlike=True))

        detJ = self.gauss_weight_standard * self.detJ_area_off_grid_cell
        # Create empty B matrix for all gauss points
        K_standard = K_mat_inner_builder(dN_dx,dN_dy,detJ,self.D_with_t)

        return K_standard

    def _generate_K_partial(self):
        if self.current_partial_arg > 0:
            self.u_pos_gauss_dense = np.append(self.u_pos_gauss_dense_partial, self.u_pos_gauss_dense_filled)
            self.v_pos_gauss_dense = np.append(self.v_pos_gauss_dense_partial, self.v_pos_gauss_dense_filled)
            u = self.u_pos_gauss_dense
            v = self.v_pos_gauss_dense
            gauss_weight= np.append(self.gauss_weight_dense_partial, self.gauss_weight_dense_filled)
            detJ = gauss_weight * self.detJ_area_off_grid_cell

            # Scatter plot
            #plt.scatter(u, v, c=gauss_weight, cmap='viridis')
            #plt.colorbar(label='Gauss Weight')
            #plt.xlabel('u')
            #plt.ylabel('v')
            #plt.title('Scatter Plot of u and v with Gauss Weight Color Coding')
            #plt.show()


            dN_dx,dN_dy = self.flatten(self.Surface.evaluate_basises(u,v,input_is_u=True,return_derivs=True))

            # Create empty B matrix for all gauss points

            K_partial = K_mat_inner_builder(dN_dx,dN_dy,detJ,self.D_with_t)
            #

            return K_partial
        else:
            self.u_pos_gauss_dense = np.array([])
            self.v_pos_gauss_dense = np.array([])
            return np.zeros((2*self.num_nodes,2*self.num_nodes))


    def flatten(self,u_v_in_list,gridlike=False):
        u=u_v_in_list[0]
        v=u_v_in_list[1]
        dim = u.shape
        if gridlike:
            num_gp = dim[0] * dim[1]
            num_shapes = dim[2] * dim[3]  #
            u_out = u.reshape(num_gp, dim[2], dim[3], order='F').reshape(num_gp, num_shapes, order='F').T
            v_out = v.reshape(num_gp, dim[2], dim[3], order='F').reshape(num_gp, num_shapes, order='F').T
        else:
            num_gp,num_shapes = dim[0],dim[1]*dim[2]
            u_out = u.reshape(num_gp, num_shapes, order='F').T
            v_out = v.reshape(num_gp, num_shapes, order='F').T
        return u_out,v_out


    def _prepare_gauss_points(self):


        if self.itteration_number == 0:
            self.Surface=BSplineSurfaceInterpolator(self.x_mesh,self.y_mesh,u_v_degree=[self.mesh_px,self.mesh_py])
            self.basis_contribution_node_pos = (
                self.Surface.evaluate_basises(self.x_mesh, self.y_mesh,input_is_u=False, return_derivs=False,gridlike=True))


            # Get differences between adjacent nodes
            x_diff = np.diff(self.x_mesh)
            y_diff = np.diff(self.y_mesh)


            self.x_pos_gauss_standard = np.column_stack(0.5*(1-self.gauss_points_flat[:,None])*self.x_mesh[:-1]
                                                        +0.5*(1+self.gauss_points_flat[:,None])*self.x_mesh[1:]).ravel()
            self.y_pos_gauss_standard = np.column_stack(0.5*(1-self.gauss_points_flat[:,None])*self.y_mesh[:-1]
                                                        +0.5*(1+self.gauss_points_flat[:,None])*self.y_mesh[1:]).ravel()

            self.u_pos_gauss_standard = self.Surface.find_u_for_x(self.x_pos_gauss_standard)
            self.v_pos_gauss_standard = self.Surface.find_v_for_y(self.y_pos_gauss_standard)
            self.alphas_to_y_pos_gauss_standard = self.topSpline.find_basis(self.u_pos_gauss_standard).T


        self.current_partial_arg = 0
        self.current_filled_arg = 0

        sf = 2
        pre_allocation_size = int(self.num_gauss_segments_in_partial * 2 * self.nx * sf)
        self.u_pos_gauss_dense_filled = np.zeros((pre_allocation_size))
        self.v_pos_gauss_dense_filled = np.zeros((pre_allocation_size))

        self.u_pos_gauss_dense_partial = np.zeros((pre_allocation_size))
        self.v_pos_gauss_dense_partial = np.zeros((pre_allocation_size))
        self.gauss_weight_dense_partial = np.zeros((pre_allocation_size))


    def _make_K_global(self,also_grad=False):

        self.K_global = np.zeros((2 * self.num_nodes, 2 * self.num_nodes))
        self.K_global += self._generate_K_standard()
        self.K_global += self._generate_K_partial()
        self.K_global_untouched=copy.deepcopy(self.K_global)


        K_global_BC, self.G= self._BC_penalty_K_global(also_grad)


        if self.gradient_penalty:
            self.K_global += K_global_BC
            self.K_lagrange = self.K_global

        if len(self.G)!=0:
            self.n_constrains = self.G.shape[0]
            self.K_lagrange = include_lagrange_in_K(self.K_global, self.G)


    def calculate_normals_and_derivatives(self,dy_dx, dy_dx_dalphas):
        # Calculate magnitude
        mag = np.sqrt(dy_dx ** 2 + 1)

        # Calculate normals
        normal_x = -dy_dx / mag
        normal_y = 1 / mag

        # Calculate derivatives of magnitude with respect to alphas
        mag_dalphas = (dy_dx[None, :] / mag[None, :]) * dy_dx_dalphas

        # Calculate derivatives of normals with respect to alphas
        normal_x_dalphas = -(mag[None, :] * dy_dx_dalphas - dy_dx[None, :] * mag_dalphas) / (mag[None, :] ** 2)
        normal_y_dalphas = -mag_dalphas / (mag[None, :] ** 2)

        return normal_x, normal_y, normal_x_dalphas, normal_y_dalphas

    def calculate_segment_length_and_derivative(self,dy_dx, dy_dx_dalphas):
        #NOTE THIS ASSUMES TWO GAUSS PONNTS IN EACH ELEMENT

        # Calculate spline length
        segment_length = np.sqrt(1 + dy_dx ** 2) * self.dx

        # Calculate derivative of spline length with respect to alphas
        segment_length_dalphas = (dy_dx[None, :] * dy_dx_dalphas) * (0.5 * self.dx / np.sqrt(1 + dy_dx ** 2)[None, :])

        return segment_length, segment_length_dalphas

    def _prepare_BC_penalty_top_spline(self):

        temp_u=self.all_u_pos_for_dense_gauss
        n=self.num_gauss_segments_in_partial
        u_pos=temp_u
        self.alphas_to_y_pos_top = self.topSpline.find_basis(u_pos).T
        v_pos=self.topSpline.evaluate(u_pos,input_is_u=True)/self.height

        self.u_pos_top_penalty = u_pos
        self.v_pos_top_penalty = v_pos

        #dy_dx_spline = np.gradient(v_pos * self.height,u_pos * self.length)
        dy_dx = np.sum(self.topSpline.find_basis_deriv(u_pos)*self.topSpline.y_data[np.newaxis,:],axis=-1)
        dy_dx_da = self.topSpline.find_basis_deriv(u_pos).T
        normal_x, normal_y, normal_x_da, normal_y_da = self.calculate_normals_and_derivatives(dy_dx,dy_dx_da)

        spline_length, spline_length_da = self.calculate_segment_length_and_derivative(dy_dx, dy_dx_da)
        spline_length, spline_length_da = spline_length/n, spline_length_da/n
        gauss_weight = spline_length
        self.gauss_weight_top_da = spline_length_da
        self.normal_x_top_da = normal_x_da
        self.normal_y_top_da = normal_y_da
        dN_dx_partials_N_is_grid, dN_dy_partials_N_is_grid = self.Surface.evaluate_basises_n_th_partial(u_pos, v_pos,
                                                                                                        deriv_order=1)
        a,b=dN_dx_partials_N_is_grid, dN_dy_partials_N_is_grid
        dN_dx_partials = a.reshape(a.shape[0], -1, a.shape[-1], order='F')
        dN_dy_partials = b.reshape(b.shape[0], -1, b.shape[-1], order='F')
        return dN_dx_partials, dN_dy_partials, normal_x, normal_y ,gauss_weight,n

    def _prepare_BC_penalty_bot(self):
        n=2
        p=self.Surface.bspline_x.degree
        u_pos=self.u_pos_gauss_standard[:-2]#[:-p*2]#
        v_pos=np.full_like(u_pos, 0)
        dN_dx_partials_N_is_grid, dN_dy_partials_N_is_grid = self.Surface.evaluate_basises_n_th_partial(u_pos,v_pos,deriv_order=1)
        a,b=dN_dx_partials_N_is_grid, dN_dy_partials_N_is_grid
        dN_dx_partials = a.reshape(a.shape[0], -1, a.shape[-1], order='F')
        dN_dy_partials = b.reshape(b.shape[0], -1, b.shape[-1], order='F')
        normal_x = np.full_like(u_pos, 0)
        normal_y = np.full_like(u_pos, -1)
        gauss_weight = np.full_like(u_pos, self.dx)/2
        return dN_dx_partials, dN_dy_partials, normal_x, normal_y,gauss_weight,n

    def _prepare_BC_penalty_right(self):
        n=2
        v_pos_end = self.topSpline.evaluate([1],input_is_u=True)[0]/self.height
        last_element = int(np.ceil(v_pos_end*self.ny))
        v_pos_last_element_end = last_element/self.ny
        v_pos = self.v_pos_gauss_standard[:last_element*2]*v_pos_end/v_pos_last_element_end
        u_pos = np.full_like(v_pos, 1)
        dN_dx_partials_N_is_grid, dN_dy_partials_N_is_grid = self.Surface.evaluate_basises_n_th_partial(u_pos,v_pos,deriv_order=1)
        a,b=dN_dx_partials_N_is_grid, dN_dy_partials_N_is_grid
        dN_dx_partials = a.reshape(a.shape[0], -1, a.shape[-1], order='F')
        dN_dy_partials = b.reshape(b.shape[0], -1, b.shape[-1], order='F')
        normal_x = np.full_like(u_pos, 1)
        normal_y = np.full_like(u_pos, 0)
        gauss_weight = np.full_like(u_pos, self.dy*v_pos_end/v_pos_last_element_end)/n
        return dN_dx_partials, dN_dy_partials, normal_x, normal_y,gauss_weight,n

    def _prepare_BC_penalty_derivs_top(self,dN_dx_partials,dN_dy_partials):
        dN_dx_partials_da = np.zeros((self.n_alphas,dN_dx_partials.shape[0],dN_dx_partials.shape[1],dN_dx_partials.shape[2]))
        dN_dy_partials_da = np.zeros_like(dN_dx_partials_da)
        dN_dy_dy = dN_dy_partials[:,:,1:]
        d_top_pos_d_alphas = self.alphas_to_y_pos_top

        dNy_d_alphas = dN_dy_partials[np.newaxis,:,:,1]*d_top_pos_d_alphas[:,:,np.newaxis]

        dN_dx_partials_normalised=np.zeros_like(dN_dx_partials)
        for i in range(dN_dx_partials.shape[-1]):
            denominator=np.maximum(1e-8, np.abs(dN_dy_partials[:, :, 0])) # avoid 0 divided by 0
            dN_dx_partials_normalised[:, :, i]=dN_dx_partials[:, :, i]/denominator

        dN_dx_partials_da = dNy_d_alphas[:,:,:,np.newaxis] * dN_dx_partials_normalised[np.newaxis, :, :,:]

        for a in range(self.n_alphas):
            for p in range(dN_dy_dy.shape[-1]):
                dN_dy_partials_da[a,:,:,p]=d_top_pos_d_alphas[a,:][:,np.newaxis]*dN_dy_dy[:,:,p]
        return dN_dx_partials_da,dN_dy_partials_da,self.gauss_weight_top_da,self.normal_x_top_da,self.normal_y_top_da

    def _prepare_BC_penalty_derivs_right(self,dN_dx_partials,dN_dy_partials):
        n=2 #THIS is not implemented propperly
        dN_dx_partials_da = np.zeros((self.n_alphas,dN_dx_partials.shape[0],dN_dx_partials.shape[1],dN_dx_partials.shape[2]))
        dN_dy_partials_da = np.zeros_like(dN_dx_partials_da)
        dN_dy_dy = dN_dy_partials[:,:,1:]

        v_pos_end = self.topSpline.evaluate([1],input_is_u=True)[0]/self.height
        last_element = int(np.ceil(v_pos_end*self.ny))
        v_pos_last_element_end = last_element/self.ny
        v_pos = self.v_pos_gauss_standard[:last_element*2]*v_pos_end/v_pos_last_element_end
        u_pos = np.full_like(v_pos, 1)

        d_y_pos_d_alphas = self.topSpline.find_basis(u_pos).T*(v_pos/v_pos_end)
        d_v_pos_end_d_alphas = self.topSpline.find_basis([1]).T/self.height

        dN_dx_partials_normalised=np.zeros_like(dN_dx_partials) #:)
        for p in range(dN_dx_partials.shape[-1]):
            denominator=np.maximum(1e-8, np.abs(dN_dy_partials[:, :, 0]))
            denominator_hacky=fill_values_axis_last(denominator,self.nx+1) #Lhopital problems
            dN_dx_partials_normalised[:, :, p]=dN_dx_partials[:, :, p]/denominator_hacky

        dNy_d_alphas = dN_dy_partials[np.newaxis,:,:,1]*d_y_pos_d_alphas [:,:,np.newaxis]
        dNy_d_alphas_hacky = fill_values_axis_last(dNy_d_alphas, self.nx+1)

        dN_dx_partials_da = dNy_d_alphas_hacky[:,:,:,np.newaxis] * dN_dx_partials_normalised[np.newaxis, :, :,:]

        for a in range(self.n_alphas):
            for p in range(dN_dy_dy.shape[-1]):
                dN_dy_partials_da[a,:,:,p]=d_y_pos_d_alphas[a,:][:,np.newaxis]*dN_dy_dy[:,:,p]
        gauss_weight_da,nxda,nyda = np.zeros((self.n_alphas, len(v_pos))),np.zeros((self.n_alphas, len(v_pos))),np.zeros((self.n_alphas, len(v_pos)))
        gauss_weight_da[-1,:] = (self.dy*d_v_pos_end_d_alphas[-1,:]/v_pos_last_element_end)/n

        return dN_dx_partials_da,dN_dy_partials_da,gauss_weight_da,nxda,nyda




    def _rigid_edge(self):
        n_elements=1
        G = np.zeros((n_elements, 2 * self.num_nodes))
        for i,y_arg in enumerate(range(1,n_elements+1)):
            current_constrain=self.basis_contribution_node_pos[self.nx, y_arg, self.nx, :]-self.basis_contribution_node_pos[self.nx, 0, self.nx, :]
            for j in range(len(current_constrain)):
                G[i,self.get_index_from_ij(self.nx,j,1)]=current_constrain[j]
        return G

    def _rigid_edge2(self):
        n_elements = 1
        G = np.zeros((n_elements,2*self.num_nodes))
        for i,y_arg in enumerate(range(1,n_elements+1)):
            G[i,self.get_index_from_ij(self.nx, 0,1)]=1
            G[i,self.get_index_from_ij(self.nx,y_arg,1)]=-1
        return G

    def _get_spessific_penalties(self,dN_dx_partials, dN_dy_partials, normal_x, normal_y, gauss_weight, n_points_per_ele):
        if self.gradient_penalty:
            K_BC = K_BC_inner_builder(dN_dx_partials, dN_dy_partials, normal_x, normal_y, gauss_weight, n_points_per_ele)
        else:
            K_BC = np.zeros((dN_dx_partials.shape[1] * 2, dN_dx_partials.shape[1] * 2))
        if self.lagrange_penalty:
            G_BC = G_BC_inner_builder(dN_dx_partials, dN_dy_partials, normal_x, normal_y, gauss_weight, n_points_per_ele)
        else:
            G_BC=np.zeros((0,dN_dx_partials.shape[1]*2))
        return K_BC, G_BC

    def _get_spessific_penalties_deriv(self,dN_dx_partials,dN_dy_partials,dN_dx_partials_da,dN_dy_partials_da,normal_x,normal_y,normal_x_da,normal_y_da,gauss_weight,gauss_weight_deriv, n_points_per_ele):
        if self.gradient_penalty:
            K_BC = K_BC_inner_builder_deriv(dN_dx_partials,dN_dy_partials,dN_dx_partials_da,dN_dy_partials_da,normal_x,normal_y,normal_x_da,normal_y_da,gauss_weight,gauss_weight_deriv, n_points_per_ele)
        else:
            K_BC = np.zeros((self.n_alphas,dN_dx_partials.shape[1] * 2, dN_dx_partials.shape[1] * 2))
        if self.lagrange_penalty:
            G_BC = G_BC_inner_builder_deriv(dN_dx_partials,dN_dy_partials,dN_dx_partials_da,dN_dy_partials_da,normal_x,normal_y,normal_x_da,normal_y_da,gauss_weight,gauss_weight_deriv, n_points_per_ele)
        else:
            G_BC=np.zeros((self.n_alphas,0,dN_dx_partials.shape[1]*2))
        return K_BC, G_BC


    def _BC_penalty_K_global(self,also_grads=False):

        ###TOP PART

        dN_dx_partials, dN_dy_partials, normal_x, normal_y, gauss_weight,n_points_per_ele = self._prepare_BC_penalty_top_spline()
        self.normal_x, self.normal_y=normal_x,normal_y
        K_BC_top,G_BC_top = self._get_spessific_penalties(dN_dx_partials, dN_dy_partials, normal_x, normal_y,gauss_weight,n_points_per_ele)

        if also_grads:
            dN_dx_partials_da,dN_dy_partials_da,gauss_weight_da,normal_x_da, normal_y_da = self._prepare_BC_penalty_derivs_top(dN_dx_partials, dN_dy_partials)
            K_BC_top_deriv,G_BC_top_deriv = self._get_spessific_penalties_deriv(dN_dx_partials,dN_dy_partials,dN_dx_partials_da,dN_dy_partials_da,normal_x,normal_y,normal_x_da,normal_y_da,gauss_weight,gauss_weight_da, n_points_per_ele)

        ###BOT PART
        if not self.BC_only_top:
            dN_dx_partials, dN_dy_partials, normal_x, normal_y, gauss_weight,n_points_per_ele = self._prepare_BC_penalty_bot()

            K_BC_bot, G_BC_bot = self._get_spessific_penalties(dN_dx_partials, dN_dy_partials, normal_x, normal_y, gauss_weight, n_points_per_ele)
            G_BC_bot_deriv = np.zeros((self.n_alphas, *G_BC_bot.shape))
            #Does not depend on alpha

        ###Right PART
        if not self.BC_only_top:

            dN_dx_partials, dN_dy_partials, normal_x, normal_y, gauss_weight,n_points_per_ele = self._prepare_BC_penalty_right()
            K_BC_right,G_BC_right = self._get_spessific_penalties(dN_dx_partials, dN_dy_partials, normal_x, normal_y,gauss_weight,n_points_per_ele)

            if also_grads:
                dN_dx_partials_da,dN_dy_partials_da,gauss_weight_da,normal_x_da, normal_y_da = self._prepare_BC_penalty_derivs_right(dN_dx_partials, dN_dy_partials)
                K_BC_right_deriv,G_BC_right_deriv = self._get_spessific_penalties_deriv(dN_dx_partials,dN_dy_partials,dN_dx_partials_da,dN_dy_partials_da,normal_x,normal_y,normal_x_da,normal_y_da,gauss_weight,gauss_weight_da, n_points_per_ele)

        if self.link_right_node_to_next:
            G_stiffen_right_edge = self._rigid_edge()
            G = np.vstack((G_stiffen_right_edge, G_BC_top, G_BC_bot, G_BC_right))
        else:
            G_stiffen_right_edge = np.zeros((0, dN_dx_partials.shape[1] * 2))
            
        G_stiffen_right_edge_deriv = np.zeros((self.n_alphas, *G_stiffen_right_edge.shape))


        if self.BC_only_top:
            G = np.vstack((G_stiffen_right_edge, G_BC_top))  # , G_BC_bot, G_BC_right))
            K_BC_penalty_unscaled = K_BC_top  # +K_BC_bot+K_BC_right
        else:
            G = np.vstack((G_stiffen_right_edge, G_BC_top, G_BC_bot, G_BC_right))
            K_BC_penalty_unscaled = K_BC_top + K_BC_right

        BC_contribution_ratio = 10

        if self.gamma_BC is None:
            ratio_BC = np.sum(np.maximum(K_BC_penalty_unscaled, 0)) / np.sum(np.maximum(self.K_global, 0))
            self.gamma_BC = BC_contribution_ratio/ratio_BC

        self.K_BC_penalty = K_BC_penalty_unscaled * self.gamma_BC

        if also_grads: #Stored and called on in dK assembly
            if self.BC_only_top:
                self.G_deriv = np.concatenate((G_stiffen_right_edge_deriv, G_BC_top_deriv),axis=1)
                self.K_BC_penalty_deriv = K_BC_top_deriv*self.gamma_BC
            else:
                self.G_deriv = np.concatenate((G_stiffen_right_edge_deriv, G_BC_top_deriv, G_BC_bot_deriv, G_BC_right_deriv),axis=1)
                self.K_BC_penalty_deriv = (K_BC_top_deriv + K_BC_right_deriv)*self.gamma_BC
        return self.K_BC_penalty,G


    def _apply_BC(self):
        # Nodes at the left edge are fixed
        y_start=self.topSpline.evaluate([0])
        self.fixed_nodes = np.where(np.logical_and(self.nodes[:, 0] == 0 , self.nodes[:, 1] <= (y_start+self.dy*1)))[0]
        self.fixed_dofs = np.hstack([2 * self.fixed_nodes, 2 * self.fixed_nodes + 1])

        self.free_dofs = np.delete(np.arange(2 * self.num_nodes), self.fixed_dofs)

        if self.basis_removal:
            threshold=self.K_global_untouched[0,0]/2000
            indices_to_keep,indices_to_remove=indices_to_keep_basis_removal(self.K_global_untouched,threshold)
            self.free_dofs=np.intersect1d(indices_to_keep, self.free_dofs)
            self.fixed_dofs=np.union1d(indices_to_remove,self.fixed_dofs)

        #self.real_dofs = [self.get_index_from_ij(self.nx, 0, 1), self.get_index_from_ij(self.nx, 1, 1)]
        # Modified stiffness matrix after applying boundary conditions
        self.K_reduced = self.K_global[np.ix_(self.free_dofs, self.free_dofs)]
        if len(self.G)!=0:
            self.K_lagrange_reduced = np.delete(self.K_lagrange,self.fixed_dofs, axis=0)
            self.K_lagrange_reduced = np.delete(self.K_lagrange_reduced,self.fixed_dofs, axis=1)

    def _make_F_global(self):
        if self.itteration_number==0:
            # Identify the bottom right node
            self.bottom_right_node = self.nx
            self.F = np.zeros(2 * self.num_nodes)

            self.F[self.get_index_from_ij(self.nx, 0, 1)] = self.point_load

            self.F_reduced = self.F[self.free_dofs]
            if len(self.G)!=0:
                self.F_lagrange_reduced = np.hstack([self.F_reduced,np.zeros(self.n_constrains)])

    def _find_compliance(self):
        self.C = np.matmul(self.diplacement_each_dof,self.F.ravel(order='F'))


    def _find_dArea_dalpha(self):
        if self.itteration_number==0:
            self.Area_of_each_basis_fun = self.topSpline.integrate_all_basis_functions()
            self.dArea_dalpha = self.Area_of_each_basis_fun



    def _find_dcompliance_dalpha(self):
        self.dC_dalpha = np.zeros(self.n_alphas)
        for i in range(self.n_alphas):
            if len(self.G) != 0:
                self.dC_dalpha[i] = -np.dot(self.diplacement_each_dof_extended.T,
                                            np.matmul(self.K_lagrange_deriv[i, :, :], self.diplacement_each_dof_extended))
            else:
                self.dC_dalpha[i] = -np.dot(self.diplacement_each_dof.T,
                                            np.matmul(self.dK_dalphas_global[i, :, :], self.diplacement_each_dof))

    def solve(self,also_grad=False):
        diplacement_each_dof = np.zeros(2 * self.num_nodes)


        if len(self.G)!=0:
            out=np.linalg.solve(self.K_lagrange_reduced, self.F_lagrange_reduced)
            diplacement_each_dof[self.free_dofs] = out[: -self.n_constrains]
            self.diplacement_each_dof_extended = np.append(diplacement_each_dof, out[-self.n_constrains:])
        else:
            diplacement_each_dof[self.free_dofs] = np.linalg.solve(self.K_reduced, self.F_reduced)



        self.diplacement_each_dof = diplacement_each_dof
        disp_x_ghost = diplacement_each_dof[::2].reshape(self.nx + 1, self.ny + 1, order='F')
        disp_y_ghost = diplacement_each_dof[1::2].reshape(self.nx + 1, self.ny + 1, order='F')
        self.basis_contribution_node_pos
        self.displacement_x=np.zeros((self.nx + 1,self.ny + 1))
        self.displacement_y=np.zeros((self.nx + 1,self.ny + 1))
        for x in range(self.nx + 1):
            for y in range(self.ny + 1):
                self.displacement_x[x, y] = np.sum(self.basis_contribution_node_pos[x, y, :, :] * disp_x_ghost)
                self.displacement_y[x, y] = np.sum(self.basis_contribution_node_pos[x, y, :, :] * disp_y_ghost)

        # First flatten the matrices using 'F' order (Fortran-style)
        flat_disp_x = self.displacement_x.ravel(order='F')
        flat_disp_y = self.displacement_y.ravel(order='F')

        # Interleave the two flattened arrays
        self.displacements_flattened = np.empty((2 * flat_disp_x.size,), dtype=flat_disp_x.dtype)
        self.displacements_flattened[0::2] = flat_disp_x
        self.displacements_flattened[1::2] = flat_disp_y

        self._find_compliance()

        if also_grad==True:
            self._make_dK_dalphas_global()
            self._find_dcompliance_dalpha()

    def plot_von_mises_quiver(self,title="",title2=""):
        n_pixles_in_each_dim = 100
        n_pixles_between_each_arrow = 5
        n_arrows_in_each_dim = n_pixles_in_each_dim // n_pixles_between_each_arrow
        # Set the same number of ticks for both axes

        u = np.linspace(0, 1, n_pixles_in_each_dim)
        v = np.linspace(0, 1, n_pixles_in_each_dim)
        #u=self.u_pos_gauss_standard
        #v=self.topSpline.evaluate(self.u_pos_gauss_standard,input_is_u=True)/self.height
        x,y=self.length*u,self.height*v

        dN_dx, dN_dy = self.flatten(
            self.Surface.evaluate_basises(u, v, input_is_u=True, return_derivs=True,gridlike=True), gridlike=True)
        #dN_dx_dx, dN_dy_dx = self.flatten(self.Surface.evaluate_basises(u, v,return_double_derivs_in_x=True, input_is_u=True, return_derivs=True,gridlike=True),gridlike=True)
        #dN_dx_dy, dN_dy_dy = self.flatten(self.Surface.evaluate_basises(u, v, return_double_derivs_in_y=True, input_is_u=True, return_derivs=True,gridlike=True), gridlike=True)
        dN_dx, dN_dy = dN_dx.T, dN_dy.T

        #K_x = np.sum(dN_dx_dx.T * self.diplacement_each_dof[np.newaxis, ::2],axis=-1)
        #K_y = np.sum(dN_dy_dy.T * self.diplacement_each_dof[np.newaxis, 1::2], axis=-1)

        epsx = np.sum(dN_dx * self.diplacement_each_dof[np.newaxis, ::2],axis=-1)
        epsy = np.sum(dN_dy * self.diplacement_each_dof[np.newaxis, 1::2],axis=-1)
        gamma_xy = np.sum(dN_dy * self.diplacement_each_dof[np.newaxis, ::2],axis=-1) + np.sum(dN_dx * self.diplacement_each_dof[np.newaxis, 1::2],axis=-1)
        #epsilon_n = epsx * self.normal_x ** 2 + epsy * self.normal_y ** 2 + gamma_xy * self.normal_x*self.normal_y
        sigma = np.matmul(self.D,np.array([epsx,epsy,gamma_xy]) )
        sigma_x = sigma[0]
        sigma_y = sigma[1]
        tau_xy = sigma[2]

        # Calculate von Mises stress

        sigma_vM = np.sqrt((sigma_x - sigma_y) ** 2 + 3 * tau_xy ** 2)
        C = sigma_vM.reshape(n_pixles_in_each_dim, n_pixles_in_each_dim, order='F')
        X, Y = np.meshgrid(x, y)
        X, Y = X.T, Y.T
        y_height = self.topSpline.evaluate(u, input_is_u=True)

        # Update the mask based on y_height
        mask = np.zeros_like(Y, dtype=bool)
        for i in range(len(y_height)):
            mask[i, :] = Y[i, :] > y_height[i]

        mask_inactive = ~mask
        C_masked_temp = np.ma.masked_array(C, mask)
        percentile_99 = 70#np.percentile(C_masked_temp.compressed(), 99)
        C_capped = np.where(C > percentile_99, percentile_99, C)

        C_masked = np.ma.masked_array(C_capped, mask)
        C_inactive = np.ma.masked_array(C_capped, mask_inactive)

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))

        # Define contour levels based on the 99th percentile of the active region
        contour_levels = np.linspace(C_masked.min(), percentile_99, 40)


        cmap_gray = mcolors.LinearSegmentedColormap.from_list(
            'gray_coolwarm',
            [(w * 0.299 + x * 0.587 + y * 0.114, w * 0.299 + x * 0.587 + y * 0.114, w * 0.299 + x * 0.587 + y * 0.114)
             for w, x, y, _ in plt.cm.coolwarm(np.linspace(0, 1, 256))]
        )

        # Entire contour plot for the inactive region using the grayscale colormap
        ax.contourf(X, Y, C_inactive, levels=contour_levels, cmap=cmap_gray)

        # Active region
        contour_active = ax.contourf(X, Y, C_masked, levels=contour_levels, cmap='jet', alpha=0.7)
        # Add colorbar
        cbar = plt.colorbar(contour_active, ax=ax)
        cbar.set_label('von Mises stress')

        ax.set_xlabel(r"$X$-axis")
        ax.set_ylabel(r"$Y$-axis")
        #ax.set_title('Masked Visualization')
        # Normalize the stress vectors to get unit vectors
        s_x = sigma_x.reshape((n_pixles_in_each_dim, n_pixles_in_each_dim), order='F')[::n_pixles_between_each_arrow,::n_pixles_between_each_arrow]
        s_y = sigma_y.reshape((n_pixles_in_each_dim, n_pixles_in_each_dim), order='F')[::n_pixles_between_each_arrow,::n_pixles_between_each_arrow]
        t_xy = tau_xy.reshape((n_pixles_in_each_dim, n_pixles_in_each_dim), order='F')[::n_pixles_between_each_arrow,::n_pixles_between_each_arrow]

        s_1,s_2,th=principal_stresses_and_angle(s_x,s_y,t_xy)
        sigma_1_x, sigma_1_y, sigma_2_x, sigma_2_y = principal_stress_components(s_1,s_2,th)

        x_arrow_flat = sigma_1_x
        y_arrow_flat = sigma_1_y
        # Reshape the stress components from Fortran-style flattened array to 2D arrays
        x_arrow_1_2d = x_arrow_flat#x_arrow_flat.reshape((n_pixles_in_each_dim, n_pixles_in_each_dim), order='F')[::n_pixles_between_each_arrow,::n_pixles_between_each_arrow]
        y_arrow_1_2d = y_arrow_flat#y_arrow_flat.reshape((n_pixles_in_each_dim, n_pixles_in_each_dim), order='F')[::n_pixles_between_each_arrow,::n_pixles_between_each_arrow]

        x_arrow_flat = sigma_2_x
        y_arrow_flat = sigma_2_y
        # Reshape the stress components from Fortran-style flattened array to 2D arrays
        x_arrow_2_2d = x_arrow_flat#x_arrow_flat.reshape((n_pixles_in_each_dim, n_pixles_in_each_dim), order='F')[::n_pixles_between_each_arrow,::n_pixles_between_each_arrow]
        y_arrow_2_2d = y_arrow_flat#y_arrow_flat.reshape((n_pixles_in_each_dim, n_pixles_in_each_dim), order='F')[::n_pixles_between_each_arrow,::n_pixles_between_each_arrow]

        #y_arrow_2_2d=y_arrow_2_2d*self.length/self.height
        # Calculate the magnitude of the stress vectors
        magnitude = np.maximum(np.sqrt(x_arrow_1_2d ** 2 + y_arrow_1_2d ** 2),np.sqrt(x_arrow_2_2d ** 2 + y_arrow_2_2d ** 2))
        unit_x_1_arrow = x_arrow_1_2d / magnitude
        unit_y_1_arrow = y_arrow_1_2d / magnitude
        unit_x_2_arrow = x_arrow_2_2d / magnitude
        unit_y_2_arrow = y_arrow_2_2d / magnitude
        max_arrow_size = 0.05
        # Add arrows to the plot
        #ax.quiver(u,v, unit_x_1_arrow, unit_y_1_arrow, scale=1 / max_arrow_size, color='black', width=0.0025)
        #ax.quiver(u, v, unit_x_2_arrow, unit_y_2_arrow, scale=1 / max_arrow_size, color='red', width=0.0025)
        ax.quiver(X[::n_pixles_between_each_arrow,::n_pixles_between_each_arrow], Y[::n_pixles_between_each_arrow,::n_pixles_between_each_arrow],
                  unit_x_1_arrow, unit_y_1_arrow, scale=1 / max_arrow_size, color='black', width=0.0025)
        ax.quiver(X[::n_pixles_between_each_arrow,::n_pixles_between_each_arrow], Y[::n_pixles_between_each_arrow,::n_pixles_between_each_arrow],
                  unit_x_2_arrow, unit_y_2_arrow, scale=1 / max_arrow_size, color='black', width=0.0025)
        ax.set_title(title)
        # Save the plot in the specified folder with the title as filename
        folder_name = "mixedcollection"
        plt.savefig(f'{folder_name}/{"p"+title2}.png')

        plt.show()


    def plot_deformation(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.3, self.figsize[1]))
        self.scale_factor = 2 * np.sign(
            self.displacements_flattened[self.get_index_from_ij(self.nx, 0, 0)]) * 1 / np.max(
            self.displacements_flattened[self.get_index_from_ij(self.nx, 0, 1)])

        deformed_nodes = self.nodes.copy()
        deformed_nodes[:, 0] += self.scale_factor * self.displacements_flattened[::2]
        deformed_nodes[:, 1] += self.scale_factor * self.displacements_flattened[1::2]

        original_quads = []
        deformed_quads = []
        green_original_quads = []
        green_deformed_quads = []
        inactive_original_quads = []
        inactive_deformed_quads = []

        for arg, elem in enumerate(self.elements):
            # Calculate the correct index for Fortran-style ordering
            arg_new=arg+arg//self.nx
            is_green = self.partal_elements[arg_new]

            is_gray_face = np.logical_not(self.all_active_elements)[arg_new]

            original_polygon = patches.Polygon(self.nodes[elem])
            deformed_polygon = patches.Polygon(deformed_nodes[elem])

            if is_green:
                green_original_quads.append(original_polygon)
                green_deformed_quads.append(deformed_polygon)
            elif is_gray_face:
                inactive_original_quads.append(original_polygon)
                inactive_deformed_quads.append(deformed_polygon)
            else:
                original_quads.append(original_polygon)
                deformed_quads.append(deformed_polygon)
        neon_green_rgb = (0.1, 1.0, 0.1)
        # Create PatchCollection objects
        original_collection = PatchCollection(original_quads, edgecolor='b', facecolor='none')
        deformed_collection = PatchCollection(deformed_quads, edgecolor='r', facecolor='none')
        green_original_collection = PatchCollection(green_original_quads, edgecolor=neon_green_rgb, facecolor='none')
        green_deformed_collection = PatchCollection(green_deformed_quads, edgecolor=neon_green_rgb, facecolor='none')
        inactive_original_collection = PatchCollection(inactive_original_quads, edgecolor='b', facecolor='lightgray')
        inactive_deformed_collection = PatchCollection(inactive_deformed_quads, edgecolor='r', facecolor='lightgray')


        # Add collections to respective axes
        ax1.add_collection(original_collection)
        ax1.add_collection(green_original_collection)
        ax2.add_collection(deformed_collection)
        ax2.add_collection(green_deformed_collection)
        ax1.add_collection(inactive_original_collection)
        ax2.add_collection(inactive_deformed_collection)

        # Set limits and labels for both axes
        for ax in [ax1, ax2]:
            ax.set_xlim([-5, 25])
            ax.set_ylim([-5, 25])
            #ax.set_xlim([-1, 11])
            #ax.set_ylim([-3, 5])
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

        # Custom legend for each axis
        original_patch = patches.Patch(edgecolor='b', label='Original', fill=False)
        deformed_patch = patches.Patch(edgecolor='r', label=r'Deformed $\alpha_{g}=1$', fill=False)
        green_original_patch = patches.Patch(edgecolor=neon_green_rgb, label='Contains Edge', fill=False)
        green_deformed_patch = patches.Patch(edgecolor=neon_green_rgb, label='Contains Edge', fill=False)

        inactive_original_patch = patches.Patch(edgecolor='b', label='Inactive Region', fill=True, facecolor='lightgray')
        inactive_deformed_patch = patches.Patch(edgecolor='r', label='Inactive Region', fill=True, facecolor='lightgray')
        ax1.legend(handles=[original_patch, green_original_patch, inactive_original_patch],loc='lower left', framealpha=1.0)
        ax2.legend(handles=[deformed_patch, green_deformed_patch, inactive_deformed_patch],loc='lower left', framealpha=1.0)
        #ax1.legend(handles=[original_patch, green_original_patch])
        #ax2.legend(handles=[deformed_patch, green_deformed_patch])

        # Plot top spline on both axes
        x = np.linspace(np.min(self.x_mesh), np.max(self.x_mesh), 20)
        y = self.topSpline.evaluate(x, input_is_u=False)
        ax1.plot(x, y, 'k--')  # Black dashed line for the top spline
        #ax2.plot(x, y, 'k--')

        # Set titles for each subplot
        ax1.set_title('Original Structure')
        ax2.set_title('Deformed Structure')

        # Display the figure
        plt.tight_layout()  # Adjusts the plot to ensure everything fits without overlapping
        plt.show()



def objective_func(alphas, beam):
    beam.update_alphas(alphas,also_solve=True)
    print("func", alphas, beam.C)
    return beam.C


def objective_grad(alphas, beam):
    print("grad",alphas)
    beam.update_alphas(alphas,also_solve=True,also_grad=True)
    print("grad is",beam.dC_dalpha)
    return beam.dC_dalpha

def area_constraint(alphas, beam):
    print("AREA",beam.find_area_quick(alphas))
    return 180-beam.find_area_quick(alphas)

def area_constraint_grad(alphas, beam):
    print("AREA deriv", beam.find_dArea_alpha_quick(alphas))
    return -beam.find_dArea_alpha_quick(alphas)

if __name__ == '__main__':
    Settings_nested_dict = Settings  # assuming the Settings function/constructor exists and works as expected
    num_vars = len(Settings_nested_dict["Geometric_settings_spline"]["x_spline_pos"])
    lower_lim,upper_lim = Settings_nested_dict["Geometric_settings_spline"]["y_lower_lim"],Settings_nested_dict["Geometric_settings_spline"]["y_upper_lim"]
    initial_alphas = Settings_nested_dict["Geometric_settings_spline"]["y_spline_pos"]

    lower_lim,upper_lim = Settings_nested_dict["Geometric_settings_spline"]["y_lower_lim"],Settings_nested_dict["Geometric_settings_spline"]["y_upper_lim"]
    bounds = [(lower_lim, upper_lim) for _ in range(num_vars)]

    beam = CantileverBeam(Settings_nested_dict, initial_alphas, also_solve=False)

    cons = {'type': 'ineq', 'fun': area_constraint, 'jac': area_constraint_grad, 'args': (beam,)}
    result = minimize(fun=objective_func, jac=objective_grad, method='trust-constr', x0=initial_alphas,constraints=cons, args=(beam,), bounds=bounds)

