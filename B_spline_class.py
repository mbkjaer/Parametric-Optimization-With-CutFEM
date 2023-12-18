import numpy as np
import matplotlib.pyplot as plt
from geomdl import BSpline
from geomdl import utilities
#from geomdl.helpers import basis_function,basis_function_ders,find_span_linear
from scipy.optimize import root_scalar
from time import time
from numba import njit

def find_span_linear(degree, knot_vector, num_ctrlpts, knot):
    span = degree + 1  # Knot span index starts from zero
    while span < num_ctrlpts and knot_vector[span] <= knot:
        span += 1
    return span - 1

@njit(cache=True)
def basis_function_ders(degree, knot_vector, span, knot, order):
    # Initialize variables
    left = np.ones(degree + 1)
    right = np.ones(degree + 1)
    ndu = np.ones((degree + 1, degree + 1))  # N[0][0] = 1.0 by definition

    for j in range(1, degree + 1):
        left[j] = knot - knot_vector[span + 1 - j]
        right[j] = knot_vector[span + j] - knot
        saved = 0.0
        for r in range(j):
            # Lower triangle
            ndu[j, r] = right[r + 1] + left[j - r]
            temp = ndu[r, j - 1] / ndu[j, r]
            # Upper triangle
            ndu[r, j] = saved + (right[r + 1] * temp)
            saved = left[j - r] * temp
        ndu[j, j] = saved

    # Load the basis functions
    ders = np.zeros((min(degree, order) + 1, degree + 1))
    for j in range(degree + 1):
        ders[0, j] = ndu[j, degree]

    # Start calculating derivatives
    a = np.ones((2, degree + 1))
    # Loop over function index
    for r in range(degree + 1):
        # Alternate rows in array a
        s1 = 0
        s2 = 1
        a[0, 0] = 1.0
        # Loop to compute k-th derivative
        for k in range(1, order + 1):
            d = 0.0
            rk = r - k
            pk = degree - k
            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]
            if rk >= -1:
                j1 = 1
            else:
                j1 = -rk
            if (r - 1) <= pk:
                j2 = k - 1
            else:
                j2 = degree - r
            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                d += a[s2, j] * ndu[rk + j, pk]
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]
            ders[k, r] = d

            # Switch rows
            s1, s2 = s2, s1

    # Multiply through by the the correct factors
    r = float(degree)
    for k in range(1, order + 1):
        for j in range(degree + 1):
            ders[k, j] *= r
        r *= (degree - k)

    # Return the basis function derivatives array
    return ders


@njit(cache=True)
def basis_function(degree, knot_vector, span, knot):
    left = np.zeros(degree + 1)
    right = np.zeros(degree + 1)
    N = np.ones(degree + 1)

    for j in range(1, degree + 1):
        left[j] = knot - knot_vector[span + 1 - j]
        right[j] = knot_vector[span + j] - knot
        saved = 0.0
        for r in range(0, j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved
    return N

class BSplineInterpolator:
    def __init__(self, x_data, y_data=None, degree=None):
        self.x_data = np.array(x_data)
        if y_data is None:
            self.y_data = np.ones_like(x_data)
        else:
            assert len(x_data) == len(y_data), "invalid y_data len is"+str(len(y_data))+ "it should match x and be" + str(len(x_data))
            self.y_data = np.array(y_data)
        if degree is None:
            self.degree = len(x_data) - 1
        else:
            self.degree = degree

        differences = np.diff(self.x_data)
        self.is_evenly_spaced = np.allclose(differences, differences[0], atol=1e-6)

        self._setup_curve()

    def _setup_curve(self):
        data_points = np.column_stack((self.x_data, self.y_data))
        self.curve = BSpline.Curve()
        self.curve.degree = self.degree
        self.curve.ctrlpts = data_points.tolist()
        self.curve.knotvector = utilities.generate_knot_vector(self.curve.degree, len(self.curve.ctrlpts))

    def find_u_for_x(self, x_target):
        if self.is_evenly_spaced:
            return x_target / np.max(self.x_data)
        elif np.isscalar(x_target) and (not self.is_evenly_spaced):
            func = lambda u: self.curve.evaluate_single(u)[0] - x_target
            u_proxy = x_target / np.max(self.x_data)
            if self.is_evenly_spaced:
                return u_proxy
            if u_proxy in [0,1]:
                return u_proxy
            else :
                sol = root_scalar(func, x0=u_proxy, bracket=[0, 1])
                return sol.root if sol.converged else None
        else:
            return np.array([self.find_u_for_x(xi) for xi in x_target])

    def evaluate_from_each_basis(self, u_or_x, y_array=None, input_is_u=True, get_x_not_y=False):
        if y_array is None:
            y_array = self.y_data

        # If the input is given in terms of x-values, find the corresponding u-values.
        if not input_is_u:
            u_or_x = self.find_u_for_x(u_or_x)

        # If the input consists of multiple values (an array), recurse and evaluate each value.
        if np.isscalar(u_or_x):
            result = self.find_basis([u_or_x])
        else:
            results = []
            basis_vals_list = self.find_basis(u_or_x)
            for basis_vals in basis_vals_list:
                results.append(basis_vals)
            result = np.array(results)

        return result

    def first_nth_derivs_each_basis(self, u_vals, deriv_order=None, input_is_u=True):
        if deriv_order is None:
            deriv_order = self.degree

        if not self.is_evenly_spaced:
            raise "only works when evenly spaced"

        # If the input is given in terms of x-values, find the corresponding u-values.
        if not input_is_u:
            u_vals = self.find_u_for_x(u_vals)
        if np.min(u_vals)<0 or np.max(u_vals)>1:
            if np.min(u_vals) < -0.0001 or np.max(u_vals) > 1.0001:
                raise "Invalid U"
            else:
                u_vals = np.clip(u_vals,0,1)

        du_dx = self.get_du_dx(u_vals, deriv_order=1)[0]
        dN_dx_all = np.zeros((len(u_vals), len(self.curve.ctrlpts),deriv_order+1))
        dN_du_powers=np.power(du_dx,np.arange(deriv_order+1))
        # dims: 0th-u,1th-
        for u_arg,u in enumerate(u_vals):
            span = find_span_linear(self.curve.degree, self.curve.knotvector, len(self.curve.ctrlpts), u)
            dN_du = np.array(basis_function_ders(self.curve.degree, np.array(self.curve.knotvector), span, u, deriv_order))
            dN_dx_all[u_arg, span - self.curve.degree:span + 1,:] = dN_du.T*dN_du_powers[np.newaxis,:]
        return dN_dx_all

    def evaluate_from_each_basis_deriv(self, u_or_x,deriv_order=1, y_array=None, input_is_u=True):


        if y_array is None:
            y_array = self.y_data

        # If the input is given in terms of x-values, find the corresponding u-values.
        if not input_is_u:
            u_or_x = self.find_u_for_x(u_or_x)
        if np.min(u_or_x)<0 or np.max(u_or_x)>1:
            if np.min(u_or_x) < -0.0001 or np.max(u_or_x) > 1.0001:
                raise "Invalid U"
            else:
                u_or_x = np.clip(u_or_x,0,1)


        # If the input consists of multiple values (an array), recurse and evaluate each value.
        if np.isscalar(u_or_x):
            result = self.find_basis_deriv([u_or_x],deriv_order=deriv_order)
        else:
            results = []
            basis_vals_list = self.find_basis_deriv(u_or_x,deriv_order=deriv_order)
            for basis_vals in basis_vals_list:
                results.append(basis_vals)
            result = np.array(results)

        return result

    def evaluate(self, u_or_x, input_is_u=True, get_x_not_y=False):
        if get_x_not_y:
            index=0
        else:
            index=1
        
        if input_is_u:
            u_val = u_or_x
            if np.isscalar(u_val):
                if u_val is not None:
                    if u_val<0:
                        print("somethings up")

                    return self.curve.evaluate_single(u_val)[index]
                else:
                    return np.nan
            else:
                return np.array([self.evaluate(ui,input_is_u=True,get_x_not_y=get_x_not_y) for ui in u_val])
        else:
            x=u_or_x
            if np.isscalar(x):
                u_val = self.find_u_for_x(x)
                if u_val is not None:
                    return self.curve.evaluate_single(u_val)[index]
                else:
                    return np.nan
            else:
                return np.array([self.evaluate(xi,input_is_u=False,get_x_not_y=get_x_not_y) for xi in x])

    def update_y_data(self, new_y_data):
        assert len(new_y_data) == len(self.x_data), "The length of new y-data must match the original x-data length."
        self.y_data = np.array(new_y_data)
        self._setup_curve()


    def find_basis(self,u_vals):
        n = len(self.curve.ctrlpts)
        basis_vals = np.zeros((len(u_vals), len(self.curve.ctrlpts)))
        for j, u in enumerate(u_vals):
            span = find_span_linear(self.curve.degree, self.curve.knotvector, len(self.curve.ctrlpts), u)
            N = basis_function(self.curve.degree, np.array(self.curve.knotvector), span, u)

            basis_vals[j, span - self.curve.degree:span + 1] = N

        return basis_vals

    def find_basis_deriv(self,u_vals,deriv_order=1):
        if deriv_order > 2:
            print("this basis derive order is not implemented")
        n = len(self.curve.ctrlpts)
        dN_dx_all = np.zeros((len(u_vals), len(self.curve.ctrlpts)))
        if deriv_order == 1:
            du_dx = self.get_du_dx(u_vals,deriv_order=1)
        elif deriv_order == 2:
            du_dx = self.get_du_dx(u_vals, deriv_order=1)
            ddu_ddx = self.get_du_dx(u_vals, deriv_order=2)

        for j, u in enumerate(u_vals):
            span = find_span_linear(self.curve.degree, self.curve.knotvector, len(self.curve.ctrlpts), u)
            if deriv_order == 1:
                dN_du = np.array(basis_function_ders(self.curve.degree, np.array(self.curve.knotvector), span, u,deriv_order)[-1])
                dN_dx = dN_du * du_dx[j]
                dN_dx_all[j, span - self.curve.degree:span + 1] = dN_dx
            elif (deriv_order == 2 and (self.curve.degree > 1)):
                derivs=np.array(basis_function_ders(self.curve.degree, np.array(self.curve.knotvector), span, u, 2))
                dN_du = derivs[-2]
                ddN_ddu = derivs[-1]
                part_1 = ddN_ddu * du_dx[j]**2
                part_2 = dN_du * ddu_ddx[j]
                dN_dx_all[j, span - self.curve.degree:span + 1] = part_1+part_2
            elif deriv_order == 2 and (self.curve.degree == 1):
                dN_dx_all[j, :]=0

        return dN_dx_all

    def plot_basis_functions(self):
        u_vals = np.linspace(0, 1, 500)
        basis_vals=self.find_basis(u_vals)
        curve_points = self.evaluate(u_vals,input_is_u=True)
        curve_points_x = self.evaluate(u_vals, input_is_u=True,get_x_not_y=True)

        plt.figure()
        for i in range(basis_vals.shape[1]):
            plt.plot(curve_points_x, basis_vals[:, i], label=f"N_{i}")
        plt.title("B-spline Basis Functions")
        plt.xlabel("x_pos")
        plt.ylabel("Basis value")
        plt.legend()
        plt.grid(True)
        plt.show()


    def plot_spline(self):
        curve_y = self.evaluate(np.linspace(0, 1, 1000),input_is_u=True)
        curve_x = self.evaluate(np.linspace(0, 1, 1000), input_is_u=True,get_x_not_y=True)
        #curve_x = [pt[0] for pt in curve_points]
        #curve_y = [pt[1] for pt in curve_points]

        ctrl_x = [pt[0] for pt in self.curve.ctrlpts]
        ctrl_y = [pt[1] for pt in self.curve.ctrlpts]

        plt.figure()
        plt.plot(curve_x, curve_y, 'b-', label="B-spline Curve")
        plt.plot(ctrl_x, ctrl_y, 'ro-', label="Control Points")
        plt.title("B-spline Curve and Control Points")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.show()


    def integrate_basis_function(self, basis_idx, num_points):
        u_vals = np.linspace(0, 1, num_points)

        # Extract the values of the basis function for the given index over the parameter values
        basis_vals = self.find_basis(u_vals)
        y_vals = basis_vals[:, basis_idx]

        # Approximate dx/du using finite differences
        x_vals = [self.evaluate(t, get_x_not_y=True) for t in u_vals]
        dx_du = np.gradient(x_vals)

        return np.trapz(y_vals * dx_du, u_vals)*(num_points-1)

    def integrate_all_basis_functions(self, num_points=501):
        num_ctrl_pts = len(self.curve.ctrlpts)
        areas = np.zeros(num_ctrl_pts)
        for i in range(num_ctrl_pts):
            areas[i] = self.integrate_basis_function(i, num_points=num_points)
        return areas

    def numerical_padding(self,u_vals,delta=1e-4):
        u_vals = np.array(u_vals)

        # Determine if start is 0 and end is 1
        start_is_zero = u_vals[0] <= delta
        end_is_one = u_vals[-1] >= 1-delta

        # Calculate the length of the padded array
        n = len(u_vals)
        padded_length = n * 3 - start_is_zero - end_is_one
        padded = np.zeros(padded_length)

        # Calculate indices for original values
        original_values_placement = np.arange(1 - start_is_zero, padded_length - (1 - end_is_one), 3)

        # Fill in the original values
        padded[original_values_placement] = u_vals

        # Determine valid indices for padding
        valid_pre_padding = original_values_placement - 1 >= 0
        valid_post_padding = original_values_placement + 1 < padded_length

        # Fill in the padded values
        padded[original_values_placement[valid_pre_padding] - 1] = u_vals[valid_pre_padding] - delta
        padded[original_values_placement[valid_post_padding] + 1] = u_vals[valid_post_padding] + delta

        return padded, original_values_placement, start_is_zero, end_is_one

    def get_dx_du(self, u_vals,deriv_order=1):
        if self.is_evenly_spaced and deriv_order==1:
            return np.max(self.x_data)
        elif  self.is_evenly_spaced and deriv_order!=1:
            return 0
        print("This code hasnt been tested for a long time")
        padded_u_vals, original_values_placement, start_is_zero, end_is_one = self.numerical_padding(u_vals)
        padded_x_vals = self.evaluate(padded_u_vals,input_is_u=True,get_x_not_y=True)

        dx_du_current_order=padded_x_vals#order 0
        # Compute the gradient using np.gradient
        for i in range(deriv_order): #increase order n times
            dx_du_current_order = np.gradient(dx_du_current_order, padded_u_vals)

        return dx_du[original_values_placement]

    def get_du_dx(self, u_vals,deriv_order=1):
        if self.is_evenly_spaced and deriv_order==1:
            return np.full_like(u_vals,1/np.max(self.x_data))
        elif  self.is_evenly_spaced and deriv_order!=1:
            return np.full_like(u_vals,0)
        print("Hasnt been tested for a long time")
        padded_u_vals, original_values_placement, start_is_zero, end_is_one = self.numerical_padding(u_vals)
        padded_x_vals = self.evaluate(padded_u_vals,input_is_u=True,get_x_not_y=True)

        du_dx_current_order=padded_u_vals#order 0
        # Compute the gradient using np.gradient
        for i in range(deriv_order): #increase order n times
            du_dx_current_order = np.gradient(du_dx_current_order, padded_x_vals)

        return du_dx_current_order[original_values_placement]




# Usage:
if __name__ == "__main__":
    x_data = np.array([0., 1., 2, 3., 4., 5., 6., 7., 8., 9., 10., 11.,12])/12
    y_data = np.array([0., 1., 0., -1., 0., 1., 0., -1., 0., 1., 0., -1.,0])

    a = BSplineInterpolator(x_data, y_data,degree=3)
    a.plot_basis_functions()
    a.first_nth_derivs_each_basis(x_data)


