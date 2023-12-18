import numpy as np
import matplotlib.pyplot as plt
from B_spline_class import BSplineInterpolator

import functools
import collections

def hashable(item):
    """
    Convert an item into a hashable representation.
    """
    if isinstance(item, collections.abc.Hashable):
        return item
    elif isinstance(item, np.ndarray):
        return (item.shape, item.dtype, item.tobytes())
    elif isinstance(item, collections.abc.Iterable):
        return tuple(hashable(subitem) for subitem in item)
    elif isinstance(item, dict):
        return tuple((key, hashable(value)) for key, value in sorted(item.items()))
    else:
        return str(item)

def memoize_with_limit(n):
    def decorator(func):
        cache = collections.OrderedDict()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a hashable key from args and kwargs
            key = (tuple(hashable(arg) for arg in args),
                   tuple((k, hashable(v)) for k, v in sorted(kwargs.items())))

            # Check if the result is in cache
            if key in cache:
                cache.move_to_end(key)  # Move the key to the end to show it was recently accessed
                return cache[key]

            # Call the function and store the result
            result = func(*args, **kwargs)
            cache[key] = result

            # If cache exceeds the size limit, remove the oldest item
            if len(cache) > n:
                cache.popitem(last=False)

            return result

        return wrapper

    return decorator

# Example usage
@memoize_with_limit(30)
def unique_wrapper(func, arr, *args, **kwargs):
    # Find unique entries and their indices
    unique_entries, inverse_indices = np.unique(arr, return_inverse=True)

    # Apply the function on unique entries
    unique_results = func(unique_entries, *args, **kwargs)

    # Map the results back to original array
    full_results = unique_results[inverse_indices]

    return full_results

def custom_outer_multiply(u, v, gridlike=False):
    if gridlike:
        return u[:, np.newaxis, :, np.newaxis] * v[np.newaxis, :, np.newaxis, :]
    else:
        return u[:, :, np.newaxis] * v[:, np.newaxis, :]

class BSplineSurfaceInterpolator:
    def __init__(self, x_data, y_data, z_data=None, u_v_degree=None):
        self.x_data, self.y_data = x_data, y_data

        if u_v_degree is not None:
            self.bspline_x = BSplineInterpolator(x_data,degree=u_v_degree[0])
            self.bspline_y = BSplineInterpolator(y_data, degree=u_v_degree[1])
        else:
            self.bspline_x = BSplineInterpolator(x_data, degree=u_v_degree)
            self.bspline_y = BSplineInterpolator(y_data, degree=u_v_degree)

        if z_data is None:
            self.z_data = np.ones((len(x_data),len(y_data)))
        else:
            self.z_data = np.array(z_data)


    def find_u_for_x(self, x_target):
        return self.bspline_x.find_u_for_x(x_target)

    def find_v_for_y(self, y_target):
        return self.bspline_y.find_u_for_x(y_target)


    def update_z_data(self,new_z_data):
        if new_z_data:
            self.z_data = np.array(new_z_data)



    def evaluate_basises_n_th_partial(self, u, v, deriv_order=None, input_is_u=True, gridlike=False, single_mixed_mode=False):

        u = np.array([u]) if np.isscalar(u) else np.array(u)
        v = np.array([v]) if np.isscalar(v) else np.array(v)

        # Check if the lengths of u and v are the same
        if len(u) != len(v) and not gridlike:
            raise ValueError("Lengths of u and v must be the same")

        N_x_deriv = unique_wrapper(self.bspline_x.first_nth_derivs_each_basis, u, input_is_u=input_is_u, deriv_order=deriv_order)
        N_y_deriv = unique_wrapper(self.bspline_y.first_nth_derivs_each_basis, v, input_is_u=input_is_u, deriv_order=deriv_order)

        dims_x = N_x_deriv.shape
        dims_y = N_y_deriv.shape

        p_x = N_x_deriv.shape[-1]
        p_y = N_y_deriv.shape[-1]

        max_p=np.max([p_x,p_y])
        dN_dx = np.zeros((dims_x[0], dims_x[1],dims_y[1], max_p))
        dN_dy = np.zeros((dims_y[0], dims_x[1],dims_y[1], max_p))

        if gridlike:
            raise "not implemented"

        for p in range(p_x):
            dN_dx[:,:,:,p] = N_x_deriv[:,:, np.newaxis,p] * N_y_deriv[:, np.newaxis,:,0]
        for p in range(p_y):
            dN_dy[:,:,:,p] = N_x_deriv[:,:, np.newaxis,0] * N_y_deriv[:,np.newaxis,:,p]
        if single_mixed_mode:
            dN_dx_single_mixed=np.zeros_like(dN_dx)
            dN_dy_single_mixed=np.zeros_like(dN_dy)
            for p in range(p_x):
                dN_dx_single_mixed[:, :, :, p] = N_x_deriv[:, :, np.newaxis, p] * N_y_deriv[:, np.newaxis, :, 1]
            for p in range(p_y):
                dN_dy_single_mixed[:, :, :, p] = N_x_deriv[:, :, np.newaxis, 1] * N_y_deriv[:, np.newaxis, :, p]

            dN_dx, dN_dy,dN_dx_single_mixed,dN_dy_single_mixed
        else:
            return dN_dx, dN_dy


    def evaluate_basises(self, u, v, return_derivs=False, return_double_derivs_in_y=False,
                         return_double_derivs_in_x=False, input_is_u=True, get_x_not_y=False, gridlike=False):

        u = np.array([u]) if np.isscalar(u) else np.array(u)
        v = np.array([v]) if np.isscalar(v) else np.array(v)

        # Check if the lengths of u and v are the same
        if len(u) != len(v) and not gridlike:
            raise ValueError("Lengths of u and v must be the same")

        eval_in_u = unique_wrapper(self.bspline_x.evaluate_from_each_basis, u, input_is_u=input_is_u, get_x_not_y=get_x_not_y)
        eval_in_v = unique_wrapper(self.bspline_y.evaluate_from_each_basis, v, input_is_u=input_is_u, get_x_not_y=get_x_not_y)

        if return_derivs:
            eval_in_u_deriv = unique_wrapper(self.bspline_x.evaluate_from_each_basis_deriv, u, input_is_u=input_is_u)
            eval_in_v_deriv = unique_wrapper(self.bspline_y.evaluate_from_each_basis_deriv, v, input_is_u=input_is_u)

            dN_dx = custom_outer_multiply(eval_in_u_deriv, eval_in_v, gridlike)
            dN_dy = custom_outer_multiply(eval_in_u, eval_in_v_deriv, gridlike)

            return dN_dx, dN_dy

        if return_double_derivs_in_y:
            eval_in_u_deriv = unique_wrapper(self.bspline_x.evaluate_from_each_basis_deriv, u, input_is_u=input_is_u, deriv_order=1)
            eval_in_v_deriv = unique_wrapper(self.bspline_y.evaluate_from_each_basis_deriv, v, input_is_u=input_is_u, deriv_order=1)
            eval_in_v_double_deriv = unique_wrapper(self.bspline_y.evaluate_from_each_basis_deriv, v, input_is_u=input_is_u, deriv_order=2)

            ddN_dyx = custom_outer_multiply(eval_in_u_deriv, eval_in_v_deriv, gridlike)
            ddN_dyy = custom_outer_multiply(eval_in_u, eval_in_v_double_deriv, gridlike)

            return ddN_dyx, ddN_dyy

        if return_double_derivs_in_x:
            eval_in_u_deriv = unique_wrapper(self.bspline_x.evaluate_from_each_basis_deriv, u, input_is_u=input_is_u, deriv_order=1)
            eval_in_u_double_deriv = unique_wrapper(self.bspline_x.evaluate_from_each_basis_deriv, u, input_is_u=input_is_u, deriv_order=2)
            eval_in_v_deriv = unique_wrapper(self.bspline_y.evaluate_from_each_basis_deriv, v, input_is_u=input_is_u, deriv_order=1)

            ddN_dxx = custom_outer_multiply(eval_in_u_double_deriv, eval_in_v, gridlike)
            ddN_dxy = custom_outer_multiply(eval_in_u_deriv, eval_in_v_deriv, gridlike)

            return ddN_dxx, ddN_dxy

        N = custom_outer_multiply(eval_in_u, eval_in_v, gridlike)

        return N

    def plot_basis_functions(self):
        self.bspline_x.plot_basis_functions()
        self.bspline_y.plot_basis_functions()


    def integrate_basis_function(self, basis_idx_u, basis_idx_v, num_points=501):
        int_x = self.bspline_x.integrate_basis_function(basis_idx_u, num_points)
        int_y = self.bspline_y.integrate_basis_function(basis_idx_v, num_points)
        return int_x*int_y

    def integrate_all_basis_functions(self, num_points=501):
        areas_x = self.bspline_x.integrate_all_basis_functions(num_points)
        areas_y = self.bspline_y.integrate_all_basis_functions(num_points)
        return np.outer(areas_x,areas_y)

    def plot_spline(self):
        # Plotting the spline for the x and y directions
        self.bspline_x.plot_spline()
        self.bspline_y.plot_spline()

    def plot_surface(self):
        surf_array = self.evaluate(np.linspace(0, 1, 500), np.linspace(0, 1, 500),self.z_data)
        x = np.arange(surf_array.shape[1])
        y = np.arange(surf_array.shape[0])
        x, y = np.meshgrid(x, y)

        # Create the 3D surface plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(x, y, surf_array, cmap='viridis')
        fig.colorbar(surf, ax=ax, label='Value')
        ax.set_title
        plt.show()

if __name__ == "__main__":
    x_data = np.linspace(0, 1, 4)
    y_data = np.linspace(0, 2, 15)
    #z_data = np.random.rand(len(x_data), len(y_data))*0.1  # Random Z values for testing


    interpolator = BSplineSurfaceInterpolator(x_data, y_data)
    x_test, y_test =0.5,0.5
    u_test, v_test = x_test/2, y_test/2
    _, ddN_dyy = interpolator.evaluate_basises(u_test, v_test, return_double_derivs_in_y=True)
