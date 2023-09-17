#ifndef NEURAL_NET_FUNC_HPP
#define NEURAL_NET_FUNC_HPP

#include <vector>

// activation function (for config files / io)
enum class a_func_t { identity = 1, sigmoid, tanhyp, reLU, leaky_reLU };

// loss function (for config files / io)
enum class l_func_t { mse = 1 };

// signal to return function or first derivative
enum class f_tag { f, f1 };
// used to pass activation functions to nodes
// using ftag::f (default): return f(x)
// using ftag::f1: return f'(x), i.e. the 1st derivative of f(x)

using a_func_ptr_t = std::vector<double> (*)(std::vector<double> const &,
                                             f_tag);
using l_func_ptr_t = double (*)(double, double, f_tag);

// activation functions
std::vector<double> identity(std::vector<double> const &x,
                             f_tag tag = f_tag::f);
std::vector<double> sigmoid(std::vector<double> const &x, f_tag tag = f_tag::f);
std::vector<double> tanhyp(std::vector<double> const &x, f_tag tag = f_tag::f);
std::vector<double> reLU(std::vector<double> const &x, f_tag tag = f_tag::f);
std::vector<double> leaky_reLU(std::vector<double> const &x,
                               f_tag tag = f_tag::f);

// loss functions
double mean_squared_error(double output, double output_target,
                          f_tag tag = f_tag::f);

// pointer assignment functions
a_func_ptr_t get_activation_func_ptr(a_func_t af);
l_func_ptr_t get_loss_func_ptr(l_func_t lf);

#endif // NEURAL_NET_FUNC_HPP