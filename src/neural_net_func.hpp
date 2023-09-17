#ifndef ACTIVATION_FUNC_HPP
#define ACTIVATION_FUNC_HPP

#include <vector>

// choosen aktivation function (for config files / io)
enum class a_func_t { identity = 1, sigmoid, tanhyp, reLU, leaky_reLU };

// signal to return function or first derivative
enum class f_tag { f, f1 };

using a_func_ptr_t = std::vector<double> (*)(std::vector<double> const &,
                                             f_tag);
// used to pass activation functions to nodes
// using ftag::f (default): return f(x)
// using ftag::f1: return f'(x), i.e. the 1st derivative of f(x)

std::vector<double> identity(std::vector<double> const &x,
                             f_tag tag = f_tag::f);
std::vector<double> sigmoid(std::vector<double> const &x, f_tag tag = f_tag::f);
std::vector<double> tanhyp(std::vector<double> const &x, f_tag tag = f_tag::f);
std::vector<double> reLU(std::vector<double> const &x, f_tag tag = f_tag::f);
std::vector<double> leaky_reLU(std::vector<double> const &x,
                               f_tag tag = f_tag::f);

a_func_ptr_t get_activation_func_ptr(a_func_t af);

#endif // ACTIVATION_FUNC_HPP