#ifndef ACTIVATION_FUNC_HPP
#define ACTIVATION_FUNC_HPP

// choosen aktivation function (for config files / io)
enum class a_func_t { identity = 1, sigmoid, tanhyp, reLU, leaky_reLU };

// signal to return function or first derivative
enum class f_tag { f, f1 };

using a_func_ptr_t = double (*)(double, f_tag);
// used to pass activation functions to nodes
// using ftag::f (default): return f(x)
// using ftag::f1: return f'(x), i.e. the 1st derivative of f(x)

double identity(double x, f_tag tag = f_tag::f); // for input layer
double sigmoid(double x, f_tag tag = f_tag::f);
double tanhyp(double x, f_tag tag = f_tag::f);
double reLU(double x, f_tag tag = f_tag::f);
double leaky_reLU(double x, f_tag tag = f_tag::f);

a_func_ptr_t get_activation_func_ptr(a_func_t af);

#endif // ACTIVATION_FUNC_HPP