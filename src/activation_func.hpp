#ifndef ACTIVATION_FUNC_HPP
#define ACTIVATION_FUNC_HPP

enum class f_tag { f, f1 }; // signal to return function or first derivative

using a_func_ptr_t = double (*)(double, f_tag);
// used to pass activation functions to nodes
// for use in neural_net constructor with &chosen_function

// activation functions
// default: return f(x)
// using ftag::f1: return f'(x), i.e. the 1st derivative of f(x)

double identity(double x, f_tag tag = f_tag::f); // for inp./outp. layers
double reLU(double x, f_tag tag = f_tag::f);
double leaky_reLU(double x, f_tag tag = f_tag::f);
double sigmoid(double x, f_tag tag = f_tag::f);
double tanhyp(double x, f_tag tag = f_tag::f);

#endif // ACTIVATION_FUNC_HPP