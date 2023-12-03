#ifndef NEURAL_NET_FUNC_HPP
#define NEURAL_NET_FUNC_HPP

#include <exception>
#include <vector>

// activation function (for config files / io)
enum class af_t { identity = 1, sigmoid, tanhyp, reLU, leaky_reLU, softmax };

// loss function
enum class lf_t { MSE = 1, CE };

// combination of loss function and activation derivatives (for config files / io)
enum class lfaf_f1_t { MSE_identity = 1, MSE_sigmoid, CE_softmax };

// signal to return function or first derivative
enum class f_tag { f, f1 };
// used to pass activation functions to nodes
// using ftag::f (default): return f(x)
// using ftag::f1: return f'(x), i.e. the 1st derivative of f(x)

using af_ptr_t = std::vector<double> (*)(std::vector<double> const&, f_tag);
using lf_ptr_t = double (*)(std::vector<double> const&, std::vector<double> const&);
using lfaf_f1_ptr_t = std::vector<double> (*)(std::vector<double> const&,
                                              std::vector<double> const&);

// activation functions
std::vector<double> identity(std::vector<double> const& x, f_tag tag = f_tag::f);
std::vector<double> sigmoid(std::vector<double> const& x, f_tag tag = f_tag::f);
std::vector<double> tanhyp(std::vector<double> const& x, f_tag tag = f_tag::f);
std::vector<double> reLU(std::vector<double> const& x, f_tag tag = f_tag::f);
std::vector<double> leaky_reLU(std::vector<double> const& x, f_tag tag = f_tag::f);
// for output layer exclusively
// (f_tag::f1 not implemented, since it is just needed together with loss function)
std::vector<double> softmax(std::vector<double> const& x, f_tag tag = f_tag::f);

// loss functions
double MSE(std::vector<double> const& output,
           std::vector<double> const& target_output_vec);
double CE(std::vector<double> const& output,
          std::vector<double> const& target_output_vec);

// combination of loss function and activation function derivatives
std::vector<double> MSE_identity_f1(std::vector<double> const& output,
                                    std::vector<double> const& target_output_vec);
std::vector<double> CE_softmax_f1(std::vector<double> const& output,
                                  std::vector<double> const& target_output_vec);

// pointer assignment functions
af_ptr_t get_af_ptr(af_t af);
lf_ptr_t get_lf_ptr(lf_t lf);

#endif // NEURAL_NET_FUNC_HPP