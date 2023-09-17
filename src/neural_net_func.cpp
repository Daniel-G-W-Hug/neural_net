
#include "neural_net_func.hpp"

#include <cmath>
#include <cstddef> // std::size_t

#include <iostream>

// activation functions
std::vector<double> identity(std::vector<double> const &x, f_tag tag) {

  std::size_t dim = x.size();
  std::vector<double> y(dim);

  switch (tag) {

  case f_tag::f:
    for (std::size_t n = 0; n < dim; ++n) {
      y[n] = x[n];
    }
    break;

  case f_tag::f1:
    for (std::size_t n = 0; n < dim; ++n) {
      y[n] = 1.0;
    }
    break;
  }

  return y;
}

std::vector<double> sigmoid(std::vector<double> const &x, f_tag tag) {

  std::size_t dim = x.size();
  std::vector<double> y(dim);

  switch (tag) {

  case f_tag::f:
    for (std::size_t n = 0; n < dim; ++n) {
      y[n] = 1.0 / (1.0 + exp(-x[n]));
    }
    break;

  case f_tag::f1:

    for (std::size_t n = 0; n < dim; ++n) {
      double val = 1.0 / (1.0 + exp(-x[n]));
      y[n] = val * (1.0 - val);
    }
    break;
  }

  return y;
}

std::vector<double> tanhyp(std::vector<double> const &x, f_tag tag) {

  std::size_t dim = x.size();
  std::vector<double> y(dim);

  switch (tag) {

  case f_tag::f:

    for (std::size_t n = 0; n < dim; ++n) {
      y[n] = 2.0 / (1.0 + exp(-2.0 * x[n])) - 1.0;
    }
    break;

  case f_tag::f1:

    for (std::size_t n = 0; n < dim; ++n) {
      double val = 2.0 / (1.0 + exp(-2.0 * x[n])) - 1.0;
      y[n] = 1.0 - val * val;
    }
    break;
  }

  return y;
}

std::vector<double> reLU(std::vector<double> const &x, f_tag tag) {

  std::size_t dim = x.size();
  std::vector<double> y(dim);

  switch (tag) {

  case f_tag::f:

    for (std::size_t n = 0; n < dim; ++n) {
      y[n] = (x[n] < 0.0) ? 0.0 : x[n];
    }
    break;

  case f_tag::f1:

    for (std::size_t n = 0; n < dim; ++n) {
      y[n] = (x[n] < 0.0) ? 0.0 : 1.0;
    }
    break;
  }

  return y;
}

std::vector<double> leaky_reLU(std::vector<double> const &x, f_tag tag) {

  std::size_t dim = x.size();
  std::vector<double> y(dim);

  switch (tag) {

  case f_tag::f:

    for (std::size_t n = 0; n < dim; ++n) {
      y[n] = (x[n] < 0.0) ? 0.01 * x[n] : x[n];
    }
    break;

  case f_tag::f1:

    for (std::size_t n = 0; n < dim; ++n) {
      y[n] = (x[n] < 0.0) ? 0.01 : 1.0;
    }
    break;
  }

  return y;
}

double mean_squared_error(double output, double output_target, f_tag tag) {

  double loss{0.0};

  switch (tag) {

  case f_tag::f:

    // scaling factor 0.5 is chosen to simplify the derivative calculation
    loss = 0.5 * std::pow(output - output_target, 2.0);
    break;

  case f_tag::f1:

    loss = output - output_target;
    break;
  }

  return loss;
}

a_func_ptr_t get_activation_func_ptr(a_func_t af) {
  switch (af) {
  case (a_func_t::identity):
    return &identity;
    break;
  case (a_func_t::sigmoid):
    return &sigmoid;
    break;
  case (a_func_t::tanhyp):
    return &tanhyp;
    break;
  case (a_func_t::reLU):
    return &reLU;
    break;
  case (a_func_t::leaky_reLU):
    return &leaky_reLU;
    break;
  }
}

l_func_ptr_t get_loss_func_ptr(l_func_t lf) {
  switch (lf) {
  case (l_func_t::mse):
    return &mean_squared_error;
    break;
  }
}