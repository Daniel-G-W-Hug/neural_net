
#include "activation_func.hpp"

#include <cmath>

// activation functions
double identity(double x, f_tag tag) {

  switch (tag) {

  case f_tag::f:
    return x;
    break;

  case f_tag::f1:
    return 1.0;
    break;
  }
}

double sigmoid(double x, f_tag tag) {

  switch (tag) {

  case f_tag::f:

    return 1.0 / (1.0 + exp(-x));
    break;

  case f_tag::f1:

    double val = 1.0 / (1.0 + exp(-x));
    return val * (1.0 - val);
    break;
  }
}

double tanhyp(double x, f_tag tag) {

  switch (tag) {

  case f_tag::f:

    return 2.0 / (1.0 + exp(-2.0 * x)) - 1.0;
    break;

  case f_tag::f1:

    double val = 2.0 / (1.0 + exp(-2.0 * x)) - 1.0;
    return 1.0 - val * val;
    break;
  }
}

double reLU(double x, f_tag tag) {

  switch (tag) {

  case f_tag::f:
    if (x < 0.0) {
      return 0.0;
    } else {
      return x;
    }
    break;

  case f_tag::f1:
    if (x < 0.0) {
      return 0.0;
    } else {
      return 1.0;
    }
    break;
  }
}

double leaky_reLU(double x, f_tag tag) {

  switch (tag) {

  case f_tag::f:
    if (x < 0.0) {
      return 0.01 * x;
    } else {
      return x;
    }
    break;

  case f_tag::f1:
    if (x < 0.0) {
      return 0.01;
    } else {
      return 1.0;
    }
    break;
  }
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