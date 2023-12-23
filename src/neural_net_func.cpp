
#include "neural_net_func.hpp"

#include <algorithm> // std::max_element()
#include <cmath>
#include <cstddef> // std::size_t
#include <iostream>
#include <utility> // std::unreachable()

// activation functions
std::vector<nn_fp_t> identity(std::vector<nn_fp_t> const& x, f_tag tag)
{

    std::size_t dim = x.size();
    std::vector<nn_fp_t> y(dim);

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

std::vector<nn_fp_t> sigmoid(std::vector<nn_fp_t> const& x, f_tag tag)
{

    std::size_t dim = x.size();
    std::vector<nn_fp_t> y(dim);

    switch (tag) {

        case f_tag::f:
            for (std::size_t n = 0; n < dim; ++n) {
                y[n] = 1.0 / (1.0 + exp(-x[n]));
            }
            break;

        case f_tag::f1:

            for (std::size_t n = 0; n < dim; ++n) {
                nn_fp_t val = 1.0 / (1.0 + exp(-x[n]));
                y[n] = val * (1.0 - val);
            }
            break;
    }

    return y;
}

std::vector<nn_fp_t> tanhyp(std::vector<nn_fp_t> const& x, f_tag tag)
{

    std::size_t dim = x.size();
    std::vector<nn_fp_t> y(dim);

    switch (tag) {

        case f_tag::f:

            for (std::size_t n = 0; n < dim; ++n) {
                y[n] = 2.0 / (1.0 + exp(-2.0 * x[n])) - 1.0;
            }
            break;

        case f_tag::f1:

            for (std::size_t n = 0; n < dim; ++n) {
                nn_fp_t val = 2.0 / (1.0 + exp(-2.0 * x[n])) - 1.0;
                y[n] = 1.0 - val * val;
            }
            break;
    }

    return y;
}

std::vector<nn_fp_t> reLU(std::vector<nn_fp_t> const& x, f_tag tag)
{

    std::size_t dim = x.size();
    std::vector<nn_fp_t> y(dim);

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

std::vector<nn_fp_t> leaky_reLU(std::vector<nn_fp_t> const& x, f_tag tag)
{

    std::size_t dim = x.size();
    std::vector<nn_fp_t> y(dim);

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

std::vector<nn_fp_t> softmax(std::vector<nn_fp_t> const& x, f_tag tag)
{

    std::size_t dim = x.size();
    std::vector<nn_fp_t> y(dim);

    // use stable version of softmax by normalizing input elements
    // (only for numeric stability, w/o effect on derivative calculation in loss function)
    nn_fp_t d = -(*std::max_element(x.begin(), x.end()));

    nn_fp_t denom{0.0};

    switch (tag) {

        case f_tag::f:

            for (std::size_t n = 0; n < dim; ++n) {
                denom += std::exp(x[n] + d);
            }

            for (std::size_t n = 0; n < dim; ++n) {
                y[n] = std::exp(x[n] + d) / denom;
            }
            break;

        case f_tag::f1:

            throw std::runtime_error("Failed to calculate derivative of softmax function."
                                     "Hint: Use softmax only for output layer in "
                                     "combination with loss function, not for hidden "
                                     "layers.\n");
            break;
    }

    return y;
}

nn_fp_t MSE(std::vector<nn_fp_t> const& output,
            std::vector<nn_fp_t> const& target_output_vec)
{
    // MSE: Mean Squared Error (with factor 0.5 for simple derivative)
    std::size_t dim = output.size();
    nn_fp_t partial_loss{0.0};
    for (std::size_t to = 0; to < dim; ++to) {
        partial_loss += std::pow(output[to] - target_output_vec[to], 2.0);
    }

    return 0.5 * partial_loss;
}

nn_fp_t CE(std::vector<nn_fp_t> const& output,
           std::vector<nn_fp_t> const& target_output_vec)
{
    // CROSS_ENTROPY (CE): CE = - sum_t(target_output[t] * log(output[t])

    std::size_t dim = output.size();
    nn_fp_t partial_loss{0.0};
    for (std::size_t to = 0; to < dim; ++to) {
        partial_loss -= target_output_vec[to] * std::log(output[to]);
    }

    return partial_loss;
}


std::vector<nn_fp_t> MSE_identity_f1(std::vector<nn_fp_t> const& output,
                                     std::vector<nn_fp_t> const& target_output_vec)
{

    std::size_t dim = output.size();
    std::vector<nn_fp_t> f1(dim);
    for (std::size_t to = 0; to < dim; ++to) {
        f1[to] = output[to] - target_output_vec[to];
    }

    return f1;
}

std::vector<nn_fp_t> CE_softmax_f1(std::vector<nn_fp_t> const& output,
                                   std::vector<nn_fp_t> const& target_output_vec)
{

    std::size_t dim = output.size();
    std::vector<nn_fp_t> f1(dim);
    for (std::size_t to = 0; to < dim; ++to) {
        f1[to] = output[to] - target_output_vec[to];
    }

    return f1;
}

af_ptr_t get_af_ptr(af_t af)
{
    switch (af) {
        case (af_t::identity):
            return &identity;
            break;
        case (af_t::sigmoid):
            return &sigmoid;
            break;
        case (af_t::tanhyp):
            return &tanhyp;
            break;
        case (af_t::reLU):
            return &reLU;
            break;
        case (af_t::leaky_reLU):
            return &leaky_reLU;
            break;
        case (af_t::softmax):
            return &softmax;
            break;
        default:
            std::unreachable();
            break;
    }
    std::unreachable();
}

lf_ptr_t get_lf_ptr(lf_t lf)
{
    switch (lf) {
        case (lf_t::MSE):
            return &MSE;
            break;
        case (lf_t::CE):
            return &CE;
            break;
        default:
            std::unreachable();
            break;
    }
    std::unreachable();
}