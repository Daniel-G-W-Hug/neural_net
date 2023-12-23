#include "neural_net.hpp"

// uncomment to disable assert()
// #define NDEBUG
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric> // std::inner_product
#include <utility> // std::unreachable()

#include "fmt/format.h"
#include "fmt/ranges.h" // support printing of (nested) containers & tuples

std::random_device rd;
std::mt19937 gen(rd());

neural_net::neural_net(nn_structure_t structure_input) : m_structure{structure_input}
{
    // set up neural network structure

    num_layers = m_structure.net_structure.size();
    // the minimum network has an input and an output layer
    // each layer has a positive number of nodes
    assert(num_layers >= 2);
    for (std::size_t l = 0; l < num_layers; ++l) {
        assert(m_structure.net_structure[l] > 0);
    }
    layer.resize(num_layers);

    num_nodes = m_structure.net_structure;

    for (std::size_t l = 0; l < num_layers; ++l) {
        if (l > 0) {
            // no bias values in input layer
            total_num_bias += num_nodes[l];
        }
        total_num_nodes += num_nodes[l];
    }

    // create and intialize nodes in all layers
    // (activation functions and weights)
    for (std::size_t l = 0; l < num_layers; ++l) {

        layer[l].z.resize(num_nodes[l]);
        layer[l].a.resize(num_nodes[l]);

        if (l > 0) {
            // no bias, bias gradient, delta or weights in input layer
            layer[l].b.resize(num_nodes[l]);
            layer[l].dLdb.resize(num_nodes[l]);
            layer[l].delta.resize(num_nodes[l]);
            layer[l].w.resize(num_nodes[l]);
            layer[l].dLdw.resize(num_nodes[l]);

            // initialize weights, biases and gradients

            // normal_disribution(mean,stddev) - for activation bias values
            std::normal_distribution<nn_fp_t> d_ran0(0.0, 1.0);

            // uniform_real_distribution(from, to)

            // // Xavier weight initialization - for activation functions 1,2,3
            // nn_fp_t divisor = std::sqrt(num_nodes[l - 1]);
            // std::uniform_real_distribution<nn_fp_t> d_ran1(-1.0 / divisor, 1.0 /
            // divisor);

            // normalized Xavier weight initialization - for activation functions 1,2,3
            nn_fp_t divisor = std::sqrt(num_nodes[l - 1] + num_nodes[l]);
            std::uniform_real_distribution<nn_fp_t> d_ran1(-std::sqrt(6.0) / divisor,
                                                           std::sqrt(6.0) / divisor);

            // normal_disribution(mean,stddev) - for activation function 4,5 (reLU)
            std::normal_distribution<nn_fp_t> d_ran2(0.0,
                                                     2.0 / std::sqrt(num_nodes[l - 1]));

            std::size_t w_cnt{0};
            for (std::size_t to = 0; to < num_nodes[l]; ++to) {

                layer[l].b[to] = d_ran0(gen);
                layer[l].dLdb[to] = 0.0;

                layer[l].w[to].resize(num_nodes[l - 1]);
                layer[l].dLdw[to].resize(num_nodes[l - 1]);
                for (std::size_t from = 0; from < num_nodes[l - 1]; ++from) {
                    if (m_structure.af_h <= af_t::tanhyp) {
                        // for identity, sigmoid and tanhyp in hidden layers
                        layer[l].w[to][from] = d_ran1(gen);
                    }
                    else {
                        // for reLu variants in hidden layers
                        layer[l].w[to][from] = d_ran2(gen);
                    }
                    layer[l].dLdw[to][from] = 0.0;
                }
                w_cnt += num_nodes[l - 1];
            }
            total_num_weights += w_cnt;
        }

        // initialize activation function for layer
        if (l == 0) {
            // input layer
            layer[l].af = get_af_ptr(af_t::identity);
        }
        else if (l == num_layers - 1) {
            // output layer
            layer[l].af = get_af_ptr(m_structure.af_o);
        }
        else {
            // hidden layers
            layer[l].af = get_af_ptr(m_structure.af_h);
        }
    }

    // set loss function and derivative f1 of combination of loss and activation functions
    lossf = get_lf_ptr(m_structure.lossf);

} // neural_net (ctor)

void neural_net::set_w_and_b_fixed(nn_fp_t val)
{

    for (std::size_t l = 1; l < num_layers; ++l) {
        for (std::size_t to = 0; to < num_nodes[l]; ++to) {
            layer[l].b[to] = val;
            for (std::size_t from = 0; from < num_nodes[l - 1]; ++from) {
                layer[l].w[to][from] = val;
            }
        }
    }
} // set_w_and_b_fixed

void neural_net::forward_pass(std::vector<nn_fp_t> const& input_vec)
{
    // propagate the input data through the network

    // set input layer nodes to user provided values
    std::size_t l = 0; // input layer
    layer[l].z = input_vec;

    // forward pass through inner network layers starts at layer 1
    for (std::size_t l = 1; l < num_layers; ++l) {

        // activate all nodes in previous layer to make output values available
        layer[l - 1].a = layer[l - 1].af(layer[l - 1].z, f_tag::f);

        // calculate summed activation for all nodes (incl. node bias)

        // //// RAW loop
        // for (std::size_t to = 0; to < num_nodes[l]; ++to) {
        //   layer[l].z[to] = layer[l].b[to];
        //   for (std::size_t from = 0; from < num_nodes[l - 1]; ++from) {
        //     layer[l].z[to] += layer[l].w[to][from] * layer[l - 1].a[from];
        //   }
        // }

        // //// with inner_product
        for (std::size_t to = 0; to < num_nodes[l]; ++to) {
            layer[l].z[to] =
                std::inner_product(layer[l].w[to].begin(), layer[l].w[to].end(),
                                   layer[l - 1].a.begin(), layer[l].b[to]);
        }

        // //// with transform_reduce potentially parallelized (but not supported by
        // clang++ yet) for (std::size_t to = 0; to < num_nodes[l]; ++to) {
        //   layer[l].z[to] = std::transform_reduce(
        //       std::execution::par_unseq, layer[l].w[to].begin(), layer[l].w[to].end(),
        //       layer[l - 1].a.begin(), layer[l].b[to]);
        // }
    }

    // activate all nodes in output layer to make output available
    l = num_layers - 1;
    layer[l].a = layer[l].af(layer[l].z, f_tag::f);

} // forward_pass

std::vector<nn_fp_t>
neural_net::forward_pass_with_output(std::vector<nn_fp_t> const& input_vec)
{

    forward_pass(input_vec);

    return layer[num_layers - 1].a;
} // forward_pass_with_output

void neural_net::backward_pass(std::vector<nn_fp_t> const& input_vec,
                               std::vector<nn_fp_t> const& output_target_vec)
{
    // calculate backward pass for contribution to gradient vector based on the
    // given input and target output vectors

    // calculate deltas in output layer
    {
        std::size_t l = num_layers - 1;

        if (m_structure.lossf_af_f1 == lfaf_f1_t::MSE_identity ||
            m_structure.lossf_af_f1 == lfaf_f1_t::MSE_sigmoid) {
            layer[l].delta = MSE_identity_f1(layer[l].a, output_target_vec);
        }

        if (m_structure.lossf_af_f1 == lfaf_f1_t::MSE_sigmoid) {
            // additional multiplication with derivative of acitivation function required

            std::vector<nn_fp_t> af1(num_nodes[l]);
            af1 = layer[l].af(layer[l].z, f_tag::f1);

            for (std::size_t to = 0; to < num_nodes[l]; ++to) {
                layer[l].delta[to] *= af1[to];
            }
        }

        if (m_structure.lossf_af_f1 == lfaf_f1_t::CE_softmax) {
            layer[l].delta = CE_softmax_f1(layer[l].a, output_target_vec);
        }
    }

    // calculate deltas in hidden layers
    for (std::size_t l = num_layers - 2; l > 0; --l) {

        std::vector<nn_fp_t> af1(num_nodes[l]);
        af1 = layer[l].af(layer[l].z, f_tag::f1);

        for (std::size_t from = 0; from < num_nodes[l]; ++from) {
            for (std::size_t to = 0; to < num_nodes[l + 1]; ++to) {
                layer[l].delta[from] = layer[l + 1].w[to][from] * layer[l + 1].delta[to];
            }
            layer[l].delta[from] *= af1[from];
        }
    }

    // Update dLdb and dLdw accordingly
    for (std::size_t l = 1; l < num_layers; ++l) {
        for (std::size_t to = 0; to < num_nodes[l]; ++to) {
            layer[l].dLdb[to] += layer[l].delta[to];
            for (std::size_t from = 0; from < num_nodes[l - 1]; ++from) {
                layer[l].dLdw[to][from] += layer[l].delta[to] * layer[l - 1].a[from];
            }
        }
    }
} // backward_pass

void neural_net::reset_dLdw_and_dLdb_to_zero()
{
    // reset all deltas for bias and weights to zero for next epochation in
    // gradient descent

    for (std::size_t l = 1; l < num_layers; ++l) {
        for (std::size_t to = 0; to < num_nodes[l]; ++to) {
            layer[l].dLdb[to] = 0.0;
            for (std::size_t from = 0; from < num_nodes[l - 1]; ++from) {
                layer[l].dLdw[to][from] = 0.0;
            }
        }
    }
} // reset_dLdw_and_dLdb_to_zero

void neural_net::update_w_and_b(nn_fp_t learning_rate, std::size_t num_samples)
{
    // update after backward_pass based on average accumulated gradient over
    // num_samples

    nn_fp_t scale_fact = learning_rate / num_samples;

    for (std::size_t l = 1; l < num_layers; ++l) {

        for (std::size_t to = 0; to < num_nodes[l]; ++to) {
            layer[l].b[to] -= scale_fact * layer[l].dLdb[to];
            for (std::size_t from = 0; from < num_nodes[l - 1]; ++from) {
                layer[l].w[to][from] -= scale_fact * layer[l].dLdw[to][from];
            }
        }
    }
} // update_w_and_b

nn_fp_t neural_net::get_partial_loss(std::vector<nn_fp_t> const& output_target_vec)
{
    //
    // REQUIRES:
    // forward_pass was done with input vector corresponding to
    // output_target_vec (i.e. neural_net is up-to-date w.r.t. the training pair)
    //
    // RETURN:
    // partial loss L_n for the training pair for given output_target_vec

    const std::size_t l = num_layers - 1;
    // partial loss L_n for the given training pair
    return lossf(layer[l].a, output_target_vec);
} // get_partial_loss

void neural_net::train(f_data_t const& fd_train, f_data_t const& td_train,
                       nn_training_meta_data_t m_data)
{
    // train the network using gradient descent

    std::string const prefix1{"   "};
    std::string const prefix2{"       "};

    // compatible number of lines for data and target values?
    std::size_t total_num_training_data_sets = fd_train.size();
    assert(total_num_training_data_sets == td_train.size());

    // core variables for iteration strategies
    std::uniform_int_distribution<> i_dist(0, total_num_training_data_sets - 1);
    std::size_t num_training_data_sets_batch;
    std::size_t num_batch_iter;
    std::vector<std::size_t> index_vec;

    switch (m_data.upstr) {

        case update_strategy_t::mini_batch_update:

            // limit mini_batch_size to total size of training data set
            if (m_data.mini_batch_size <= total_num_training_data_sets) {
                num_training_data_sets_batch = m_data.mini_batch_size;
            }
            else {
                // just in case the user required too large mini_batch_size
                num_training_data_sets_batch = total_num_training_data_sets;
            }
            num_batch_iter = total_num_training_data_sets / num_training_data_sets_batch;
            index_vec.resize(num_training_data_sets_batch);
            // acutal random indices will be assigned in mini batch loop
            break;

        case update_strategy_t::immediate_update:
            [[fallthrough]];
        case update_strategy_t::full_batch_update:

            num_training_data_sets_batch = total_num_training_data_sets;
            index_vec.resize(num_training_data_sets_batch);
            for (std::size_t i = 0; i < num_training_data_sets_batch; ++i) {
                index_vec[i] = i;
            }
            num_batch_iter = 1;
            break;

        default:
            std::unreachable();
    }

    fmt::print("Update strategy:              ");
    switch (m_data.upstr) {
        case update_strategy_t::immediate_update:
            fmt::println("immediate update");
            break;
        case update_strategy_t::mini_batch_update:
            fmt::println("mini batch update");
            break;
        case update_strategy_t::full_batch_update:
            fmt::println("full batch update");
            break;
        default:
            std::unreachable();
    }
    fmt::println("Number of training data sets: {}", total_num_training_data_sets);
    fmt::println("Batch size                  : {}", num_training_data_sets_batch);
    fmt::println("Number of batch iterations  : {}\n", num_batch_iter);


    nn_fp_t total_loss{0.0}, total_loss_old{0.0}; // total loss in respective epoch
    nn_fp_t total_loss_change_rate{0.0};

    for (size_t epoch = 1; epoch <= m_data.epochmax; ++epoch) {

        // fmt::println("{}Epoch: {}", prefix1, epoch);

        if (m_data.upstr == update_strategy_t::full_batch_update) {
            // fmt::println("{}full batch update: reset grad.", prefix1);

            reset_dLdw_and_dLdb_to_zero();
        }

        total_loss = 0.0;

        // mini batch iteration (num_batch_iter == 1 for most other cases)
        for (std::size_t mb_iter = 0; mb_iter < num_batch_iter; ++mb_iter) {

            if (m_data.upstr == update_strategy_t::mini_batch_update) {

                // select the minibatch subset of samples
                for (std::size_t i = 0; i < num_training_data_sets_batch; ++i) {
                    index_vec[i] = i_dist(gen);
                }
                // fmt::println("{}mini batch update: selected new batch of size {}",
                //              prefix2, num_training_data_sets_batch);
                // fmt::print("{}selected indices of batch : [ ", prefix2);
                // for (std::size_t i = 0; i < num_training_data_sets_batch; ++i) {
                //     if (i < num_training_data_sets_batch - 1) {
                //         fmt::print("{}, ", index_vec[i]);
                //     }
                //     else {
                //         fmt::println("{} ]", index_vec[i]);
                //     }
                // }

                // fmt::println("{}mini batch update: e:{}, mb:{}), reset grad.", prefix2,
                //              epoch, mb_iter);

                reset_dLdw_and_dLdb_to_zero();
            }

            // for each training pair in training batch do the learning cycle via
            // gradient calculation and weight and bias adaptation
            for (std::size_t n = 0; n < num_training_data_sets_batch; ++n) {

                if (m_data.upstr == update_strategy_t::immediate_update) {
                    // fmt::println("{}immediate update: e:{}, mb:{}, n: {}), reset "
                    //              "grad.",
                    //              prefix2, epoch, mb_iter, n);

                    // online variant: update directly after each training pair
                    reset_dLdw_and_dLdb_to_zero();
                }

                // select the next training pair
                std::vector<nn_fp_t> input_vec{fd_train[index_vec[n]]};
                std::vector<nn_fp_t> target_output_vec{td_train[index_vec[n]]};

                forward_pass(input_vec);
                total_loss +=
                    get_partial_loss(target_output_vec) / total_num_training_data_sets;

                backward_pass(input_vec, target_output_vec);

                if (m_data.upstr == update_strategy_t::immediate_update) {
                    // fmt::println("{}immediate update: e:{}, mb:{}, n: {}),"
                    //              "update grad.",
                    //              prefix2, epoch, mb_iter, n);

                    // online variant: update directly after each training pair
                    update_w_and_b(m_data.learning_rate, 1);
                }

            } // batch loop

            if (m_data.upstr == update_strategy_t::mini_batch_update) {
                // fmt::println("{}mini batch update: e:{}, mb:{}), update grad.",
                // prefix2, epoch, mb_iter);

                // update after the corresponding mini-batch of training samples
                update_w_and_b(m_data.learning_rate, num_training_data_sets_batch);
            }

        } // mini batch iteration

        if (m_data.upstr == update_strategy_t::full_batch_update) {
            // fmt::println("{}full batch update: update gradient", prefix1);

            // offline variant: update gradient after full set of training samples only
            update_w_and_b(m_data.learning_rate, num_training_data_sets_batch);
        }

        total_loss_change_rate = std::abs((total_loss - total_loss_old) / total_loss);

        if (epoch % m_data.epoch_output_skip == 0) {
            std::cout << "epoch " << std::setw(5) << epoch << ": ";
            std::cout << "L_tot: " << std::setw(9) << std::setprecision(4) << total_loss;
            if (epoch == 1) {
                // there is not reasonable dL yet
                std::cout << "\n";
            }
            else {
                std::cout << ", dL_tot: " << std::setw(9) << std::setprecision(4)
                          << total_loss - total_loss_old << ", dL_rel: " << std::setw(9)
                          << std::setprecision(4) << total_loss_change_rate << "\n";
            }
        }

        if (total_loss_change_rate < m_data.min_relative_loss_change_rate) {
            std::cout << "\nRelative change rate between epochs too small! Stopped "
                         "iteration early.\n\n";
            std::cout << "epoch " << epoch << ": dL_rel = " << total_loss_change_rate
                      << "\n\n\n";
            break;
        }
        total_loss_old = total_loss;

        if (total_loss < m_data.min_target_loss) {
            std::cout << "\nMinimum target loss reached before specified number of "
                         "iterations. Stopped iteration early.\n\n";
            std::cout << "epoch " << epoch << ": total_loss = " << total_loss << "\n\n\n";
            break;
        }
    } // epoch loop

} // train


bool is_classified_correctly(std::vector<nn_fp_t> const& output_vec,
                             std::vector<nn_fp_t> const& target_vec)
{
    if (output_vec.size() == 1) {
        // try this: delta^2 < 0.001 (arbitrarly chosen value!)
        nn_fp_t delta = output_vec[0] - target_vec[0];
        return (delta * delta < 0.001);
    }
    else {
        // try this: maximum of output and target vector are at same index and output
        // value at that index is larger than 0.5 (this is reasonable for a softmax
        // output layer, but questionable for other modes)

        auto out_iter = std::max_element(output_vec.begin(), output_vec.end());
        auto out_idx = std::distance(output_vec.begin(), out_iter);

        auto target_iter = std::max_element(target_vec.begin(), target_vec.end());
        auto target_idx = std::distance(target_vec.begin(), target_iter);

        return ((out_idx == target_idx) && (*out_iter > 0.5));
    }
}

void neural_net::test(f_data_t const& fd_test, f_data_t const& td_test)
{

    // fmt::println("target test data: {}", fmt::join(td_test, ", "));

    std::size_t num_test_samples = fd_test.size();
    std::string prefix{"    "};
    fmt::println("{}Target test data has a total of {} samples.", prefix,
                 num_test_samples);

    std::vector<nn_fp_t> output;
    std::size_t not_classified_correctly{0};

    for (std::size_t cnt = 0; cnt < num_test_samples; ++cnt) {
        output = forward_pass_with_output(fd_test[cnt]);
        // fmt::println("output: {}, target: {}", fmt::join(output, ", "),
        //              fmt::join(td_test[cnt], ", "));
        bool classified_correctly = is_classified_correctly(output, td_test[cnt]);
        if (!classified_correctly) ++not_classified_correctly;
        // fmt::println("output: {:8.3},    target: {:1.1},    correctly classifed:
        // {}",
        //              fmt::join(output, ", "), fmt::join(td_test[cnt], ", "),
        //              classified_correctly);
    }
    nn_fp_t accuracy =
        nn_fp_t(num_test_samples - not_classified_correctly) / num_test_samples;
    fmt::println("{}Correcly classified out of total samples: ({}/{}) => "
                 "accuracy: {:.2}\n",
                 prefix, num_test_samples - not_classified_correctly, num_test_samples,
                 accuracy);

} // test
