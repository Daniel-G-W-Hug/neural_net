#include "neural_net.hpp"

// uncomment to disable assert()
// #define NDEBUG
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric> // std::inner_product

std::random_device rd;
std::mt19937 gen(rd());

// uniform_real_distribution(from, to)
// std::uniform_real_distribution<double> d_ran(0.0, 1.0);

// normal_disribution(mean,stddev)
std::normal_distribution<double> d_ran(0.0, 1.0);

neural_net::neural_net(nn_structure_t structure_input)
    : m_structure{std::move(structure_input)} {
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
      std::size_t w_cnt{0};
      for (std::size_t to = 0; to < num_nodes[l]; ++to) {

        layer[l].b[to] = d_ran(gen); // random initialization
        layer[l].dLdb[to] = 0.0;

        layer[l].w[to].resize(num_nodes[l - 1]);
        layer[l].dLdw[to].resize(num_nodes[l - 1]);
        for (std::size_t from = 0; from < num_nodes[l - 1]; ++from) {
          layer[l].w[to][from] = d_ran(gen); // random initialization
          layer[l].dLdw[to][from] = 0.0;
        }
        w_cnt += num_nodes[l - 1];
      }
      total_num_weights += w_cnt;
    }

    // initialize activation function for layer
    if (l == 0) {
      // input layer
      layer[l].af = get_activation_func_ptr(a_func_t::identity);
    } else if (l == num_layers - 1) {
      // output layer
      layer[l].af = get_activation_func_ptr(m_structure.af_o);
    } else {
      // hidden layers
      layer[l].af = get_activation_func_ptr(m_structure.af_h);
    }
  }

  // set loss function
  lf = get_loss_func_ptr(m_structure.lf);

} // neural_net (ctor)

void neural_net::set_w_and_b_fixed(double val) {

  for (std::size_t l = 1; l < num_layers; ++l) {
    for (std::size_t to = 0; to < num_nodes[l]; ++to) {
      layer[l].b[to] = val;
      for (std::size_t from = 0; from < num_nodes[l - 1]; ++from) {
        layer[l].w[to][from] = val;
      }
    }
  }
} // set_w_and_b_fixed

void neural_net::forward_pass(std::vector<double> const &input_vec) {
  // propagate the input data through the network

  // set input layer nodes to user provided values
  std::size_t l = 0; // input layer
  layer[l].z = input_vec;

  // forward pass through inner network layers starts at layer 1
  for (std::size_t l = 1; l < num_layers; ++l) {

    // activate all nodes in previous layer to make output values available
    layer[l - 1].a = layer[l - 1].af(layer[l - 1].z, f_tag::f);

    // calculate summed activation for all nodes (incl. node bias)

    // for (std::size_t to = 0; to < num_nodes[l]; ++to) {
    //   layer[l].z[to] = 0.0;
    //   for (std::size_t from = 0; from < num_nodes[l - 1]; ++from) {
    //     layer[l].z[to] += layer[l].w[to][from] * layer[l - 1].a[from];
    //   }
    // }

    for (std::size_t to = 0; to < num_nodes[l]; ++to) {
      layer[l].z[to] =
          std::inner_product(layer[l].w[to].begin(), layer[l].w[to].end(),
                             layer[l - 1].a.begin(), layer[l].b[to]);
      // hint: use std::transform_reduce, in case if scalar product needs to be
      // parallelized later on
    }
  }

  // activate all nodes in output layer to make output available
  l = num_layers - 1;
  layer[l].a = layer[l].af(layer[l].z, f_tag::f);

} // forward_pass

std::vector<double>
neural_net::forward_pass_with_output(std::vector<double> const &input_vec) {

  forward_pass(input_vec);

  return layer[num_layers - 1].a;
} // forward_pass_with_output

void neural_net::backward_pass(std::vector<double> const &input_vec,
                               std::vector<double> const &output_target_vec) {
  // calculate backward pass for contribution to gradient vector based on the
  // given input and target output vectors

  // calculate deltas in output layer
  // TODO: replace specific implementation for MSE by arbitray loss function
  {
    std::size_t l = num_layers - 1;

    std::vector<double> af1(num_nodes[l]);
    af1 = layer[l].af(layer[l].z, f_tag::f1);

    for (std::size_t to = 0; to < num_nodes[l]; ++to) {
      layer[l].delta[to] =
          lf(layer[l].a[to], output_target_vec[to], f_tag::f1) * af1[to];
    }
  }

  // calculate deltas in hidden layers
  for (std::size_t l = num_layers - 2; l > 0; --l) {

    std::vector<double> af1(num_nodes[l]);
    af1 = layer[l].af(layer[l].z, f_tag::f1);

    for (std::size_t from = 0; from < num_nodes[l]; ++from) {
      for (std::size_t to = 0; to < num_nodes[l + 1]; ++to) {
        // TODO: check whether this can be put in an algo call
        layer[l].delta[from] =
            layer[l + 1].w[to][from] * layer[l + 1].delta[to] * af1[to];
      }
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

void neural_net::reset_dLdw_and_dLdb_to_zero() {
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

void neural_net::update_w_and_b(double learning_rate, std::size_t num_samples) {
  double scale_fact = learning_rate / num_samples;

  for (std::size_t l = 1; l < num_layers; ++l) {

    for (std::size_t to = 0; to < num_nodes[l]; ++to) {
      layer[l].b[to] -= scale_fact * layer[l].dLdb[to];
      for (std::size_t from = 0; from < num_nodes[l - 1]; ++from) {
        layer[l].w[to][from] -= scale_fact * layer[l].dLdw[to][from];
      }
    }
  }
} // update_w_and_b

double
neural_net::get_partial_loss(std::vector<double> const &output_target_vec) {
  //
  // REQUIRES:
  // forward_pass was done with input vector corresponding to
  // output_target_vec (i.e. neural_net is up-to-date w.r.t. the input vector)
  //
  // RETURN:
  // partial loss L_n for the training pair for given output_target_vec

  std::size_t l = num_layers - 1;
  double partial_loss{0.0};
  for (std::size_t to = 0; to < num_nodes[l]; ++to) {
    partial_loss += lf(layer[l].a[to], output_target_vec[to], f_tag::f);
  }
  // partial loss L_n for the given training pair
  return partial_loss;
} // get_partial_loss

void neural_net::train(f_data_t const &fd, f_data_t const &td,
                       nn_training_meta_data_t m_data) {
  // train the network using gradient descent

  // compatible number of lines for data and target values?
  std::size_t total_num_training_data_sets = fd.size();
  assert(total_num_training_data_sets == td.size());
  std::cout << "total_num_training_data_sets = " << total_num_training_data_sets
            << std::endl;

  std::uniform_int_distribution<> i_dist(0, total_num_training_data_sets - 1);
  std::size_t num_training_data_sets_batch;
  std::vector<std::size_t> index_vec;

  if (m_data.upstr == update_strategy_t::mini_batch_update) {

    // limit mini_batch_size to total size of training data set
    if (m_data.mini_batch_size <= total_num_training_data_sets) {
      num_training_data_sets_batch = m_data.mini_batch_size;
    } else {
      num_training_data_sets_batch = total_num_training_data_sets;
    }
    std::cout << "update strategy: mini batch,"
              << " batch size: " << num_training_data_sets_batch << "\n\n";

    index_vec.resize(num_training_data_sets_batch);
    // acutal random indices will be assigned in mini batch loop

  } else {

    if (m_data.upstr == update_strategy_t::immediate_update) {
      std::cout << "update strategy: immediate update\n\n";
    }
    if (m_data.upstr == update_strategy_t::full_batch_update) {

      std::cout << "update strategy: full batch update\n\n";
    }
    num_training_data_sets_batch = total_num_training_data_sets;
    index_vec.resize(num_training_data_sets_batch);
    for (std::size_t i = 0; i < num_training_data_sets_batch; ++i) {
      index_vec[i] = i;
    }
  }

  double total_loss{0.0}, total_loss_old{0.0}; // total loss in respective epoch
  double total_loss_change_rate{0.0};

  for (size_t epoch = 1; epoch <= m_data.epochmax; ++epoch) {

    if (m_data.upstr != update_strategy_t::immediate_update) {
      reset_dLdw_and_dLdb_to_zero();
    }
    total_loss = 0.0;

    if (m_data.upstr == update_strategy_t::mini_batch_update) {
      // select the minibatch subset of samples for this epoch
      for (std::size_t i = 0; i < num_training_data_sets_batch; ++i) {
        index_vec[i] = i_dist(gen);
      }
      // std::cout << "selected indices of batch : [ ";
      // for (std::size_t i = 0; i < num_training_data_sets_batch; ++i) {
      //   if (i < num_training_data_sets_batch - 1) {
      //     std::cout << index_vec[i] << ", ";
      //   } else {
      //     std::cout << index_vec[i] << " ]\n";
      //   }
      // }
    }

    // for each training pair in training batch do the learning cycle via
    // gradient calculation and weight and bias adaptation
    for (std::size_t n = 0; n < num_training_data_sets_batch; ++n) {

      if (m_data.upstr == update_strategy_t::immediate_update) {
        // online variant: update directly after each training pair
        reset_dLdw_and_dLdb_to_zero();
      }

      // select the next training pair
      std::vector<double> input_vec{fd[index_vec[n]]};
      std::vector<double> target_output_vec{td[index_vec[n]]};

      forward_pass(input_vec);
      total_loss +=
          get_partial_loss(target_output_vec) / num_training_data_sets_batch;

      backward_pass(input_vec, target_output_vec);

      if (m_data.upstr == update_strategy_t::immediate_update) {
        // online variant: update directly after each training pair
        update_w_and_b(m_data.learning_rate, num_training_data_sets_batch);
        // update_w_and_b(m_data.learning_rate, 1);
      }

    } // batch loop

    if (m_data.upstr != update_strategy_t::immediate_update) {
      // offline variant: update after the corresponding batch of
      // training samples
      update_w_and_b(m_data.learning_rate, num_training_data_sets_batch);
    }

    total_loss_change_rate =
        std::abs((total_loss - total_loss_old) / total_loss);

    if (epoch % m_data.epoch_output_skip == 0) {
      std::cout << "epoch " << std::setw(5) << epoch << ": ";
      std::cout << "L_total: " << std::setw(9) << std::setprecision(4)
                << total_loss;
      if (epoch == 1) {
        // there is not reasonable dL yet
        std::cout << "\n";
      } else {
        std::cout << ", dL_total: " << std::setw(9) << std::setprecision(4)
                  << total_loss - total_loss_old
                  << ", dL_rel_change: " << std::setw(9) << std::setprecision(4)
                  << total_loss_change_rate << "\n";
      }
    }

    if (total_loss_change_rate < m_data.min_relative_loss_change_rate) {
      std::cout << "\nRelative change rate between epochs too small! Stopped "
                   "iteration early.\n\n";
      std::cout << "epoch " << epoch
                << ": dL_rel_change = " << total_loss_change_rate << "\n\n\n";
      break;
    }
    total_loss_old = total_loss;

    if (total_loss < m_data.min_target_loss) {
      std::cout << "\nMinimum target loss reached before specified number of "
                   "iterations. Stopped iteration early.\n\n";
      std::cout << "epoch " << epoch << ": total_loss = " << total_loss
                << "\n\n\n";
      break;
    }
  } // epoch loop

} // train
