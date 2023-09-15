#include "neural_net.hpp"

// uncomment to disable assert()
// #define NDEBUG
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>

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
  assert(num_layers >= 2);

  num_nodes = m_structure.net_structure;

  for (int l = 0; l < num_layers; ++l) {
    if (l > 0) {
      // calc no. of bias values before adding the extra nodes
      total_num_bias += num_nodes[l];
    }
    // add const output nodes in input and hidden layers
    // for bias calculation
    num_nodes[l] += 1;
    total_num_nodes += num_nodes[l];
  }

  // create and intialize nodes in all layers
  // (activation functions and bias values)
  for (int l = 0; l < num_layers; ++l) {
    std::vector<nn_node_t> tmp_nodes;
    for (int n = 0; n < num_nodes[l]; ++n) {
      nn_node_t tmp_node;
      if (l == 0) {
        // for input layer
        tmp_node.af = get_activation_func_ptr(a_func_t::identity);
      } else if (l == num_layers - 1) {
        tmp_node.af = get_activation_func_ptr(m_structure.af_o);
      } else {
        tmp_node.af = get_activation_func_ptr(m_structure.af_h);
      }
      if (n == num_nodes[l] - 1) {
        // for const input node used for bias calculation
        tmp_node.af = get_activation_func_ptr(a_func_t::identity);
        tmp_node.a = 1.0;
      }
      tmp_nodes.push_back(tmp_node);
    }
    nodes.push_back(tmp_nodes);
  }

  // create and intialize arrays for weights w and corresponding dLdw
  int w_cnt{0};
  for (int l = 1; l < num_layers; ++l) {

    // to use w later:
    // int l_idx = l - 1; // simple index transformation, since weights
    //                    // are only needed starting with layer 1 and above
    //                    // however, weight indices still start with 0

    std::vector<std::vector<double>> tmp_weights;
    std::vector<std::vector<double>> tmp_dweights;

    for (int to = 0; to < num_nodes[l] - 1; ++to) {

      std::vector<double> tmp_w;
      std::vector<double> tmp_dLdw;
      for (int from = 0; from < num_nodes[l - 1]; ++from) {

        double dval = 0.0; // dLdw is zero initialized

        // assign weights random values
        double val = d_ran(gen);

        tmp_w.push_back(val);
        tmp_dLdw.push_back(dval);
      }

      tmp_weights.push_back(tmp_w);
      tmp_dweights.push_back(tmp_dLdw);
      w_cnt += tmp_w.size();
    }
    w.push_back(tmp_weights);
    dLdw.push_back(tmp_dweights);
  }
  total_num_weights = w_cnt; // total number of weights in network

} // neural_net (ctor)

void neural_net::set_w_fixed(double val) {

  for (int l = 1; l < num_layers; ++l) {
    int l_idx = l - 1; // index transformation for weights (start index 0)
    for (int to = 0; to < num_nodes[l] - 1; ++to) {
      for (int from = 0; from < num_nodes[l - 1]; ++from) {
        w[l_idx][to][from] = val;
      }
    }
  }
} // set_w_and_b_fixed

void neural_net::forward_pass(std::vector<double> &input_vec) {
  // propagate the input data through the network

  // set input layer nodes to user provided values
  int l = 0; // input layer
  for (int to = 0; to < num_nodes[l] - 1; ++to) {
    nodes[l][to].a = input_vec[to];
  }
  // // set const output node for bias calculation
  // nodes[l][num_nodes[l] - 1].a = 1.0;

  // forward pass through inner network layers starts at layer 1
  for (int l = 1; l < num_layers; ++l) {

    int l_idx = l - 1; // index transformation for weights (start index 0)

    // activate all nodes in previous layer to make output values available
    for (int from = 0; from < num_nodes[l - 1]; ++from) {
      nodes[l - 1][from].o =
          nodes[l - 1][from].af(nodes[l - 1][from].a, f_tag::f);
    }

    // calculate summed activation for all nodes (incl. node bias)
    for (int to = 0; to < num_nodes[l] - 1; ++to) {
      nodes[l][to].a = 0.0;
      for (int from = 0; from < num_nodes[l - 1]; ++from) {
        nodes[l][to].a += w[l_idx][to][from] * nodes[l - 1][from].o;
      }
    }
  }

  // activate all nodes in output layer to make output available
  l = num_layers - 1;
  for (int to = 0; to < num_nodes[l] - 1; ++to) {
    nodes[l][to].o = nodes[l][to].af(nodes[l][to].a, f_tag::f);
  }

} // forward_pass

std::vector<double>
neural_net::forward_pass_with_output(std::vector<double> &input_vec) {
  // propagate the input data through the network

  forward_pass(input_vec);

  std::vector<double> output_vec;

  int l = num_layers - 1;
  output_vec.reserve(num_nodes[l] - 1);

  // copy output layer to output vector
  for (int to = 0; to < num_nodes[l] - 1; ++to) {
    output_vec[to] = nodes[l][to].o;
  }

  return output_vec;
} // forward_pass_with_output

void neural_net::backward_pass(std::vector<double> &input_vec,
                               std::vector<double> &output_target_vec) {
  // calculate backward pass for contribution to gradient vector based on the
  // given input and target output vectors

  // calculate deltas in output layer
  int l = num_layers - 1;
  for (int to = 0; to < num_nodes[l] - 1; ++to) {
    nodes[l][to].delta = (nodes[l][to].o - output_target_vec[to]) *
                         nodes[l][to].af(nodes[l][to].a, f_tag::f1);
  }

  // calculate deltas in hidden layers
  for (int l = num_layers - 2; l > 0; --l) {
    int l_idx = l - 1; // index transformation for weights (start index 0)
    for (int from = 0; from < num_nodes[l]; ++from) {
      for (int to = 0; to < num_nodes[l + 1] - 1; ++to) {
        nodes[l][from].delta = w[l_idx + 1][to][from] * nodes[l + 1][to].delta *
                               nodes[l][from].af(nodes[l][from].a, f_tag::f1);
      }
    }
  }

  // Update dLdb and dLdw accordingly
  for (int l = 1; l < num_layers; ++l) {
    int l_idx = l - 1; // index transformation for weights (start index 0)
    for (int to = 0; to < num_nodes[l] - 1; ++to) {
      for (int from = 0; from < num_nodes[l - 1]; ++from) {
        dLdw[l_idx][to][from] += nodes[l][to].delta * nodes[l - 1][from].o;
      }
    }
  }
} // backward_pass

void neural_net::reset_dLdw_to_zero() {
  // reset all deltas for bias and weights to zero for next epochation in
  // gradient descent

  for (int l = 1; l < num_layers; ++l) {
    int l_idx = l - 1; // index transformation for weights (start index 0)
    for (int to = 0; to < num_nodes[l] - 1; ++to) {
      for (int from = 0; from < num_nodes[l - 1]; ++from) {
        dLdw[l_idx][to][from] = 0.0;
      }
    }
  }
} // reset_dLdw_to_zero

void neural_net::update_w(double learning_rate, int num_samples) {
  double scale_fact = learning_rate / num_samples;

  for (int l = 1; l < num_layers; ++l) {
    int l_idx = l - 1; // index transformation for weights (start index 0)
    for (int to = 0; to < num_nodes[l] - 1; ++to) {
      for (int from = 0; from < num_nodes[l - 1]; ++from) {
        w[l_idx][to][from] -= scale_fact * dLdw[l_idx][to][from];
      }
    }
  }
} // update_w

double neural_net::get_partial_loss(std::vector<double> &output_target_vec) {
  //
  // REQUIRES:
  // forward_pass was done with input vector corresponding to
  // output_target_vec (i.e. neural_net is up-to-date w.r.t. the input vector)
  //
  // RETURNS:
  // partial loss L_n for the training pair for given output_target_vec
  //
  // total loss L = 1/N*sum(1..N)( 0.5*(y_actual - y_target)^2 )
  //              = 1/N*sum(1..N)(L_n)
  //
  // (factor 0.5 was chosen to simplify derivatives dEdw and dEdb)

  int l = num_layers - 1;
  double partial_loss;
  for (int to = 0; to < num_nodes[l] - 1; ++to) {
    partial_loss = std::pow(nodes[l][to].o - output_target_vec[to], 2.0);
  }
  // partial loss L_n for the given training pair
  return 0.5 * partial_loss;
} // get_partial_loss

void neural_net::train(f_data_t &fd, f_data_t &td,
                       nn_training_meta_data_t m_data) {
  // train the network using gradient descent

  // compatible number of lines for data and target values?
  int total_num_training_data_sets = fd.size();
  assert(total_num_training_data_sets == td.size());
  std::cout << "total_num_training_data_sets = " << total_num_training_data_sets
            << std::endl;

  std::uniform_int_distribution<> i_dist(0, total_num_training_data_sets - 1);
  int num_training_data_sets_batch;
  std::vector<int> index_vec;

  if (m_data.upstr == update_strategy_t::mini_batch_update) {

    // limit mini_batch_size to total size of training data set
    if (m_data.mini_batch_size <= total_num_training_data_sets) {
      num_training_data_sets_batch = m_data.mini_batch_size;
    } else {
      num_training_data_sets_batch = total_num_training_data_sets;
    }
    std::cout << "update strategy: mini batch,"
              << " batch size: " << num_training_data_sets_batch << "\n\n";

    index_vec.reserve(num_training_data_sets_batch);
    // acutal random indices will be assigned in mini batch loop

  } else {

    if (m_data.upstr == update_strategy_t::immediate_update) {
      std::cout << "update strategy: immediate update\n\n";
    }
    if (m_data.upstr == update_strategy_t::full_batch_update) {

      std::cout << "update strategy: full batch update\n\n";
    }
    num_training_data_sets_batch = total_num_training_data_sets;
    index_vec.reserve(num_training_data_sets_batch);
    for (int i = 0; i < num_training_data_sets_batch; ++i) {
      index_vec[i] = i;
    }
  }

  double total_loss{0.0}, total_loss_old{0.0}; // total loss in respective epoch
  double total_loss_change_rate{0.0};

  for (size_t epoch = 1; epoch <= m_data.epochmax; ++epoch) {

    if (m_data.upstr != update_strategy_t::immediate_update) {
      reset_dLdw_to_zero();
    }
    total_loss = 0.0;

    if (m_data.upstr == update_strategy_t::mini_batch_update) {
      // select the minibatch subset of samples for this epoch
      for (int i = 0; i < num_training_data_sets_batch; ++i) {
        index_vec[i] = i_dist(gen);
      }
      // std::cout << "selected indices of batch : [ ";
      // for (int i = 0; i < num_training_data_sets_batch; ++i) {
      //   if (i < num_training_data_sets_batch - 1) {
      //     std::cout << index_vec[i] << ", ";
      //   } else {
      //     std::cout << index_vec[i] << " ]\n";
      //   }
      // }
    }

    // for each training pair in training batch do the learning cycle via
    // gradient calculation and weight and bias adaptation
    for (int n = 0; n < num_training_data_sets_batch; ++n) {

      if (m_data.upstr == update_strategy_t::immediate_update) {
        // online variant: update directly after each training pair
        reset_dLdw_to_zero();
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
        update_w(m_data.learning_rate, num_training_data_sets_batch);
        // update_w_and_b(m_data.learning_rate, 1);
      }

    } // batch loop

    if (m_data.upstr != update_strategy_t::immediate_update) {
      // offline variant: update after the corresponding batch of
      // training samples
      update_w(m_data.learning_rate, num_training_data_sets_batch);
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
