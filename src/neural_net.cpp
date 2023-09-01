#include "neural_net.hpp"

// uncomment to disable assert()
// #define NDEBUG
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>

std::random_device rd;
std::mt19937 gen(rd());

// TODO: make random distribution a selectable meta parameter

// uniform_real_distribution(from, to)
std::uniform_real_distribution<> d_ran(-1.0, 1.0);

// normal_disribution(mean,stddev)
// std::normal_distribution<double> d_ran(0.0, 1.0);

neural_net::neural_net(std::vector<int> &nn_nodes, a_func_ptr_t af) {

  // HINT: nn_nodes must only contain the number of nodes, the user requires
  // (not the additional bias handling nodes - they will be added internally!)

  // set up neural network structure:
  // add constant output nodes to network in order to enable simple handling of
  // bias values as weights (weight[l-1][i][0] corresponds to bias[l][i])

  num_layers = nn_nodes.size();
  // the minimum network has an input and an output layer
  assert(num_layers >= 2);

  num_nodes = nn_nodes;

  for (int l = 0; l < num_layers; ++l) {
    total_num_nodes += num_nodes[l];
    if (l > 0) {
      // bias values will not be used in input layer
      total_num_bias += num_nodes[l];
    }
  }

  // create and intialize nodes in all layers
  for (int l = 0; l < num_layers; ++l) {
    std::vector<nn_node> tmp_nodes;
    for (int n = 0; n < num_nodes[l]; ++n) {
      nn_node tmp_node;
      if (l == 0 || l == num_layers - 1) {
        // assign identity function for nodes in input and output layers
        tmp_node.af = &identity;
      } else {
        // assign user provided activation function for hidden layers
        tmp_node.af = af;
      }
      if (l > 0) {
        // assign bias with fixed or random values
        // tmp_node.b = 0.5;
        tmp_node.b = d_ran(gen);
      }
      tmp_nodes.push_back(tmp_node);
    }
    nodes.push_back(tmp_nodes);
  }

  // create and intialize arrays for weights w and corresponding dEdw
  int w_cnt{0};
  for (int l = 1; l < num_layers; ++l) {

    // to use w later:
    // int l_idx = l - 1; // simple index transformation, since weights
    //                    // are only needed starting with layer 1 and above
    //                    // however, weight indices still start with 0

    std::vector<std::vector<double>> tmp_weights;
    std::vector<std::vector<double>> tmp_dweights;

    for (int to = 0; to < num_nodes[l]; ++to) {

      std::vector<double> tmp_w;
      std::vector<double> tmp_dEdw;
      for (int from = 0; from < num_nodes[l - 1]; ++from) {

        double dval = 0.0; // dEdw is zero initialized

        // assign weights with fixed or random values
        // double val = 0.5;
        double val = d_ran(gen);

        tmp_w.push_back(val);
        tmp_dEdw.push_back(dval);
      }

      tmp_weights.push_back(tmp_w);
      tmp_dweights.push_back(tmp_dEdw);
      w_cnt += tmp_w.size();
    }
    w.push_back(tmp_weights);
    dEdw.push_back(tmp_dweights);
  }
  total_num_weights = w_cnt; // total number of weights in network
}

double neural_net::forward_pass(std::vector<double> &input_vec,
                                std::vector<double> &output_target_vec) {
  // propagate the input data through the network and return the value of the
  // partial error function E_n for this training pair

  // set input layer nodes to user provided values
  int l = 0; // input layer
  for (int to = 0; to < num_nodes[l]; ++to) {
    nodes[l][to].a = input_vec[to];
  }

  // forward pass through network starts at layer 1
  for (int l = 1; l < num_layers; ++l) {

    int l_idx = l - 1; // index transformation for weights (start index 0)

    // activate all nodes in previous layer to make output values available
    for (int from = 0; from < num_nodes[l - 1]; ++from) {
      nodes[l - 1][from].o =
          nodes[l - 1][from].af(nodes[l - 1][from].a, f_tag::f);
    }

    // calculate summed activation for all nodes incl. node bias
    for (int to = 0; to < num_nodes[l]; ++to) {
      nodes[l][to].a = nodes[l][to].b;
      for (int from = 0; from < num_nodes[l - 1]; ++from) {
        nodes[l][to].a += w[l_idx][to][from] * nodes[l - 1][from].o;
      }
    }
  }

  // activate all nodes in output layer to make output available
  l = num_layers - 1;
  for (int to = 0; to < num_nodes[l]; ++to) {
    nodes[l][to].o = nodes[l][to].af(nodes[l][to].a, f_tag::f);
  }

  double partial_error;
  for (int to = 0; to < num_nodes[l]; ++to) {
    partial_error = std::pow(nodes[l][to].o - output_target_vec[to], 2.0);
  }

  // partial error E_n for the given training pair
  return 0.5 * partial_error;
}

void neural_net::backward_pass(std::vector<double> &input_vec,
                               std::vector<double> &output_target_vec) {
  // calculate backward pass for contribution to gradient vector based on the
  // given input and target output vectors

  // calculate deltas in output layer
  int l = num_layers - 1;
  int l_idx = l - 1; // index transformation for weights (start index 0)

  for (int to = 0; to < num_nodes[l]; ++to) {
    nodes[l][to].delta = (nodes[l][to].o - output_target_vec[to]) *
                         nodes[l][to].af(nodes[l][to].a, f_tag::f1);
  }

  // Update dEdb and dEdw accordingly
  for (int to = 0; to < num_nodes[l]; ++to) {
    nodes[l][to].dEdb += nodes[l][to].delta;
    for (int from = 0; from < num_nodes[l - 1]; ++from) {
      dEdw[l_idx][to][from] += nodes[l][to].delta * nodes[l - 1][from].o;
    }
  }

  // calculate deltas in remaining layers
  for (int l = num_layers - 2; l > 0; --l) {

    int l_idx = l - 1; // index transformation for weights (start index 0)

    for (int from = 0; from < num_nodes[l]; ++from) {
      for (int to = 0; to < num_nodes[l + 1]; ++to) {
        nodes[l][from].delta = w[l_idx + 1][to][from] * nodes[l + 1][to].delta *
                               nodes[l][from].af(nodes[l][from].a, f_tag::f1);
      }
    }

    // Update dEdb and dEdw accordingly
    for (int to = 0; to < num_nodes[l]; ++to) {
      nodes[l][to].dEdb += nodes[l][to].delta;
      for (int from = 0; from < num_nodes[l - 1]; ++from) {
        dEdw[l_idx][to][from] += nodes[l][to].delta * nodes[l - 1][from].o;
      }
    }
  }
}

void neural_net::reset_dEdw_and_dEdb_to_zero() {
  // reset all deltas for bias and weights to zero for next iteration in
  // gradient descent

  for (int l = 1; l < num_layers; ++l) {
    int l_idx = l - 1; // index transformation for weights (start index 0)
    for (int to = 0; to < num_nodes[l]; ++to) {
      nodes[l][to].dEdb = 0.0;
      for (int from = 0; from < num_nodes[l - 1]; ++from) {
        dEdw[l_idx][to][from] = 0.0;
      }
    }
  }
}

void neural_net::update_w_and_b(double learning_rate, int num_samples) {

  double scale_fact = learning_rate / num_samples;

  for (int l = 1; l < num_layers; ++l) {
    int l_idx = l - 1; // index transformation for weights (start index 0)
    for (int to = 0; to < num_nodes[l]; ++to) {
      nodes[l][to].b -= scale_fact * nodes[l][to].dEdb;
      for (int from = 0; from < num_nodes[l - 1]; ++from) {
        w[l_idx][to][from] -= scale_fact * dEdw[l_idx][to][from];
      }
    }
  }
}

void neural_net::train(f_data_t &fd, f_data_t &td, update_strategy_t as) {
  // train the network using gradient descent

  const int itermax = 1500;

  // TODO: add lean_rate as argument
  double learning_rate = 0.1;

  // compatible number of lines for data and target values?
  int num_training_data_sets = fd.size();
  assert(num_training_data_sets == td.size());
  std::cout << "num_training_data_sets = " << num_training_data_sets
            << std::endl;

  switch (as) {
  case update_strategy_t::immediate_update:
    std::cout << "update strategy: immediate update" << std::endl;
    break;
  case update_strategy_t::batch_update:
    std::cout << "update strategy: batch update" << std::endl;
    break;
  }
  std::cout << std::endl;

  double total_iter_error{100.0}, total_iter_error_old{100.0};
  int iter{0}, cnt{0};
  while (total_iter_error > 0.01 && iter < itermax) {

    total_iter_error = 0.0;
    reset_dEdw_and_dEdb_to_zero();

    double total_error = 0.0;
    // for each input line do the weight & bias optimization pass
    for (int n = 0; n < num_training_data_sets; ++n) {

      std::vector<double> input_vec{fd[n]};
      std::vector<double> target_output_vec{td[n]};

      double partial_error = forward_pass(input_vec, target_output_vec);
      total_error += partial_error;

      backward_pass(input_vec, target_output_vec);

      if (as == update_strategy_t::immediate_update) {
        // online variant: directly update after each training pair
        update_w_and_b(learning_rate, num_training_data_sets);
      }

      if (cnt % num_training_data_sets == 0) {
        std::cout << "iteration over training data set: " << iter << std::endl;
      }
      std::cout << "  n: " << std::setw(3) << n << ", E_n: " << std::setw(9)
                << std::setprecision(4) << partial_error
                << ", E_total : " << std::setw(9) << std::setprecision(4)
                << total_error << std::endl;

      ++cnt;
    }

    if (as == update_strategy_t::batch_update) {
      // offline variant: update after the full set of training samples
      update_w_and_b(learning_rate, num_training_data_sets);
    }

    total_iter_error = total_error;
    if (iter > 0) {
      std::cout << "E_total(" << iter - 1 << ")-E_total(" << iter
                << "): " << total_error - total_iter_error_old << "\n\n";
    } else {
      std::cout << "\n";
    }
    total_iter_error_old = total_error;

    ++iter;
  }
}

void neural_net::print_parameters(std::string_view tag) {

  std::cout << "'" << tag << "' neural network with " << num_layers
            << " layers:" << std::endl;

  // print number of nodes with user provided sizes only (leave out bias
  // nodes)
  for (int l = 0; l < num_layers; ++l) {
    std::cout << "layer " << l << " : " << num_nodes[l] << " nodes"
              << std::endl;
  }
  std::cout << "number of nodes: " << total_num_nodes << std::endl;
  std::cout << "total number of weights: " << total_num_weights << std::endl;
  std::cout << "total number of bias values: " << total_num_bias << std::endl;
  std::cout << "total number of learning parameters: "
            << total_num_weights + total_num_bias << std::endl;
  std::cout << "+-------------------------------------------------------------+"
            << std::endl;

  return;
}

void neural_net::print_nodes(std::string_view tag) {

  for (int l = 0; l < num_layers; ++l) {
    std::cout << "'" << tag << "' - nodes layer " << l;
    if (l == 0) {
      std::cout << " (input layer):";
    } else if (l == num_layers - 1) {
      std::cout << " (ouput layer):";
    } else {
      std::cout << " (hidden layer):";
    }
    std::cout << std::endl;
    for (int n = 0; n < num_nodes[l]; ++n) {
      std::cout << "  n: " << n << " nodes[" << l << "][" << n
                << "].a = " << nodes[l][n].a << ", .b = " << nodes[l][n].b
                << ", .dEdb = " << nodes[l][n].dEdb
                << ", .o = " << nodes[l][n].o
                << ", .delta = " << nodes[l][n].delta;
      // std::cout << " &af = " << (void *)nodes[l][n].af;
      std::cout << std::endl;
    }
    if (l < num_layers - 1) {
      std::cout << std::endl;
    } else {
      std::cout
          << "+-------------------------------------------------------------+"
          << std::endl;
    }
  }
  // std::cout << std::endl;
}

void neural_net::print_weights(std::string_view tag) {

  for (int l = 1; l < num_layers; ++l) {
    std::cout << "'" << tag << "' - weights layer " << l;
    if (l == num_layers - 1) {
      std::cout << " (ouput layer):";
    } else {
      std::cout << " (hidden layer):";
    }
    std::cout << std::endl;
    int l_idx = l - 1; // index transformation for weights (start index 0)

    // show to user as index l, while internally using the index
    // transformation
    for (int to = 0; to < num_nodes[l]; ++to) {
      for (int from = 0; from < num_nodes[l - 1]; ++from) {
        std::cout << "    w[" << l << "][" << to << "][" << from << "] = ";
        std::cout.precision(4);
        std::cout << w[l_idx][to][from] << std::endl;
      }
    }
    std::cout << std::endl;
    for (int to = 0; to < num_nodes[l]; ++to) {
      for (int from = 0; from < num_nodes[l - 1]; ++from) {
        std::cout << "    dEdw[" << l << "][" << to << "][" << from << "] = ";
        std::cout.precision(4);
        std::cout << dEdw[l_idx][to][from] << std::endl;
      }
    }
    if (l < num_layers - 1) {
      std::cout << std::endl;
    } else {
      std::cout
          << "+-------------------------------------------------------------+"
          << std::endl;
    }
  }
}
