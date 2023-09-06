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
// std::uniform_real_distribution<double> d_ran(-1.0, 1.0);

// normal_disribution(mean,stddev)
std::normal_distribution<double> d_ran(0.0, 1.0);

neural_net::neural_net(nn_meta_data_t meta_data_input)
    : m_data{std::move(meta_data_input)} {
  // set up neural network structure

  num_layers = m_data.net_structure.size();
  // the minimum network has an input and an output layer
  assert(num_layers >= 2);

  num_nodes = m_data.net_structure;

  for (int l = 0; l < num_layers; ++l) {
    total_num_nodes += num_nodes[l];
    if (l > 0) {
      // bias values will not be used in input layer
      total_num_bias += num_nodes[l];
    }
  }

  // create and intialize nodes in all layers
  for (int l = 0; l < num_layers; ++l) {
    std::vector<nn_node_t> tmp_nodes;
    for (int n = 0; n < num_nodes[l]; ++n) {
      nn_node_t tmp_node;
      if (l == 0) {
        // assign identity function for nodes in input layer
        tmp_node.af = &identity;
      } else {
        // assign user provided activation function for other layers
        switch (m_data.af) {
        case (a_func_t::identity):
          tmp_node.af = &identity;
          break;
        case (a_func_t::sigmoid):
          tmp_node.af = &sigmoid;
          break;
        case (a_func_t::tanhyp):
          tmp_node.af = &tanhyp;
          break;
        case (a_func_t::reLU):
          tmp_node.af = &reLU;
          break;
        case (a_func_t::leaky_reLU):
          tmp_node.af = &leaky_reLU;
          break;
        }
      }
      if (l > 0) {
        // assign bias with fixed or random values
        // tmp_node.b = 1.0;
        tmp_node.b = d_ran(gen);
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

    for (int to = 0; to < num_nodes[l]; ++to) {

      std::vector<double> tmp_w;
      std::vector<double> tmp_dLdw;
      for (int from = 0; from < num_nodes[l - 1]; ++from) {

        double dval = 0.0; // dLdw is zero initialized

        // assign weights with fixed or random values
        // double val = 1.0;
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
}

void neural_net::forward_pass(std::vector<double> &input_vec) {
  // propagate the input data through the network

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
}

std::vector<double>
neural_net::forward_pass_with_output(std::vector<double> &input_vec) {
  // propagate the input data through the network

  forward_pass(input_vec);

  std::vector<double> output_vec;

  int l = num_layers - 1;
  output_vec.reserve(num_nodes[l]);

  // copy output layer to output vector
  for (int to = 0; to < num_nodes[l]; ++to) {
    output_vec[to] = nodes[l][to].o;
  }

  return output_vec;
}

void neural_net::backward_pass(std::vector<double> &input_vec,
                               std::vector<double> &output_target_vec) {
  // calculate backward pass for contribution to gradient vector based on the
  // given input and target output vectors

  // calculate deltas in output layer
  int l = num_layers - 1;
  for (int to = 0; to < num_nodes[l]; ++to) {
    nodes[l][to].delta = (nodes[l][to].o - output_target_vec[to]) *
                         nodes[l][to].af(nodes[l][to].a, f_tag::f1);
  }

  // calculate deltas in hidden layers
  for (int l = num_layers - 2; l > 0; --l) {
    int l_idx = l - 1; // index transformation for weights (start index 0)
    for (int from = 0; from < num_nodes[l]; ++from) {
      for (int to = 0; to < num_nodes[l + 1]; ++to) {
        nodes[l][from].delta = w[l_idx + 1][to][from] * nodes[l + 1][to].delta *
                               nodes[l][from].af(nodes[l][from].a, f_tag::f1);
      }
    }
  }

  // Update dLdb and dLdw accordingly
  for (int l = 1; l < num_layers; ++l) {
    int l_idx = l - 1; // index transformation for weights (start index 0)
    for (int to = 0; to < num_nodes[l]; ++to) {
      nodes[l][to].dLdb += nodes[l][to].delta;
      for (int from = 0; from < num_nodes[l - 1]; ++from) {
        dLdw[l_idx][to][from] += nodes[l][to].delta * nodes[l - 1][from].o;
      }
    }
  }
}

void neural_net::reset_dLdw_and_dLdb_to_zero() {
  // reset all deltas for bias and weights to zero for next epochation in
  // gradient descent

  for (int l = 1; l < num_layers; ++l) {
    int l_idx = l - 1; // index transformation for weights (start index 0)
    for (int to = 0; to < num_nodes[l]; ++to) {
      nodes[l][to].dLdb = 0.0;
      for (int from = 0; from < num_nodes[l - 1]; ++from) {
        dLdw[l_idx][to][from] = 0.0;
      }
    }
  }
}

void neural_net::update_w_and_b(double learning_rate, int num_samples) {
  double scale_fact = learning_rate / num_samples;

  for (int l = 1; l < num_layers; ++l) {
    int l_idx = l - 1; // index transformation for weights (start index 0)
    for (int to = 0; to < num_nodes[l]; ++to) {
      nodes[l][to].b -= scale_fact * nodes[l][to].dLdb;
      for (int from = 0; from < num_nodes[l - 1]; ++from) {
        w[l_idx][to][from] -= scale_fact * dLdw[l_idx][to][from];
      }
    }
  }
}

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
  for (int to = 0; to < num_nodes[l]; ++to) {
    partial_loss = std::pow(nodes[l][to].o - output_target_vec[to], 2.0);
  }
  // partial loss L_n for the given training pair
  return 0.5 * partial_loss;
}

void neural_net::train(f_data_t &fd, f_data_t &td) {
  // train the network using gradient descent

  // compatible number of lines for data and target values?
  int num_training_data_sets = fd.size();
  assert(num_training_data_sets == td.size());
  std::cout << "num_training_data_sets = " << num_training_data_sets
            << std::endl;

  switch (m_data.upstr) {
  case update_strategy_t::immediate_update:
    std::cout << "update strategy: immediate update" << std::endl;
    break;
  case update_strategy_t::batch_update:
    std::cout << "update strategy: batch update" << std::endl;
    break;
  }
  std::cout << std::endl;

  bool keep_running{true};
  double total_epoch_loss{0.0}, total_epoch_loss_old{0.0};
  double epoch_loss_updated{0.0};

  int epoch{0}, cnt{0};
  do {

    reset_dLdw_and_dLdb_to_zero(); // implicitly happens for each pass through
                                   // training data

    total_epoch_loss = 0.0;

    double total_loss = 0.0;
    // for each input line do the weight & bias optimization pass
    for (int n = 0; n < num_training_data_sets; ++n) {

      if (m_data.upstr == update_strategy_t::immediate_update) {
        // online variant: update directly after each training pair
        reset_dLdw_and_dLdb_to_zero();
      }

      std::vector<double> input_vec{fd[n]};
      std::vector<double> target_output_vec{td[n]};

      forward_pass(input_vec);
      double partial_loss = get_partial_loss(target_output_vec);
      total_loss += partial_loss / num_training_data_sets;

      backward_pass(input_vec, target_output_vec);

      if (m_data.upstr == update_strategy_t::immediate_update) {
        // online variant: update directly after each training pair
        update_w_and_b(m_data.learning_rate, 1);
      }

      if (cnt % num_training_data_sets == 0 &&
          epoch % m_data.epoch_output_skip == 0) {
        std::cout << "epoch " << std::setw(5) << epoch << ": ";
      }

      ++cnt;
    }

    if (m_data.upstr == update_strategy_t::batch_update) {
      // offline variant: update after the full set of training samples
      update_w_and_b(m_data.learning_rate, num_training_data_sets);
      reset_dLdw_and_dLdb_to_zero();
    }

    total_epoch_loss = total_loss;
    if (epoch % m_data.epoch_output_skip == 0) {
      std::cout << "L_total: " << std::setw(9) << std::setprecision(4)
                << total_loss;
      // std::cout << "\nL_total after update: " << std::setw(9)
      //           << std::setprecision(4) << epoch_loss_updated;
      if (epoch == 0) {
        // there is not reasonable dL yet
        std::cout << "\n";
      } else {
        std::cout << ", dL_total: " << std::setw(9) << std::setprecision(4)
                  << total_loss - total_epoch_loss_old << "\n";
      }
    }

    if (std::abs((total_loss - total_epoch_loss_old) / total_loss) <
        m_data.min_relative_loss_change_rate) {
      // stop if change rate becomes too small
      keep_running = false;
      std::cout << "\nRelative change rate too small! Stopped "
                   "iteration.\n\n\n";
    }

    total_epoch_loss_old = total_loss;
    ++epoch;

  } while ((total_epoch_loss > m_data.min_target_loss) &&
           (epoch < m_data.epochmax) && keep_running);
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
  std::cout << "+------------------------------------------------------"
               "-------+"
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
      std::cout << "  n: " << n << " nodes[" << l << "][" << n << "].a = ";
      std::cout.precision(5);
      std::cout << nodes[l][n].a << ", .b = " << nodes[l][n].b
                << ", .dLdb = " << nodes[l][n].dLdb
                << ", .o = " << nodes[l][n].o
                << ", .delta = " << nodes[l][n].delta;
      // std::cout << " &af = " << (void *)nodes[l][n].af;
      std::cout << std::endl;
    }
    if (l < num_layers - 1) {
      std::cout << std::endl;
    } else {
      std::cout << "+--------------------------------------------------"
                   "-----------+"
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
        std::cout.precision(5);
        std::cout << w[l_idx][to][from] << std::endl;
      }
    }
    std::cout << std::endl;
    for (int to = 0; to < num_nodes[l]; ++to) {
      for (int from = 0; from < num_nodes[l - 1]; ++from) {
        std::cout << "    dLdw[" << l << "][" << to << "][" << from << "] = ";
        std::cout.precision(5);
        std::cout << dLdw[l_idx][to][from] << std::endl;
      }
    }
    if (l < num_layers - 1) {
      std::cout << std::endl;
    } else {
      std::cout << "+--------------------------------------------------"
                   "-----------+"
                << std::endl;
    }
  }
}
