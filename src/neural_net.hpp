#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP

#include "activation_func.hpp"
#include "file_io.hpp"

#include <random>
#include <string_view>
#include <vector>

enum class update_strategy_t { immediate_update, batch_update };

struct nn_node {

  // training is done by given pairs (x, y) - x and y can be vectors
  // in each layer l=0..num_layers-1 there are num_nodes[l]

  double a{0.0}; // input side of node (=activation)
                 // either gets direct input (for input layer)
                 // or gets weighted sum of inputs of previous layer

  double b{0.0};    // bias of node (not used for input layer)
  double dEdb{0.0}; // delta of error function E depending on bias
                    // (used for backpropagation)

  double o{0.0}; // output side of node; gets its value after applying the
                 // activation function o = af(sum_prev_layer(a*x) + b)

  double delta{0.0}; // storage for backpropagation

  a_func_ptr_t af{nullptr}; // ptr to activation function
};

struct neural_net {

  int num_layers; // number of layers in neural_net
                  // layer (l == 0): input layer
                  // (l > 1) && (l < num_layers - 1): hidden layer
                  // layer (l == num_layers-1): output layer

  std::vector<int> num_nodes; // number of nodes in each layer
  int total_num_nodes{0};     // total number of nodes

  int total_num_bias{0}; // total number of adjustable bias
                         // values for gradient descent

  using node_matrix_t = std::vector<std::vector<nn_node>>;
  using weight_matrix_t = std::vector<std::vector<std::vector<double>>>;

  node_matrix_t nodes;  // vector of nodes in each layer
  weight_matrix_t w;    // weight matrix
  weight_matrix_t dEdw; // delta of error function E depending on weights w
                        // (used for backpropagation)
  // indexing: w[layer][to_node in l][from_node in l-1]
  // layout optimized for scalar product in
  // sum over activations coming from nodes in previous layer:
  // sum of w[l][to][from]*y[l-1][from]
  // e.g. for fast calculation of forward pass in applications of trained net

  int total_num_weights; // total number of weights in network

  neural_net(std::vector<int> &nn_nodes, a_func_ptr_t af);

  double forward_pass(std::vector<double> &input_vec,
                      std::vector<double> &target_vec);
  void backward_pass(std::vector<double> &input_vec,
                     std::vector<double> &target_vec);

  void reset_dEdw_and_dEdb_to_zero();
  void update_w_and_b(double learn_rate, int num_samples);

  void train(f_data_t &fd, f_data_t &td,
             update_strategy_t as = update_strategy_t::immediate_update);

  void print_parameters(std::string_view tag);
  void print_nodes(std::string_view tag);
  void print_weights(std::string_view tag);
};

#endif // NEURAL_NET_HPP
