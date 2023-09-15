#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP

#include "activation_func.hpp"

#include <random>
#include <string_view>
#include <vector>

// training data comes in csv files organized in lines with constant number of
// integer or double values stored in f_data_t
using f_data_t = std::vector<std::vector<double>>;

enum class update_strategy_t {
  immediate_update = 1,
  mini_batch_update,
  full_batch_update
};

struct nn_structure_t {

  std::vector<int> net_structure; // number of nodes for each layer

  a_func_t af_h; // chosen activation function for hidden layers
  a_func_t af_o; // chosen activation function for output layer
};
struct nn_training_meta_data_t {

  int epochmax;          // max. number of iterations over the whole data set
  int epoch_output_skip; // output if epoch%epoch_output_skip == 0

  double learning_rate;   // chosen learning rate for gradient descent
  double min_target_loss; // minimum loss in training to stop prescribed
                          // iteration, even if epoch < epochmax
  double min_relative_loss_change_rate; // stop iteration, if loss change
                                        // becomes too small between epochs

  update_strategy_t upstr; // update strategy for gradient descent
  int mini_batch_size;     // number of training pairs for mini batch
};

struct nn_node_t {

  // training is done by given pairs (x, y) - x and y can be vectors
  // in each layer l=0..num_layers-1 there are num_nodes[l]

  double a{0.0}; // input side of node (=activation)
                 // either gets direct input (for input layer)
                 // or gets weighted sum of inputs of previous layer

  double b{0.0};    // bias of node (not used for input layer)
  double dLdb{0.0}; // delta of error function E depending on bias
                    // (used for backpropagation)

  double o{0.0}; // output side of node; gets its value after applying the
                 // activation function o = af(sum_prev_layer(a*x) + b)

  double delta{0.0}; // storage for backpropagation

  a_func_ptr_t af{nullptr}; // ptr to activation function
};

struct neural_net {

  nn_structure_t m_structure;

  int num_layers; // number of layers in neural_net:
                  // l == 0: input layer
                  // (l > 1) && (l < num_layers - 1): hidden layer
                  // l == num_layers-1: output layer

  std::vector<int> num_nodes; // number of nodes in each layer
  int total_num_nodes{0};     // total number of nodes

  int total_num_bias{0}; // total number of adjustable bias
                         // values for gradient descent

  using node_matrix_t = std::vector<std::vector<nn_node_t>>;
  using weight_matrix_t = std::vector<std::vector<std::vector<double>>>;

  node_matrix_t nodes;  // vector of nodes in each layer
  weight_matrix_t w;    // weight matrix
  weight_matrix_t dLdw; // delta of loss function L depending on weights w
                        // (used for backpropagation)
  // indexing: l_idx = layer - 1 (layer starts a 0, weights at 1)
  // w[l_idx][to_node in l][from_node in l-1]
  // layout optimized for scalar product in sum over activations coming from
  // nodes in previous layer: sum of w[l][to][from]*nodes[l-1][from].o
  // for fast calculation of forward pass in applications of trained net

  int total_num_weights; // total number of weights in network

  neural_net(nn_structure_t structure_input);

  void forward_pass(std::vector<double> &input_vec);
  std::vector<double> forward_pass_with_output(std::vector<double> &input_vec);

  void backward_pass(std::vector<double> &input_vec,
                     std::vector<double> &target_vec);

  void reset_dLdw_and_dLdb_to_zero();
  void update_w_and_b(double learn_rate, int num_samples);
  double get_partial_loss(std::vector<double> &target_vec);

  void train(f_data_t &fd, f_data_t &td, nn_training_meta_data_t m_data);
};

#endif // NEURAL_NET_HPP