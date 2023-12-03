#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP

#include "neural_net_func.hpp"

#include <cstddef> // std::size_t
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

    std::vector<std::size_t> net_structure; // number of nodes for each layer

    af_t af_h{af_t::sigmoid};  // activation function for hidden layers
    af_t af_o{af_t::identity}; // activation function for output layer

    lf_t lossf{lf_t::MSE}; // loss function for neural net

    lfaf_f1_t lossf_af_f1{
        lfaf_f1_t::MSE_identity}; // combined loss and activation for output layer
};
struct nn_training_meta_data_t {

    std::size_t epochmax;          // max. number of iterations over the whole data set
    std::size_t epoch_output_skip; // output if epoch%epoch_output_skip == 0

    double learning_rate;                 // chosen learning rate for gradient descent
    double min_target_loss;               // minimum loss in training to stop prescribed
                                          // iteration, even if epoch < epochmax
    double min_relative_loss_change_rate; // stop iteration, if loss change
                                          // becomes too small between epochs

    update_strategy_t upstr;     // update strategy for gradient descent
    std::size_t mini_batch_size; // number of training pairs for mini batch
};

struct nn_layer_t {

    // training is done by given pairs (x, y); x and y can be vectors
    // in each layer l=0..num_layers-1 there are num_nodes[l] node entries

    std::vector<double> z; // input side of nodes
    // either gets direct input (for input layer)
    // or gets weighted sum of inputs of previous layer + bias
    // z[to] = sum_from( layer[l].w[to][from]*.layer[l-1].a[from] ) + b[to]

    std::vector<double> b;    // bias value of nodes
    std::vector<double> dLdb; // gradient of bias of nodes

    std::vector<double> a; // output side of nodes (=activation)
                           // gets its value after applying the
                           // activation function a = af(z)

    std::vector<double> delta; // storage for backpropagation

    af_ptr_t af; // ptr to activation function

    using weight_matrix_t = std::vector<std::vector<double>>;
    weight_matrix_t w;    // weight matrix
    weight_matrix_t dLdw; // delta of loss function L depending on weights w
                          // w[to_node in l][from_node in l-1]
    // layout optimized for scalar product in sum over activations coming from
    // nodes in previous layer: sum of layer[l].w[to][from]*layer[l-1][from].a
    // for fast calculation of forward pass in applications of trained net
};

struct neural_net {

    nn_structure_t m_structure;

    std::size_t num_layers; // number of layers in neural_net:
                            // l == 0: input layer
                            // (l > 1) && (l < num_layers - 1): hidden layer
                            // l == num_layers-1: output layer

    std::vector<std::size_t> num_nodes; // number of nodes in each layer
    std::size_t total_num_nodes{0};     // total number of nodes
    std::size_t total_num_bias{0};      // total number of adjustable bias
                                        // values for gradient descent
    std::size_t total_num_weights{0};   // total number of weights in network

    std::vector<nn_layer_t> layer;

    lf_ptr_t lossf; // ptr to loss function

    neural_net(nn_structure_t structure_input);
    void set_w_and_b_fixed(double val);

    void forward_pass(std::vector<double> const& input_vec);
    std::vector<double> forward_pass_with_output(std::vector<double> const& input_vec);

    void backward_pass(std::vector<double> const& input_vec,
                       std::vector<double> const& target_vec);

    void reset_dLdw_and_dLdb_to_zero();
    void update_w_and_b(double learn_rate, std::size_t num_samples);
    double get_partial_loss(std::vector<double> const& target_vec);

    void train(f_data_t const& fd_train, f_data_t const& td_train,
               nn_training_meta_data_t m_data);

    void test(f_data_t const& fd_test, f_data_t const& td_test);
};

#endif // NEURAL_NET_HPP