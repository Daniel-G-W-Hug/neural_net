// simple implementation of neural network
//
// Daniel Hug, 08/2023

#include "file_io.hpp"
#include "neural_net.hpp"

// uncomment to disable assert()
// #define NDEBUG
#include <cassert>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char *argv[]) {

  try {

    // TODO: chose update strategy for training as input parameter
    // TODO: chose case names as input arguments
    // TODO: learn_rate in nn.train must be a selectable parameter

    std::string fname_training, fname_target;

    std::vector net_structure = {2, 1}; // nodes per layer
    fname_training = "../input/2x1_linear_classify_training_data.csv";
    fname_target = "../input/2x1_linear_classify_target_data.csv";

    // // std::vector net_structure = {2, 2, 1}; // nodes per layer
    // fname_training = "../input/2x2x1_xor_training_data.csv";
    // fname_target = "../input/2x2x1_xor_target_data.csv";

    // // std::vector net_structure = {4, 3, 1}; // nodes per layer
    // fname_training = "../input/iris_training_data.csv";
    // fname_target = "../input/iris_target_data.csv";

    neural_net nn(net_structure, &sigmoid);

    f_data_t fd = read_f_data(fname_training, nn.num_nodes[0]);
    f_data_t td = read_f_data(fname_target, nn.num_nodes[nn.num_layers - 1]);

    print_f_data("training data", fd);
    print_f_data("training target data", td);

    // std::cout << "neural_net nodes not yet activated." << std::endl;
    nn.print_parameters("nn");
    // nn.print_nodes("nn");
    // nn.print_weights("nn");

    std::cout << "\nStart training cycle...\n\n";

    nn.train(fd, td, update_strategy_t::immediate_update);
    // nn.train(fd, td, update_strategy_t::batch_update);

    std::cout << "Stop training cycle...\n\n\n";

    nn.print_nodes("nn");
    nn.print_weights("nn");

  } catch (std::exception &e) {
    std::cout << "Exception: " << e.what() << std::endl;
  }

  return 0;
}
