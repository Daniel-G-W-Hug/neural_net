// simple implementation of neural network based on perceptrons with
// backpropagation
//
// based on: https://github.com/gokadin/ai-backpropagation#code-example
//
// Daniel Hug, 08/2023

#include "file_io.hpp"
#include "neural_net.hpp"

// uncomment to disable assert()
// #define NDEBUG
#include <cassert>
#include <exception>
#include <iostream>
#include <vector>

int main() {

  try {

    // std::vector net_structure = {2, 1}; // nodes per layer

    // std::vector net_structure = {2, 2, 1}; // nodes per layer

    std::vector net_structure = {4, 3, 1}; // nodes per layer

    neural_net nn(net_structure, &sigmoid);

    // std::cout << "neural_net nodes not yet activated." << std::endl;
    nn.print_parameters("nn");
    // nn.print_nodes("nn");
    // nn.print_weights("nn");

    // f_data_t fd = read_training_data(
    //     "../input/2x1_linear_classify_training_data.csv", nn.num_nodes[0]);
    // print_data("training data", fd);
    // f_data_t td =
    //     read_training_data("../input/2x1_linear_classify_target_data.csv",
    //                        nn.num_nodes[nn.num_layers - 1]);
    // print_data("training target data", td);

    // f_data_t fd = read_training_data("../input/2x2x1_xor_training_data.csv",
    //                                  nn.num_nodes[0]);
    // print_data("training data", fd);
    // f_data_t td = read_training_data("../input/2x2x1_xor_target_data.csv",
    //                                  nn.num_nodes[nn.num_layers - 1]);
    // print_data("training target data", td);

    f_data_t fd =
        read_training_data("../input/iris_training_data.csv", nn.num_nodes[0]);
    print_data("training data", fd);
    f_data_t td = read_training_data("../input/iris_target_data.csv",
                                     nn.num_nodes[nn.num_layers - 1]);
    print_data("training target data", td);

    std::cout << "\n\nStart training cycle...\n\n";
    nn.train(fd, td);
    std::cout << "\n\nStop training cycle...\n\n";
    nn.print_nodes("nn");
    nn.print_weights("nn");

  } catch (std::exception &e) {
    std::cout << "Exception: " << e.what() << std::endl;
  }

  return 0;
}
