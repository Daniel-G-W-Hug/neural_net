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

// TODO: export/import of trained values for w and b
// TODO: make random distribution in neural_net.cpp a selectable meta parameter

int main(int argc, char *argv[]) {

  try {

    if (argc != 2) {
      std::cout << "Provide a 'case_name' as an argument, please.\n\n";
      std::cout << "Input file names will be derived from 'case_name':\n\n";
      std::cout << "  - 'case_name.cfg' -> config file.\n";
      std::cout << "  - 'case_name_training_data.csv' -> training data set.\n";
      std::cout << "  - 'case_name_target_data.csv' -> target data set.\n\n";
      throw std::runtime_error(
          "Failed to provide required case_name as argument.\n");
    }
    std::string case_name(argv[1]);

    std::string f_cfg{"../input/" + case_name + ".cfg"};
    std::string f_training{"../input/" + case_name + "_training_data.csv"};
    std::string f_target{"../input/" + case_name + "_target_data.csv"};

    nn_meta_data_t nn_meta = read_cfg(f_cfg);

    neural_net nn(nn_meta);

    f_data_t fd = read_f_data(f_training, nn.num_nodes[0]);
    f_data_t td = read_f_data(f_target, nn.num_nodes[nn.num_layers - 1]);

    print_f_data("training data", fd);
    print_f_data("training target data", td);

    // std::cout << "neural_net nodes not yet activated." << std::endl;
    nn.print_parameters("nn");
    nn.print_nodes("nn");
    nn.print_weights("nn");

    std::cout << "\nStart training cycle...\n\n";
    nn.train(fd, td);
    std::cout << "\nStop training cycle...\n\n\n";

    nn.print_nodes("nn");
    nn.print_weights("nn");

    std::cout << "Prediction with trained network:";

    // // for 2x2x1_example
    // std::vector<double> inp1{-7., -3.}, inp2{20., 2.}, outp;
    // outp = nn.forward_pass_with_output(inp1);
    // std::cout << "inp1 => " << outp[0] << std::endl;
    // outp = nn.forward_pass_with_output(inp2);
    // std::cout << "inp2 => " << outp[0] << std::endl;

    // // for 2x2x1_xor
    // std::vector<double> inp1{1., 0.}, inp2{1., 1.}, outp;
    // outp = nn.forward_pass_with_output(inp1);
    // std::cout << "inp1 => " << outp[0] << std::endl;
    // outp = nn.forward_pass_with_output(inp2);
    // std::cout << "inp2 => " << outp[0] << std::endl;

    // for iris_example
    std::vector<double> inp1{4.8, 3.01, 1.45, 0.15},
        inp2{6.31, 2.29, 4.45, 1.3}, outp;
    outp = nn.forward_pass_with_output(inp1);
    std::cout << "inp1 => " << outp[0] << std::endl;
    outp = nn.forward_pass_with_output(inp2);
    std::cout << "inp2 => " << outp[0] << std::endl;

  } catch (std::exception &e) {
    std::cout << "Exception: " << e.what() << std::endl;
  }

  return 0;
}
