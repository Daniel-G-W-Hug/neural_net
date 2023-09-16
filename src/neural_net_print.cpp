#include "neural_net_print.hpp"

#include <iostream>

void print_parameters(std::string_view tag, neural_net const &nn) {
  std::cout << "'" << tag << "' neural network with " << nn.num_layers
            << " layers:" << std::endl;

  // print number of nodes with user provided sizes only (leave out bias
  // nodes)
  for (int l = 0; l < nn.num_layers; ++l) {
    std::cout << "layer " << l << " : " << nn.num_nodes[l] - 1 << " nodes"
              << std::endl;
  }
  std::cout << "number of nodes (w/o extra nodes for bias calculation): "
            << nn.total_num_nodes - nn.num_layers << std::endl;
  std::cout << "total number of weights: " << nn.total_num_weights << std::endl;
  std::cout << "thereof number of weights for bias values: "
            << nn.total_num_bias << std::endl;
  std::cout << "+------------------------------------------------------"
               "-------+"
            << std::endl;

  return;
}

void print_nodes(std::string_view tag, neural_net const &nn) {
  for (int l = 0; l < nn.num_layers; ++l) {
    std::cout << "'" << tag << "' - nodes layer " << l;
    if (l == 0) {
      std::cout << " (input layer):";
    } else if (l == nn.num_layers - 1) {
      std::cout << " (ouput layer):";
    } else {
      std::cout << " (hidden layer):";
    }
    std::cout << std::endl;
    for (int n = 0; n < nn.num_nodes[l]; ++n) {
      std::cout << "  n: " << n << " nodes[" << l << "][" << n << "].z = ";
      std::cout.precision(5);
      std::cout << nn.nodes[l][n].z << ", .a = " << nn.nodes[l][n].a
                << ", .delta = " << nn.nodes[l][n].delta;
      // std::cout << " &af = " << (void *)nodes[l][n].af;
      std::cout << std::endl;
    }
    if (l < nn.num_layers - 1) {
      std::cout << std::endl;
    } else {
      std::cout << "+--------------------------------------------------"
                   "-----------+"
                << std::endl;
    }
  }
  // std::cout << std::endl;
}

void print_weights(std::string_view tag, neural_net const &nn) {
  for (int l = 1; l < nn.num_layers; ++l) {
    std::cout << "'" << tag << "' - weights layer " << l;
    if (l == nn.num_layers - 1) {
      std::cout << " (ouput layer):";
    } else {
      std::cout << " (hidden layer):";
    }
    std::cout << std::endl;
    int l_idx = l - 1; // index transformation for weights (start index 0)

    // show to user as index l, while internally using the index
    // transformation
    for (int to = 0; to < nn.num_nodes[l] - 1; ++to) {
      for (int from = 0; from < nn.num_nodes[l - 1]; ++from) {
        std::cout << "    w[" << l << "][" << to << "][" << from << "] = ";
        std::cout.precision(5);
        if (from == nn.num_nodes[l - 1] - 1) {
          std::cout << nn.w[l_idx][to][from] << " (for bias)" << std::endl;
        } else {
          std::cout << nn.w[l_idx][to][from] << std::endl;
        }
      }
    }
    std::cout << std::endl;
    for (int to = 0; to < nn.num_nodes[l] - 1; ++to) {
      for (int from = 0; from < nn.num_nodes[l - 1]; ++from) {
        std::cout << "    dLdw[" << l << "][" << to << "][" << from << "] = ";
        std::cout.precision(5);
        if (from == nn.num_nodes[l - 1] - 1) {
          std::cout << nn.dLdw[l_idx][to][from] << " (for bias)" << std::endl;
        } else {
          std::cout << nn.dLdw[l_idx][to][from] << std::endl;
        }
      }
    }
    if (l < nn.num_layers - 1) {
      std::cout << std::endl;
    } else {
      std::cout << "+--------------------------------------------------"
                   "-----------+"
                << std::endl;
    }
  }
}