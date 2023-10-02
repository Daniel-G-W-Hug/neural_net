#include "neural_net_print.hpp"

#include <iomanip>
#include <iostream>

void print_parameters(std::string_view tag, neural_net const& nn)
{
    std::cout << "'" << tag << "' neural network with " << nn.num_layers
              << " layers:" << std::endl;

    // print number of nodes with user provided sizes only (leave out bias
    // nodes)
    for (std::size_t l = 0; l < nn.num_layers; ++l) {
        std::cout << "layer " << l << " : " << nn.num_nodes[l] << " nodes" << std::endl;
    }
    std::cout << "number of nodes: " << nn.total_num_nodes << std::endl;
    std::cout << "total number of weights: " << nn.total_num_weights << std::endl;
    std::cout << "thereof number bias values: " << nn.total_num_bias << std::endl;
    std::cout << "total number of training parameters: "
              << nn.total_num_weights + nn.total_num_bias << std::endl;
    std::cout << "+------------------------------------------------------"
                 "-------+"
              << std::endl;

    return;
}

void print_nodes(std::string_view tag, neural_net const& nn)
{
    for (std::size_t l = 0; l < nn.num_layers; ++l) {
        std::cout << "'" << tag << "' - nodes layer " << l;
        if (l == 0) {
            std::cout << " (input layer):";
        }
        else if (l == nn.num_layers - 1) {
            std::cout << " (ouput layer):";
        }
        else {
            std::cout << " (hidden layer):";
        }
        std::cout << std::endl;
        for (std::size_t n = 0; n < nn.num_nodes[l]; ++n) {
            if (l == 0) {
                // only some arrays are initialized in input layer

                std::cout.precision(6);
                std::cout << "  n: " << n << " z[" << n << "] = ";
                std::cout << std::setw(12) << nn.layer[l].z[n];
                std::cout << ", a[" << n << "] = ";
                std::cout << std::setw(12) << nn.layer[l].a[n];
                std::cout << std::endl;
            }
            else {
                std::cout.precision(6);
                std::cout << "  n: " << n << " z[" << n << "] = ";
                std::cout << std::setw(12) << nn.layer[l].z[n];
                std::cout << ", b[" << n << "] = ";
                std::cout << std::setw(12) << nn.layer[l].b[n];
                std::cout << ", dLdb[" << n << "] = ";
                std::cout << std::setw(12) << nn.layer[l].dLdb[n];
                std::cout << ", a[" << n << "] = ";
                std::cout << std::setw(12) << nn.layer[l].a[n];
                std::cout << ", delta[" << n << "] = ";
                std::cout << std::setw(12) << nn.layer[l].delta[n];
                std::cout << std::endl;
            }
        }
        if (l < nn.num_layers - 1) {
            std::cout << std::endl;
        }
        else {
            std::cout << "+--------------------------------------------------"
                         "-----------+"
                      << std::endl;
        }
    }
    // std::cout << std::endl;
}

void print_weights(std::string_view tag, neural_net const& nn)
{
    for (std::size_t l = 1; l < nn.num_layers; ++l) {
        std::cout << "'" << tag << "' - weights layer " << l;
        if (l == nn.num_layers - 1) {
            std::cout << " (ouput layer):";
        }
        else {
            std::cout << " (hidden layer):";
        }
        std::cout << std::endl;

        // show to user as index l, while internally using the index
        // transformation
        for (std::size_t to = 0; to < nn.num_nodes[l]; ++to) {
            for (std::size_t from = 0; from < nn.num_nodes[l - 1]; ++from) {
                std::cout << "    w[" << to << "][" << from << "]    = ";
                std::cout.precision(6);
                std::cout << std::setw(12) << nn.layer[l].w[to][from] << std::endl;
            }
        }
        std::cout << std::endl;
        for (std::size_t to = 0; to < nn.num_nodes[l]; ++to) {
            for (std::size_t from = 0; from < nn.num_nodes[l - 1]; ++from) {
                std::cout << "    dLdw[" << to << "][" << from << "] = ";
                std::cout.precision(6);
                std::cout << std::setw(12) << nn.layer[l].dLdw[to][from] << std::endl;
            }
        }
        if (l < nn.num_layers - 1) {
            std::cout << std::endl;
        }
        else {
            std::cout << "+--------------------------------------------------"
                         "-----------+"
                      << std::endl;
        }
    }
}