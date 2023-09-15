#ifndef NEURAL_NET_PRINT_HPP
#define NEURAL_NET_PRINT_HPP

#include "neural_net.hpp"

#include <string_view>

void print_parameters(std::string_view tag, neural_net const& nn);
void print_nodes(std::string_view tag, neural_net const& nn);
void print_weights(std::string_view tag, neural_net const& nn);

#endif // NEURAL_NET_PRINT_HPP