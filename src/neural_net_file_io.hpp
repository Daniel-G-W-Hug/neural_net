#ifndef NEURAL_NET_FILE_IO_HPP
#define NEURAL_NET_FILE_IO_HPP

#include "neural_net.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

std::tuple<std::size_t, std::stringstream> get_next_line(std::ifstream& ifs);
std::tuple<nn_structure_t, nn_training_meta_data_t>
read_training_cfg(std::string const& fname);
f_data_t read_f_data(std::string const& fname, std::size_t assert_size);
void print_f_data(std::string const& tag, f_data_t& fd);

std::tuple<f_data_t, f_data_t> read_mnist_data(std::string const& fname);

#endif // NEURAL_NET_FILE_IO_HPP