#ifndef NEURAL_NET_FILE_IO_HPP
#define NEURAL_NET_FILE_IO_HPP

#include "neural_net.hpp"

#include <fstream>
#include <sstream>
#include <string_view>
#include <tuple>
#include <vector>

std::tuple<std::size_t, std::stringstream> get_next_line(std::ifstream &ifs);
std::tuple<nn_structure_t, nn_training_meta_data_t>
read_training_cfg(std::string_view fname);
f_data_t read_f_data(std::string_view fname, std::size_t assert_size);
void print_f_data(std::string_view tag, f_data_t &fd);

#endif // NEURAL_NET_FILE_IO_HPP