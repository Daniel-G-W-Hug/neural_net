#ifndef READ_DATA_HPP
#define READ_DATA_HPP

#include "neural_net.hpp"

#include <fstream>
#include <sstream>
#include <string_view>
#include <tuple>
#include <vector>

std::tuple<int, std::stringstream> get_next_line(std::ifstream &ifs);
nn_meta_data_t read_cfg(std::string_view fname);
f_data_t read_f_data(std::string_view fname, int assert_size);
void print_f_data(std::string_view tag, f_data_t &fd);

#endif // READ_DATA_HPP