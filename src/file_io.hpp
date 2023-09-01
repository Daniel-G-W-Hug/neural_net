#ifndef READ_DATA_HPP
#define READ_DATA_HPP

#include <string_view>
#include <vector>

// read csv files or raw data organized in lines with constant number of integer
// or double values stored in f_data_t

using f_data_t = std::vector<std::vector<double>>;

f_data_t read_f_data(std::string_view fname, int assert_size);
void print_f_data(std::string_view tag, f_data_t &fd);

#endif // READ_DATA_HPP