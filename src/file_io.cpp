#include "file_io.hpp"

#include <fstream>
#include <iostream>
#include <mutex> // once_flag
#include <sstream>
#include <stdexcept>
#include <string>

f_data_t read_training_data(std::string_view fname, int assert_size) {
  // read file consisting of rows of csv data (same amount of data in each row)

  f_data_t file_data;
  std::ifstream ifs(fname);

  if (ifs.is_open()) {

    // for detection of different number of items per line
    std::once_flag flag;
    int items_per_line_current, items_per_line;

    // for reading the file
    std::string line;
    double in;
    char ch;
    int line_number{0};

    // read line by line
    while (std::getline(ifs, line)) {

      if (line[0] == '#') {
        // ignore comment lines
        // std::cout << "Found comment - line ignored.\n";
        continue;
      }

      std::stringstream iss(line);
      // std::cout << "line " << line << std::endl;

      ++line_number;

      std::vector<double> line_data;

      while (iss >> in) {
        // read double values as long as available
        line_data.push_back(in);
        iss >> ch; // read space or comma
      }

      file_data.push_back(line_data);

      // make sure that every line has the same number of items
      // (first line in file defines expected number of items per line)
      call_once(flag, [&]() { items_per_line = line_data.size(); });

      items_per_line_current = line_data.size();
      if (items_per_line_current != items_per_line) {
        throw std::runtime_error(
            "Different number of items per line in file: " +
            std::string(fname) + ", line " + std::to_string(line_number));
      }
    }

    // assure fit to required number of elements
    // either number of input or number of output nodes
    assert(items_per_line == assert_size);

    ifs.close();

  } else {
    throw std::runtime_error("Failed to open file: " + std::string(fname));
  }

  return file_data;
}

void print_data(std::string_view tag, f_data_t &fd) {

  std::cout << "'" << tag << "':" << std::endl;

  int line_cnt{0};

  for (auto &line : fd) {
    ++line_cnt;

    std::cout << "  " << line_cnt << ": ";
    int items = line.size();
    for (int i = 0; i < items; ++i) {
      if (i < items - 1) {
        std::cout << line[i] << ", ";
      } else {
        std::cout << line[i] << std::endl;
      }
    }
  }
  //   std::cout << std::endl;
  std::cout << "+-------------------------------------------------------------+"
            << std::endl;

  return;
}
