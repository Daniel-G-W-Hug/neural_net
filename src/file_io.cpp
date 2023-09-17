#include "file_io.hpp"

#include <iostream>
#include <mutex> // once_flag
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

std::tuple<std::size_t, std::stringstream> get_next_line(std::ifstream &ifs) {

  // for reading the file
  std::string line;
  static std::size_t line_number;

  // read line by line (ignore comment lines)
  while (std::getline(ifs, line)) {

    if (line[0] == '#') {
      // ignore comment lines starting with # in 1st column
      // std::cout << "Found comment - line ignored.\n";
      ++line_number;
      continue;
    }

    ++line_number;

    std::stringstream iss(line);
    // std::cout << "line " << line_number << ": " << line << std::endl;

    // std::stringstream does not have a copy-ctor, so it must be moved
    return std::make_tuple(line_number, std::move(iss));
  }
  throw std::runtime_error(
      "Could not find next line in file. Got stuck at line " +
      std::to_string(line_number) + ".\n");
}

std::tuple<nn_structure_t, nn_training_meta_data_t>
read_training_cfg(std::string_view fname) {

  nn_structure_t m_structure;
  nn_training_meta_data_t m_data;

  std::ifstream ifs(fname);
  std::size_t line_no;
  std::stringstream iss;

  int int_in;
  double double_in;
  char ch_in;

  if (ifs.is_open()) {

    // read network structure
    std::tie(line_no, iss) = get_next_line(ifs);
    while (iss >> int_in) {
      // read int values as long as available (must be comma separated ints!)
      if (int_in > 0) {
        m_structure.net_structure.push_back(int_in);
      } else {
        throw std::runtime_error("Positive number of nodes required in each "
                                 "layer. Wrong input in line " +
                                 std::to_string(line_no) + ".\n");
      }

      iss >> ch_in; // read and skip comma and whitespace
    }
    // std::cout << "m_data.net_structure.size() = " <<
    // m_data.net_structure.size()
    //           << "\n";

    // read activation function for hidden layers
    std::tie(line_no, iss) = get_next_line(ifs);
    iss >> int_in;
    if (int_in > 0 && int_in <= 5) {
      m_structure.af_h = static_cast<a_func_t>(int_in);
    } else {
      throw std::runtime_error(
          "Wrong enum value for activation function for hidden layers: " +
          std::to_string(int_in) + " in line " + std::to_string(line_no) +
          ".\n");
    }

    // read activation function for output layer
    std::tie(line_no, iss) = get_next_line(ifs);
    iss >> int_in;
    if (int_in > 0 && int_in <= 5) {
      m_structure.af_o = static_cast<a_func_t>(int_in);
    } else {
      throw std::runtime_error(
          "Wrong enum value for activation function for output layer: " +
          std::to_string(int_in) + " in line " + std::to_string(line_no) +
          ".\n");
    }

    // read epochmax
    std::tie(line_no, iss) = get_next_line(ifs);
    iss >> int_in;
    if (int_in > 0) {
      m_data.epochmax = int_in;
    } else {
      throw std::runtime_error("epochmax must be a positive value. Line " +
                               std::to_string(line_no) + ".\n");
    }

    // read epoch_output_skip
    std::tie(line_no, iss) = get_next_line(ifs);
    iss >> int_in;
    if (int_in > 0) {
      m_data.epoch_output_skip = int_in;
    } else {
      throw std::runtime_error(
          "epoch_output_skip must be a positive value. Line " +
          std::to_string(line_no) + ".\n");
    }

    // read learning_rate
    std::tie(line_no, iss) = get_next_line(ifs);
    iss >> double_in;
    if (double_in > 0.0) {
      m_data.learning_rate = double_in;
    } else {
      throw std::runtime_error("learning rate must be a positive value. Line " +
                               std::to_string(line_no) + ".\n");
    }

    // read min_target_loss
    std::tie(line_no, iss) = get_next_line(ifs);
    iss >> double_in;
    if (double_in > 0.0) {
      m_data.min_target_loss = double_in;
    } else {
      throw std::runtime_error(
          "min_target_loss must be a positive value. Line " +
          std::to_string(line_no) + ".\n");
    }

    // read min_relative_loss_change_rate
    std::tie(line_no, iss) = get_next_line(ifs);
    iss >> double_in;
    if (double_in > 0.0) {
      m_data.min_relative_loss_change_rate = double_in;
    } else {
      throw std::runtime_error(
          "min_relative_loss_change_rate must be a positive value. Line " +
          std::to_string(line_no) + ".\n");
    }

    // read update strategy
    std::tie(line_no, iss) = get_next_line(ifs);
    iss >> int_in;
    if (int_in > 0 && int_in <= 3) {
      m_data.upstr = static_cast<update_strategy_t>(int_in);
    } else {
      throw std::runtime_error(
          "Wrong enum value for update strategy: " + std::to_string(int_in) +
          " in line " + std::to_string(line_no) + ".\n");
    }

    // read mini_batch_size
    std::tie(line_no, iss) = get_next_line(ifs);
    iss >> int_in;
    if (int_in > 0) {
      m_data.mini_batch_size = int_in;
    } else {
      throw std::runtime_error(
          "mini_batch_size must be a positive value. Line " +
          std::to_string(line_no) + ".\n");
    }

    ifs.close();

  } else {
    throw std::runtime_error("Failed to open file: " + std::string(fname));
  }

  return std::make_tuple(m_structure, m_data);
}

f_data_t read_f_data(std::string_view fname, std::size_t assert_size) {
  // read file consisting of rows of csv data (same amount of data in each
  // row)

  f_data_t file_data;
  std::ifstream ifs(fname);

  if (ifs.is_open()) {

    // for detection of different number of items per line
    std::once_flag flag;
    std::size_t items_per_line_current, items_per_line;

    // for reading the file
    std::string line;
    double in;
    char ch;
    std::size_t line_number{0};

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

void print_f_data(std::string_view tag, f_data_t &fd) {

  std::cout << "'" << tag << "':" << std::endl;

  std::size_t line_cnt{0};

  for (auto &line : fd) {
    ++line_cnt;

    std::cout << "  " << line_cnt << ": ";
    std::size_t items = line.size();
    for (std::size_t i = 0; i < items; ++i) {
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