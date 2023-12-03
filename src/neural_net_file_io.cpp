#include "neural_net_file_io.hpp"

#include <cassert>
#include <iostream>
#include <mutex> // once_flag
#include <stdexcept>
#include <utility> // std::unreachable()
#include <vector>

std::tuple<std::size_t, std::stringstream> get_next_line(std::ifstream& ifs)
{

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
    throw std::runtime_error("Could not find next line in file. Got stuck at line " +
                             std::to_string(line_number) + ".\n");
}

std::tuple<nn_structure_t, nn_training_meta_data_t>
read_training_cfg(std::string const& fname)
{

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
            }
            else {
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
            m_structure.af_h = static_cast<af_t>(int_in);
        }
        else {
            throw std::runtime_error(
                "Wrong enum value for activation function for hidden layers: " +
                std::to_string(int_in) + " in line " + std::to_string(line_no) + ".\n");
        }

        // read combination of loss function and activation function for output layer
        std::tie(line_no, iss) = get_next_line(ifs);
        iss >> int_in;
        if (int_in > 0 && int_in <= 3) {
            m_structure.lossf_af_f1 = static_cast<lfaf_f1_t>(int_in);
            switch (m_structure.lossf_af_f1) {
                case (lfaf_f1_t::MSE_identity):
                    m_structure.af_o = af_t::identity;
                    m_structure.lossf = lf_t::MSE;
                    break;
                case (lfaf_f1_t::MSE_sigmoid):
                    m_structure.af_o = af_t::sigmoid;
                    m_structure.lossf = lf_t::MSE;
                    break;
                case (lfaf_f1_t::CE_softmax):
                    m_structure.af_o = af_t::softmax;
                    m_structure.lossf = lf_t::CE;
                    break;
                default:
                    std::unreachable();
                    break;
            }
        }
        else {
            throw std::runtime_error(
                "Wrong enum value for activation function for output layer: " +
                std::to_string(int_in) + " in line " + std::to_string(line_no) + ".\n");
        }

        // read epochmax
        std::tie(line_no, iss) = get_next_line(ifs);
        iss >> int_in;
        if (int_in > 0) {
            m_data.epochmax = int_in;
        }
        else {
            throw std::runtime_error("epochmax must be a positive value. Line " +
                                     std::to_string(line_no) + ".\n");
        }

        // read epoch_output_skip
        std::tie(line_no, iss) = get_next_line(ifs);
        iss >> int_in;
        if (int_in > 0) {
            m_data.epoch_output_skip = int_in;
        }
        else {
            throw std::runtime_error("epoch_output_skip must be a positive value. Line " +
                                     std::to_string(line_no) + ".\n");
        }

        // read learning_rate
        std::tie(line_no, iss) = get_next_line(ifs);
        iss >> double_in;
        if (double_in > 0.0) {
            m_data.learning_rate = double_in;
        }
        else {
            throw std::runtime_error("learning rate must be a positive value. Line " +
                                     std::to_string(line_no) + ".\n");
        }

        // read min_target_loss
        std::tie(line_no, iss) = get_next_line(ifs);
        iss >> double_in;
        if (double_in > 0.0) {
            m_data.min_target_loss = double_in;
        }
        else {
            throw std::runtime_error("min_target_loss must be a positive value. Line " +
                                     std::to_string(line_no) + ".\n");
        }

        // read min_relative_loss_change_rate
        std::tie(line_no, iss) = get_next_line(ifs);
        iss >> double_in;
        if (double_in > 0.0) {
            m_data.min_relative_loss_change_rate = double_in;
        }
        else {
            throw std::runtime_error(
                "min_relative_loss_change_rate must be a positive value. Line " +
                std::to_string(line_no) + ".\n");
        }

        // read update strategy
        std::tie(line_no, iss) = get_next_line(ifs);
        iss >> int_in;
        if (int_in > 0 && int_in <= 3) {
            m_data.upstr = static_cast<update_strategy_t>(int_in);
        }
        else {
            throw std::runtime_error(
                "Wrong enum value for update strategy: " + std::to_string(int_in) +
                " in line " + std::to_string(line_no) + ".\n");
        }

        // read mini_batch_size
        std::tie(line_no, iss) = get_next_line(ifs);
        iss >> int_in;
        if (int_in > 0) {
            m_data.mini_batch_size = int_in;
        }
        else {
            throw std::runtime_error("mini_batch_size must be a positive value. Line " +
                                     std::to_string(line_no) + ".\n");
        }

        ifs.close();
    }
    else {
        throw std::runtime_error("Failed to open file: " + std::string(fname));
    }

    return std::make_tuple(m_structure, m_data);
}

f_data_t read_f_data(std::string const& fname, std::size_t assert_size)
{
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
                    "Different number of items per line in file: " + std::string(fname) +
                    ", line " + std::to_string(line_number));
            }
        }

        // assure fit to required number of elements
        // either number of input or number of output nodes
        assert(items_per_line == assert_size);

        ifs.close();
    }
    else {
        throw std::runtime_error("Failed to open file: " + std::string(fname));
    }

    return file_data;
}

void print_f_data(std::string const& tag, f_data_t& fd)
{

    std::cout << "'" << tag << "':" << std::endl;

    std::size_t line_cnt{0};

    for (auto& line : fd) {
        ++line_cnt;

        std::cout << "  " << line_cnt << ": ";
        std::size_t items = line.size();
        for (std::size_t i = 0; i < items; ++i) {
            if (i < items - 1) {
                std::cout << line[i] << ", ";
            }
            else {
                std::cout << line[i] << std::endl;
            }
        }
    }
    //   std::cout << std::endl;
    std::cout << "+-------------------------------------------------------------+"
              << std::endl;

    return;
}

std::tuple<f_data_t, f_data_t> read_mnist_data(std::string const& fname,
                                               std::size_t num_output_nodes)
{
    // read file with mnist data with first column containing target data
    //
    // (i.e. having target value 0..9 in column 1 for the MNIST character data set
    // and column 2..785 consisting of 28x28 = 784 grey values in range [0..255])
    // (same amount of data in each row)
    //
    // for num_output_nodes > 1:
    // the target data will be transformed into an output_vector with num_output_nodes
    // elements that serve as ground truth values (i.e. = 1.0 at corresponding index and
    // otherwise 0.0)
    //
    // for num_output_nodes == 1:
    // output target value is contained in target_data[0]

    if (num_output_nodes == 0) {
        // there must be at least one output node
        throw std::runtime_error("Config error: there must be at least one output node.");
    }

    f_data_t training_data;
    f_data_t target_data;

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

            // make sure that every line has the same number of items
            // (first line in file defines expected number of items per line)
            call_once(flag, [&]() { items_per_line = line_data.size(); });

            items_per_line_current = line_data.size();
            if (items_per_line_current != items_per_line) {
                throw std::runtime_error(
                    "Different number of items per line in file: " + std::string(fname) +
                    ", line " + std::to_string(line_number));
            }

            if (num_output_nodes > 1 &&
                (line_data[0] < 0.0 || line_data[0] >= num_output_nodes)) {
                // Just check for case of more than one output node.
                // In case one output node only, there might be more than one target
                // output state.
                throw std::runtime_error(
                    "Potential ground truth value " + std::to_string(line_data[0]) +
                    " in file: " + std::string(fname) + ", line " +
                    std::to_string(line_number) + " does not match expected range.");
            }
            // ATTENTION: This only works reasonably for integer input data
            std::size_t idx = static_cast<std::size_t>(line_data[0]);

            std::vector<double> tmp(num_output_nodes, 0.0);
            if (num_output_nodes > 1) {
                tmp[idx] = 1.0; // Use target value from 1st column of input file as index
                                // for ground truth value 1.0. Other values are set to 0.0
                                // during initialization.
            }
            else {
                tmp[0] = line_data[0];
            }
            target_data.push_back(tmp);

            tmp.clear();
            // the remaining line contains the training data
            for (std::size_t cnt = 1; cnt < items_per_line; ++cnt) {
                tmp.push_back(line_data[cnt]);
            }
            training_data.push_back(tmp);
        }

        ifs.close();
    }
    else {
        throw std::runtime_error("Failed to open file: " + std::string(fname));
    }

    return std::make_tuple(training_data, target_data);
}