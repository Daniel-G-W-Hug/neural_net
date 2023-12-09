// simple implementation of neural network
//
// Daniel Hug, 08/2023

#include "neural_net.hpp"
#include "neural_net_file_io.hpp"
#include "neural_net_print.hpp"

// uncomment to disable assert()
// #define NDEBUG
#include <algorithm> // for_each
#include <cassert>
#include <exception>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "fmt/format.h"
#include "fmt/ranges.h" // support printing of (nested) containers & tuples

// TODO: export/import of trained values for w and b
// TODO: make random distribution in neural_net.cpp a selectable meta parameter

void print_help_message()
{
    fmt::println("Provide a 'case_name' as single argument, please.\n");
    fmt::println("Remaining file names will be derived from 'case_name':\n");
    fmt::println("  - 'case_name.cfg' -> config file incl. meta data.");
    fmt::println("  - 'case_name_train.csv' -> training data.");
    fmt::println("  - 'case_name_test.csv' -> test data.\n");
    fmt::println("If case_name ends with '_validate', i.e. is "
                 "'case_name_validate' then:\n");
    fmt::println("  - 'case_name_validate.cfg' -> config file incl. meta data.");
    fmt::println("  - 'case_name_train.csv' -> validation data.");
    fmt::println("  - 'case_name_validate.csv' -> test data.\n");
}

std::tuple<std::string, std::string, std::string>
get_file_names(std::string const& prefix, std::string const& case_name)
{

    // create local copy of case_name for modification in case of validation data
    std::string loc_case_name{case_name};

    std::string fn_cfg{prefix + loc_case_name + ".cfg"};
    std::string fn_train;
    std::string fn_test;

    std::string str_to_erase{"_validate"};
    if (loc_case_name.contains(str_to_erase)) {
        fn_test = prefix + loc_case_name + ".csv";
        // remove "_validate", to get correct file name for test data below
        loc_case_name.erase(loc_case_name.find(str_to_erase), str_to_erase.length());
        fn_train = prefix + loc_case_name + "_train.csv";
    }
    else {
        fn_test = prefix + loc_case_name + "_test.csv";
        fn_train = prefix + loc_case_name + "_train.csv";
    }


    return std::make_tuple(fn_cfg, fn_train, fn_test);
}

void scale_data(f_data_t& data, double scale)
{
    for (std::size_t cnt = 0; cnt < data.size(); ++cnt) {
        for_each(data[cnt].begin(), data[cnt].end(),
                 [&scale](double& elem) { elem *= scale; });
    }
}

int main(int argc, char* argv[])
{

    try {

        ///////////////////////////////////////////////////////////////////////
        // evaluate arguments and get file names
        ///////////////////////////////////////////////////////////////////////
        if (argc == 1 || (argc == 2 && std::string(argv[1]).contains("--help"))) {
            print_help_message();
            return 0;
        }

        if (argc != 2) {
            print_help_message();
            throw std::runtime_error(
                "Failed to provide required case_name as argument.\n");
        }

        std::string prefix{"../input/"};
        std::string case_name{std::string(argv[1])};
        auto const [fn_cfg, fn_train, fn_test] = get_file_names(prefix, case_name);

        fmt::println("\nData files:\n");
        fmt::println("Config  : {}", fn_cfg);
        fmt::println("Training: {}", fn_train);
        fmt::println("Test    : {}\n", fn_test);

        ///////////////////////////////////////////////////////////////////////
        // read config, setup neural net and read training, target & test data
        ///////////////////////////////////////////////////////////////////////
        auto const [nn_structure, nn_meta] = read_cfg(fn_cfg);

        neural_net nn(nn_structure);
        // nn.set_w_and_b_fixed(0.1);

        // f_data_t fd = read_f_data(f_training, nn.num_nodes[0]);
        // f_data_t td = read_f_data(f_target, nn.num_nodes[nn.num_layers - 1]);

        auto [fd_train, td_train] =
            read_mnist_data(fn_train, nn.num_nodes[0], nn.num_nodes[nn.num_layers - 1]);
        // print_f_data("training data", fd_train);
        // print_f_data("training target data", td_train);

        auto [fd_test, td_test] =
            read_mnist_data(fn_test, nn.num_nodes[0], nn.num_nodes[nn.num_layers - 1]);
        // print_f_data("test data", fd_test);
        // print_f_data("test target data", td_test);

        if (case_name.contains("mnist")) {
            // scale picture data, i.e. gray scale range
            // from range 0 ... 255 to range 0.0 ... 1.0
            fmt::println("Scaling mnist data\n");
            scale_data(fd_train, 1. / 255.);
            scale_data(fd_test, 1. / 255.);
        }

        // fmt::println("neural_net nodes not yet activated.");
        print_parameters("nn", nn);
        // print_nodes("nn", nn);
        // print_weights("nn", nn);

        ///////////////////////////////////////////////////////////////////////
        // train network
        ///////////////////////////////////////////////////////////////////////
        fmt::println("\nStart training cycle...\n");
        nn.train(fd_train, td_train, nn_meta);
        fmt::println("\nStop training cycle...\n\n");

        // print_nodes("nn", nn);
        // print_weights("nn", nn);

        ///////////////////////////////////////////////////////////////////////
        // run test data through trained network
        ///////////////////////////////////////////////////////////////////////
        fmt::println("Prediction with trained network:\n");
        nn.test(fd_test, td_test);
    }
    catch (std::exception& e) {
        fmt::println("Exception: ", e.what());
    }

    return 0;
}
