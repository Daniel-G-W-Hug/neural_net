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

// TODO: export/import of trained values for w and b
// TODO: make random distribution in neural_net.cpp a selectable meta parameter

void print_help_message()
{
    std::cout << "Provide a 'case_name' as single argument, please.\n\n";
    std::cout << "Remaining file names will be derived from 'case_name':\n\n";
    std::cout << "  - 'case_name.cfg' -> config file incl. meta data.\n";
    std::cout << "  - 'case_name_train.csv' -> training data.\n";
    std::cout << "  - 'case_name_test.csv' -> test data.\n";
    std::cout << "\n";
    std::cout << "If case_name ends with '_validate', i.e. is "
                 "'case_name_validate' then:\n\n";
    std::cout << "  - 'case_name_validate.cfg' -> config file incl. meta data.\n";
    std::cout << "  - 'case_name_validate.csv' -> validation data.\n";
    std::cout << "  - 'case_name_test.csv' -> test data.\n\n";
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
        fn_train = prefix + loc_case_name + ".csv";
        // remove "_validate", to get correct file name for test data below
        loc_case_name.erase(loc_case_name.find(str_to_erase), str_to_erase.length());
    }
    else {
        fn_train = prefix + loc_case_name + "_train.csv";
    }
    fn_test = prefix + loc_case_name + "_test.csv";

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
            // scale picture data, i.e. gray scale range from range 0..255 to 0..1
            std::cout << "\nScaling mnist data\n\n";
            scale_data(fd_train, 1. / 255.);
            scale_data(fd_test, 1. / 255.);
        }

        // std::cout << "neural_net nodes not yet activated." << std::endl;
        print_parameters("nn", nn);
        // print_nodes("nn", nn);
        // print_weights("nn", nn);

        ///////////////////////////////////////////////////////////////////////
        // train network
        ///////////////////////////////////////////////////////////////////////
        std::cout << "\nStart training cycle...\n\n";
        nn.train(fd_train, td_train, nn_meta);
        std::cout << "\nStop training cycle...\n\n\n";

        // print_nodes("nn", nn);
        // print_weights("nn", nn);

        ///////////////////////////////////////////////////////////////////////
        // run test data through trained network
        ///////////////////////////////////////////////////////////////////////
        std::cout << "Prediction with trained network:\n";
        nn.test(fd_test, td_test);

        // // for 2x1_linear classify and 2x2x1_example
        // std::vector<double> inp1{-7., -3.}, inp2{20., 2.}, outp;
        // outp = nn.forward_pass_with_output(inp1);
        // std::cout << "inp1 => outp[0]: " << outp[0] << std::endl;
        // if (nn.num_nodes[nn.num_layers - 1] > 1)
        //     std::cout << "inp1 => outp[1]: " << outp[1] << std::endl;
        // outp = nn.forward_pass_with_output(inp2);
        // std::cout << "inp2 => outp[0]: " << outp[0] << std::endl;
        // if (nn.num_nodes[nn.num_layers - 1] > 1)
        //     std::cout << "inp2 => outp[1]: " << outp[1] << std::endl;
    }
    catch (std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }

    return 0;
}
