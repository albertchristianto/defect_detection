#include <string>
#include <argparse/argparse.hpp>
#include <nf/utilities/logger.hpp>
#include "TrtEngine.hpp"

#define LOG_LEVEL NF_LOGGER_LEVEL_TRACE
#define SHOW_IN_CMD_PROMPT true //set it to false if you want to use txt files

int main(int argc, char *argv[]) {
    if (!(NF_LOGGER_INITIALIZE()))
        NF_LOGGER_INIT(SHOW_IN_CMD_PROMPT, LOG_LEVEL);// Initialize the logger system. IT IS A MUST TO ADD THIS LINE OF CODE!!!!
    std::string program_name = "Onnx-TensorRT_Engine_Generator";
    argparse::ArgumentParser program(program_name);

    program.add_argument("--batch_size")
        .help("the batch size of the model")
        .default_value(1)
        .scan<'i', int>();
    program.add_argument("--onnx_weight_path")
        .required()
        .help("specify the weights file path.");

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        NF_LOGGER_ERROR("{0}: {1}", program_name, err.what());
        return -1;
    }

    int batch_size = program.get<int>("--batch_size");
    std::string onnx_weight_path = program.get<std::string>("--onnx_weight_path");  // "orange"
    NF_LOGGER_TRACE("{0}: batch size set to {1}", program_name, onnx_weight_path);
    NF_LOGGER_TRACE("{0}: batch size set to {1}", program_name, batch_size);
    std::string out_path = dd::TrtParseOnnxModel(program_name, onnx_weight_path, batch_size, true);
    NF_LOGGER_INFO("{0}: Saved to {1}", program_name, out_path);
    if (NF_LOGGER_INITIALIZE())
        NF_LOGGER_CLOSE();// Close the logger system. IT IS A MUST TO ADD THIS LINE OF CODE!!!!
    return 0;
}