#include <iostream>
#include <fstream>
#include <string>
#include <exception>
#include <nlohmann/json.hpp>
#include <boost/filesystem.hpp>
#include <nf/utilities/logger.hpp>
#include <argparse/argparse.hpp>
#include "ImgClassifier.hpp"

#define LOG_LEVEL NF_LOGGER_LEVEL_TRACE
#define SHOW_IN_CMD_PROMPT true //set it to false if you want to 

int main(int argc, char *argv[]) {
    if (!(NF_LOGGER_INITIALIZE()))
        NF_LOGGER_INIT(SHOW_IN_CMD_PROMPT, LOG_LEVEL);// Initialize the logger system. IT IS A MUST TO ADD THIS LINE OF CODE!!!!
    std::string program_name = "Image Classifier Tester";
    // NF_LOGGER_TRACE("{0}: starting!", program_name);
    argparse::ArgumentParser program(program_name);

    program.add_argument("--config_path")
        .default_value(std::string{"ResNet50_ImgClassifier.json"})   // might otherwise be type const char* leading to an error when trying program.get<std::string>
        .help("specify the config file path.");
    program.add_argument("--image_path")
        .default_value(std::string{"sample01.jpg"})   // might otherwise be type const char* leading to an error when trying program.get<std::string>
        .help("specify the image file path.");

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        NF_LOGGER_ERROR("{0}: {1}", program_name, err.what());
        return -1;
    }

    std::string the_config_path = program.get<std::string>("--config_path");
    std::string the_image_path = program.get<std::string>("--image_path");
    std::shared_ptr<nf::I_InferEngine<BASE_DATUM_SP>> the_engine = std::make_shared<dd::ImageClassifier<BASE_DATUM_SP>>(the_config_path, 0);
    if (!the_engine->Init()) {
        NF_LOGGER_ERROR("{0}: Failed to init the engine!!", program_name);
        return -1;
    }
    BASE_DATUM_SP the_datum = std::make_shared<BASE_DATUM>();
    the_datum->cvInputData = cv::imread(the_image_path);
    std::vector<BASE_DATUM_SP> the_input_data;
    the_input_data.push_back(the_datum);
    // NF_LOGGER_TRACE("{0}: before forward!", program_name);
    the_engine->Forward(the_input_data);
    NF_LOGGER_INFO("{0}: {1}", program_name, the_datum->className);
    if (NF_LOGGER_INITIALIZE())
        NF_LOGGER_CLOSE();// Close the logger system. IT IS A MUST TO ADD THIS LINE OF CODE!!!!

    return 0;
}