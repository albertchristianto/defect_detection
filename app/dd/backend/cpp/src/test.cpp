#include <iostream>
#include <fstream>
#include <string>
#include <exception>
#include <nlohmann/json.hpp>
#include <boost/filesystem.hpp>
#include <nf/utilities/logger.hpp>

#define LOG_LEVEL NF_LOGGER_LEVEL_TRACE
#define SHOW_IN_CMD_PROMPT true //set it to false if you want to 

int main() {
    if (!(NF_LOGGER_INITIALIZE()))
        NF_LOGGER_INIT(SHOW_IN_CMD_PROMPT, LOG_LEVEL);// Initialize the logger system. IT IS A MUST TO ADD THIS LINE OF CODE!!!!

    NF_LOGGER_TRACE("this is trace logging!");
    NF_LOGGER_INFO("this is info logging!");
    NF_LOGGER_WARN("this is warning logging!");
    NF_LOGGER_CRITICAL("this is critical logging!");
    NF_LOGGER_ERROR("this is error logging!");

    std::string the_path = "ResNet50_ImgClassifier.json";
    if (!boost::filesystem::exists(boost::filesystem::path(the_path))) {
        NF_LOGGER_ERROR("Heyy: Could not find the json config file!!!");//throw an error
        return -1;
    }
    std::ifstream f(the_path);
    nlohmann::json data = nlohmann::json::parse(f);

    std::vector<double> means = data["means"].get<std::vector<double>>();
    for (int i = 0; i < means.size(); ++i)
        std::cout << means[i] << ", ";
    std::cout << std::endl;
    std::vector<double> stds = data["stds"].get<std::vector<double>>();
    for (int i = 0; i < stds.size(); ++i)
        std::cout << stds[i] << ", ";
    std::cout << std::endl;
    std::vector<std::string> class_name = data["class_name"].get<std::vector<std::string>>();
    for (int i = 0; i < class_name.size(); ++i)
        std::cout << class_name[i] << ", ";
    std::cout << std::endl;

    if (NF_LOGGER_INITIALIZE())
        NF_LOGGER_CLOSE();// Close the logger system. IT IS A MUST TO ADD THIS LINE OF CODE!!!!

    return 0;
}