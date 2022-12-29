#include <iostream>
#include <fstream>
#include <string>
#include <exception>
#include <nlohmann/json.hpp>
#include <boost/filesystem.hpp>

int main() {
    std::string the_path = "ResNet50_ImgClassifier.json";
    if (!boost::filesystem::exists(boost::filesystem::path(the_path))) {
        std::cout << "Heyy: Could not find the json config file!!!\n";//throw an error
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

    return 0;
}