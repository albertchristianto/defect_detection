#include <iostream>
#include <string>
#include <atomic>
#include <exception>
#include <thread>

#include <argparse/argparse.hpp>
#include <opencv2/opencv.hpp>

#include "cWrapper.h"

std::atomic<bool> the_tag = false;

int my_send_results(dd::C_Results& the_output) {
    std::string the_log = "the output is " + std::string(the_output.ClassName);
    dd::Info(the_log.c_str());
    the_tag = true;
    return 0;
}

int main(int argc, char *argv[]) {
    dd::Initialize();
    int api_id = dd::AddApiFuncPtr(&my_send_results);
    std::string program_name = "Defect Detection API Tester";
    // NF_LOGGER_TRACE("{0}: starting!", program_name);
    argparse::ArgumentParser program(program_name);

    program.add_argument("--image_path")
        .default_value(std::string{"sample01.jpg"})   // might otherwise be type const char* leading to an error when trying program.get<std::string>
        .help("specify the image file path.");

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::string the_log = std::string(err.what());
        dd::Error(the_log.c_str());
        dd::Terminate();
        return -1;
    }

    std::string the_image_path = program.get<std::string>("--image_path");
    cv::Mat the_image = cv::imread(the_image_path);
    dd::C_Image to_be_sent;
    to_be_sent.TimeStamp = 0;
    to_be_sent.Ptr = the_image.data;
    to_be_sent.Height = the_image.rows;
    to_be_sent.Width = the_image.cols;
    to_be_sent.Depth = the_image.channels();
    dd::SendImage(api_id, to_be_sent);

    while (!the_tag.load())
        std::this_thread::yield();
    dd::Terminate();
    return 0;
}