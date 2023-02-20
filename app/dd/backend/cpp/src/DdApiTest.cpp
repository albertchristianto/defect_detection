#include <iostream>
#include "cWrapper.h"

int main() {
    dd::Initialize();
    dd::Restart();
    dd::Terminate();

    return 0;
}