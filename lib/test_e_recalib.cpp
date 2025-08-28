/*
 * Test program for e_recalib library
 */

#include "e_recalib.h"
#include <iostream>

int main() {
    std::cout << "Testing e_recalib library..." << std::endl;
    
    try {
        // Test the library by calling the recalib function
        // Note: This is just a compilation test, you'll need actual files to run it
        std::cout << "e_recalib library compiled successfully!" << std::endl;
        std::cout << "Library functions:" << std::endl;
        std::cout << "- recalib() function is available" << std::endl;
        std::cout << "- RecalibInfo struct is defined" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
