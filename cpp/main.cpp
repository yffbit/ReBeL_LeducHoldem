// #define EIGEN_USE_MKL_ALL
#include "command.h"
using std::cout;
using std::endl;

void print_help(const char* exe) {
    printf("%s [file]\n", exe);
    cout << "file: command file path. If not given, you should input commands line by line.\n";
    print_possible_cmd();
}

int main(int argc, char* argv[]) {
    if(argc != 1 && argc != 2) {
        print_help(argv[0]);
        return -1;
    }
    Command cmd;
    try {
        if(argc == 1) {
            print_possible_cmd();
            cmd.input_cmd();
        }
        else cmd.load_file_cmd(argv[1]);
    }
    catch (const std::exception& e) {
        cout << e.what() << endl;
    }
    return 0;
}
