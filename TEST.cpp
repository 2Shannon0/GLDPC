#include <iostream>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/number.hpp>

namespace mp = boost::multiprecision;
using big_float = mp::number<mp::cpp_dec_float<1000>>;

int main() {
    big_float up = 2.0;
    big_float down = 3.0;

    big_float result = mp::log(up / down);
    std::cout << std::setprecision(100) << result << std::endl;

    return 0;
}
