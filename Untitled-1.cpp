#include <iostream>

int main() {
    int x = 10;
    std::cout << "Value of x: " << x << std::endl;

    std::cout << "Value of ref: " << &x << std::endl;

    int *ptr = &x;
    std::cout << "Value pointed to by ptr: " << *ptr << std::endl;

    int &ref = x;
    std::cout << "Value of ref: " << ref << std::endl;

    int *ptr1 = &ref;
}