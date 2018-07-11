#include <stdio.h>


template<class T, class X, X f>
void test() {
  T x = 7.0;
  f(x);
};


void func(float x) {
  printf("%f\n", x);
}

// int main() {
//   test<float, decltype(&func), &func>();
//
// }
