#pragma once
#include <chrono>

using std::chrono::milliseconds;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;




// In milliseconds.
template <class F>
float measureDuration(F fn) {
  auto start = high_resolution_clock::now();

  fn();

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  return duration.count();
}
