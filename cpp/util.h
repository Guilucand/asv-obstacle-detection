#pragma once
#include <iostream>
#include <mutex>
#include "image_stab.h"

extern bool use_gpu;

float* allocate(int rows, int cols, int step, int count = 1);

int align(int size, int alignment);

inline cv::Point trpoint(TransformMat trmat, cv::Point pt, bool inverse) {
  auto tpoint = (cv::Matx13f(pt.x, pt.y, 1) * (inverse ? trmat.trinv.t() : trmat.trmat.t()) ).get_minor<1, 2>(0, 0);
  return cv::Point(tpoint(0, 0), tpoint(0, 1));
}

void draw_text(cv::Mat image, std::string text, cv::Point origin);

void Canny(cv::Mat image);

void blend(cv::Mat dst, cv::Mat src);

template <class T>
class Allocator {
public:
  Allocator(int size = 32) {
    pools_size = 0;
    max_index = 0;
    gen_pool(size);
  }

  // Copy initialization
  Allocator(const Allocator& other) {
    pools = other.pools;
    free_mem = other.free_mem;
    free_indices = other.free_indices;
    pools_size = other.pools_size;
  }

  ~Allocator() {
    for (auto pool : pools) {
      free(pool);
    }
    pools.clear();
  }

  T* allocate() {
    T* tmp;
    int idx;
    mtx.lock();
    if (free_mem.empty()) {
      gen_pool(std::max(1, pools_size));
    }

    tmp = free_mem.back();
    idx = free_indices.back();

    free_mem.pop_back();
    free_indices.pop_back();

    mtx.unlock();
    new(tmp) T();
    tmp->index = idx;
    return tmp;
  }

  void deallocate(T* ptr) {
    ptr->~T();
    free_mem.push_back(ptr);
    free_indices.push_back(ptr->index);
  }


  // Move assignment
  Allocator& operator = (Allocator&& other) {
    // std::lock(mtx, other.mtx);
    // std::lock_guard<std::mutex> self_lock(mtx, std::adopt_lock);
    // std::lock_guard<std::mutex> other_lock(other.mtx, std::adopt_lock);
    pools = other.pools;
    other.pools.clear();
    free_mem = other.free_mem;
    other.free_mem.clear();
    pools_size = other.pools_size;
    other.pools_size = 0;
    return *this;
  }

  // Copy assignment
  Allocator& operator = (const Allocator& other) {
    // std::lock(mtx, other.mtx);
    // std::lock_guard<std::mutex> self_lock(mtx, std::adopt_lock);
    // std::lock_guard<std::mutex> other_lock(other.mtx, std::adopt_lock);
    pools = other.pools;
    free_mem = other.free_mem;
    pools_size = other.pools_size;
    return *this;
  }
private:

  T* gen_pool(int size) {
    T* pool = (T*)malloc(size * sizeof(T));
    for (int i = size-1; i >= 0; i--) {
      free_mem.push_back(pool+i);
      free_indices.push_back(max_index+i);
    }
    max_index += size;

    pools.push_back(pool);
    pools_size += size;
    std::cout << "Allocating, new size = " << pools_size << std::endl;
    return pool;
  }

  std::vector<T*> free_mem;
  std::vector<T*> pools;
  std::vector<int> free_indices;
  int max_index;
  std::mutex mtx;
  int pools_size;
};
