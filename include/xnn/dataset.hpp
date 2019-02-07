#ifndef __XNN_DATASET_HPP__
#define __XNN_DATASET_HPP__

#include "xnn/utils/helpers.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xview.hpp"

#include <algorithm>
#include <iterator>
#include <random>
#include <tuple>
#include <vector>

#include <iostream>

namespace xnn {

template <class T1, class T2>
class Dataset {
 public:
  Dataset(xt::xarray<T1>&& x, xt::xarray<T2>&& t) : x(x), t(t) {}

  template <class S>
  Dataset(xt::xarray<T1>&& x, xt::xarray<T2>&& t, S seed)
      : x(x), t(t), rng1(seed), rng2(seed) {}

  void shuffle() {
    xt::random::shuffle(x, rng1);
    xt::random::shuffle(t, rng2);
  }

  xt::xarray<T1> x_data() { return x; }
  xt::xarray<T2>& t_data() { return t; }

  void for_each(
      std::size_t batchsize,
      std::function<void(xt::xarray<T1>&, xt::xarray<T2>&)> f) {
    for (std::size_t i = 0; i < utils::len(x); i += batchsize) {
      xt::xarray<T1> x_batch = xt::view(x, xt::range(i, i + batchsize));
      xt::xarray<T2> t_batch = xt::view(t, xt::range(i, i + batchsize));
      f(x_batch, t_batch);
    }
  }

 private:
  xt::xarray<T1> x;
  xt::xarray<T2> t;

  std::mt19937 rng1;
  std::mt19937 rng2;
};

}  // namespace xnn

#endif  // __XNN_DATASET_HPP__
