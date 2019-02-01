#include "doctest.h"
#include "doctest-helper.h"

#include "xnn/functions/pooling/max_pooling.hpp"

#include "xtensor/xarray.hpp"

namespace F = xnn::functions;

TEST_CASE("max pooling k=2 s=2") {
  // clang-format off
  xt::xarray<float> x = {{{
    {0, 1, 2, 3},
    {8, 7, 5, 6},
    {4, 3, 1, 2},
    {0, -1, -2, -3}
  }}};

  xt::xarray<float> e_y = {{{
    {8, 6},
    {4, 2}
  }}};

  xt::xarray<std::size_t> e_i = {{{
    {2, 3},
    {0, 1}
  }}};
  
  xt::xarray<float> dy = {{{
    {1, 2},
    {3, 4}
  }}};

  xt::xarray<float> e_dx = {{{
    {0, 0, 0, 0},
    {1, 0, 0, 2},
    {3, 0, 0, 4},
    {0, 0, 0, 0}
  }}};
  // clang-format on

  xt::xarray<std::size_t> a_i;
  xt::xarray<float> a_y;

  std::tie(a_y, a_i) = F::pooling::max_pooling_2d(x, 2, 2, 0);
  CLOSE(a_y - e_y, epsilon);
  REQUIRE(a_i == e_i);

  xt::xarray<float> a_dx = F::pooling::max_pooling_2d_grad(dy, a_i, 2, 2, 0);
  CLOSE(a_dx - e_dx, epsilon);
}

TEST_CASE("max pooling k=2 s=1") {
  // clang-format off
  xt::xarray<float> x = {{{
    {0, 1, 2, 3},
    {8, 7, 5, 6},
    {4, 3, 1, 2},
    {0, -1, -2, -3}
  }}};

  xt::xarray<float> e_y = {{{
    {8, 7, 6},
    {8, 7, 6},
    {4, 3, 2}
  }}};

  xt::xarray<std::size_t> e_i = {{{
    {2, 2, 3},
    {0, 0, 1},
    {0, 0, 1}
  }}};
  // clang-format on

  xt::xarray<std::size_t> a_i;
  xt::xarray<float> a_y;

  std::tie(a_y, a_i) = F::pooling::max_pooling_2d(x, 2, 1, 0);
  CLOSE(a_y - e_y, epsilon);
  REQUIRE(a_i == e_i);
}
