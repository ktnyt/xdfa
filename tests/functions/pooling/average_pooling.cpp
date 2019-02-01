#include "doctest.h"
#include "doctest-helper.h"

#include "xnn/functions/pooling/average_pooling.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"

namespace F = xnn::functions;

TEST_CASE("average pooling k=2 s=2") {
  // clang-format off
  xt::xarray<float> x = {{{
    {0, 1, 2, 3},
    {8, 7, 5, 6},
    {4, 3, 1, 2},
    {0, -1, -2, -3}
  }}};

  xt::xarray<float> e_y = {{{
    {4, 4},
    {1.5, -0.5},
  }}};
  // clang-format on

  xt::xarray<float> a_y = F::pooling::average_pooling_2d(x, 2, 2, 0);
  CLOSE(a_y - e_y, epsilon);
}

TEST_CASE("average pooling k=2 s=1") {
  // clang-format off
  xt::xarray<float> x = {{{
    {0, 1, 2, 3},
    {8, 7, 5, 6},
    {4, 3, 1, 2},
    {0, -1, -2, -3}
  }}};

  xt::xarray<float> e_y = {{{
    {16.0, 15.0, 16.0},
    {22.0, 16.0, 14.0},
    {6.0, 1.0, -2.0}
  }}};

  e_y /= 4.0f;
  // clang-format on

  xt::xarray<float> a_y = F::pooling::average_pooling_2d(x, 2, 1, 0);
  CLOSE(a_y - e_y, epsilon);
}

TEST_CASE("average pooling grad k=2 s=2") {
  // clang-format off
  xt::xarray<float> x = {{{
    {4, 3},
    {1.5, -0.5},
  }}};

  xt::xarray<float> e_y = {{{
    {4, 4, 3, 3},
    {4, 4, 3, 3},
    {1.5, 1.5, -0.5, -0.5},
    {1.5, 1.5, -0.5, -0.5}
  }}};
  e_y /= 4.0f;
  // clang-format on

  xt::xarray<float> a_y = F::pooling::average_pooling_2d_grad(x, 4, 4, 2, 2, 0);
  CLOSE(a_y - e_y, epsilon);
}

TEST_CASE("average pooling grad k=2 s=2") {
  // clang-format off
  xt::xarray<float> x = {{{
    {0, 1, 2},
    {8, 7, 5},
    {4, 3, 1}
  }}};

  xt::xarray<float> e_y = {{{
    {0, 1, 3, 2},
    {8, 16, 15, 7},
    {12, 22, 16, 6},
    {4, 7, 4, 1}
  }}};
  e_y /= 4.0f;
  // clang-format on

  xt::xarray<float> a_y = F::pooling::average_pooling_2d_grad(x, 4, 4, 2, 1, 0);
  CLOSE(a_y - e_y, epsilon);
}
