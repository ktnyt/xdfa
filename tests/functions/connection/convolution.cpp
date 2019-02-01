#include "doctest.h"
#include "doctest-helper.h"

#include "xnn/functions/connection/convolution.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"

namespace F = xnn::functions;

TEST_CASE("convolution k=3, s=1, p=0") {
  // clang-format off
  xt::xarray<float> W = {{{
    {0.3f,  0.1f, 0.2f},
    {0.0f, -0.1f, -0.1f},
    {0.05f, -0.2f, 0.05f}
  }}, {{
    {0.0f, -0.1f, 0.1f},
    {0.1f, -0.2f, 0.3f},
    {0.2f, -0.3f, 0.2f}
  }}};

  xt::xarray<float> x = {{{
    {3, 2, 1, 5, 2},
    {3, 0, 2, 0, 1},
    {0, 6, 1, 1, 10},
    {3, -1, 2, 9, 0},
    {1, 2, 1, 5, 5}
  }}};

  xt::xarray<float> e_y = {{{
    {-0.05, 1.65, 1.45},
    {1.05, 0.0, -2.0},
    {0.4, 1.15, 0.8}
  }, {
    {-0.8, 1.1, 2.1},
    {0.6, 1.5, 0.7},
    {0.4, 3.3, -1.0}
  }}};
  // clang-format on

  xt::xarray<float> a_y = F::connection::convolution_2d(x, W, 1, 0);
  CLOSE(a_y - e_y, 1e-5);
}

TEST_CASE("convolution k=3, s=2, p=0") {
  // clang-format off
  xt::xarray<float> W = xt::zeros<float>({1, 1, 3, 3}) + 0.5f;

  xt::xarray<float> x = {{{
    {0, 1, 2, 3, 4},
    {1, 2, 3, 4, 5},
    {2, 3, 4, 5, 6},
    {3, 4, 5, 6, 7},
    {4, 5, 6, 7, 8},
  }}};

  xt::xarray<float> e_y = {{{
    {9, 18},
    {18, 27}
  }}};

  xt::xarray<float> dy = {{{
    {-1.0f, 2.0f},
    {3.0f, 0.0f}
  }}};

  xt::xarray<float> e_dx = {{{
    {-0.5f, -0.5f, 0.5f, 1.0f, 1.0f},
    {-0.5f, -0.5f, 0.5f, 1.0f, 1.0f},
    {1.0f, 1.0f, 2.0f, 1.0f, 1.0f},
    {1.5f, 1.5f, 1.5f, 0.0f, 0.0f},
    {1.5f, 1.5f,  1.5f,  0.0f, 0.0f}
  }}};

  xt::xarray<float> e_dW = {{{
    {10.0f, 14.0f, 18.0f},
    {14.0f, 18.0f, 22.0f},
    {18.0f, 22.0f, 26.0f}
  }}};
  // clang-format on

  xt::xarray<float> a_y = F::connection::convolution_2d(x, W, 2, 0);
  CLOSE(a_y - e_y, 1e-5);

  xt::xarray<float> a_dx = F::connection::deconvolution_2d(dy, W, 2, 0);
  CLOSE(a_dx - e_dx, 1e-5);

  xt::xarray<float> a_dW = F::connection::convolution_2d_grad(x, W, dy, 2, 0);
  CLOSE(a_dW - e_dW, 1e-5);
}
