#include "doctest.h"
#include "doctest-helper.h"

#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xview.hpp"

#include "xnn/utils/convolution.hpp"

#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#define between(a, b, c) (a <= b && b < c)

template <class T>
std::string to_string(std::vector<T> v) {
  std::stringstream ss;
  ss << "{ ";
  for (std::size_t i = 0; i < v.size(); ++i) {
    ss << v[i];
    if (i < v.size() - 1) {
      ss << ", ";
    }
  }
  ss << " }";
  std::string tmp = ss.str();
  return std::string(tmp.begin(), tmp.end());
}

TEST_CASE("image to column conversion") {
  std::size_t w = 10;
  std::size_t h = 8;

  std::vector<std::size_t> params;
  SUBCASE("") { params = {1, 1, 1, 1, 1, 1}; }
  SUBCASE("") { params = {2, 2, 2, 2, 2, 2}; }
  SUBCASE("") { params = {1, 2, 2, 1, 1, 2}; }
  SUBCASE("") { params = {1, 2, 3, 4, 1, 2}; }
  SUBCASE("") { params = {1, 2, 3, 4, 4, 5}; }
  SUBCASE("") { params = {3, 3, 2, 2, 1, 1}; }

  std::string pstr = to_string(params);
  CAPTURE(pstr);

  std::size_t kh = params[0];
  std::size_t kw = params[1];
  std::size_t sy = params[2];
  std::size_t sx = params[3];
  std::size_t ph = params[4];
  std::size_t pw = params[5];

  std::vector<std::size_t> shape = {2, 3, h, w};
  xt::xarray<float> img = xt::random::rand<float>(shape, -1.0f, 1.0f);
  xt::xarray<float> col = xnn::utils::im2col(img, kh, kw, sy, sx, ph, pw);
  std::size_t col_h = xnn::utils::get_conv_outsize(h, kh, sy, ph);
  std::size_t col_w = xnn::utils::get_conv_outsize(w, kw, sx, pw);

  std::vector<std::size_t> wanted_shape = {2, 3, kh, kw, col_h, col_w};
  std::vector<std::size_t> actual_shape(col.shape().begin(), col.shape().end());
  REQUIRE(actual_shape == wanted_shape);

  for (std::size_t y = 0; y < col_h; ++y) {
    for (std::size_t x = 0; x < col_w; ++x) {
      for (std::size_t ky = 0; ky < kh; ++ky) {
        for (std::size_t kx = 0; kx < kw; ++kx) {
          long long oy = y * sy - ph + ky;
          long long ox = x * sx - pw + kx;
          auto v = xt::view(col, xt::all(), xt::all(), ky, kx, y, x);
          if (between(0, oy, h) && between(0, ox, w)) {
            v -= xt::view(img, xt::all(), xt::all(), oy, ox);
          }
          CLOSE(v, epsilon);
        }
      }
    }
  }
}

TEST_CASE("column to image conversion") {
  std::size_t w = 10;
  std::size_t h = 8;

  std::vector<std::size_t> params;
  SUBCASE("") { params = {1, 1, 1, 1, 1, 1}; }
  SUBCASE("") { params = {2, 2, 2, 2, 2, 2}; }
  SUBCASE("") { params = {1, 2, 2, 1, 1, 2}; }
  SUBCASE("") { params = {1, 2, 3, 4, 1, 2}; }
  SUBCASE("") { params = {1, 2, 3, 4, 4, 5}; }
  SUBCASE("") { params = {3, 3, 2, 2, 1, 1}; }

  std::string pstr = to_string(params);
  CAPTURE(pstr);

  std::size_t kh = params[0];
  std::size_t kw = params[1];
  std::size_t sy = params[2];
  std::size_t sx = params[3];
  std::size_t ph = params[4];
  std::size_t pw = params[5];

  std::size_t col_h = xnn::utils::get_conv_outsize(h, kh, sy, ph);
  std::size_t col_w = xnn::utils::get_conv_outsize(w, kw, sx, pw);
  std::vector<std::size_t> shape = {2, 3, kh, kw, col_h, col_w};
  xt::xarray<float> col = xt::random::rand<float>(shape, -1.0f, 1.0f);

  xt::xarray<float> img = xnn::utils::col2im(col, sy, sx, ph, pw, h, w);

  std::vector<std::size_t> wanted_shape = {2, 3, h, w};
  std::vector<std::size_t> actual_shape(img.shape().begin(), img.shape().end());
  REQUIRE(actual_shape == wanted_shape);

  for (std::size_t y = 0; y < h; ++y) {
    for (std::size_t x = 0; x < w; ++x) {
      xt::xarray<float> v = xt::zeros<float>({2, 3});
      xt::xarray<float> u = xt::view(img, xt::all(), xt::all(), y, x);

      for (std::size_t ky = 0; ky < kh; ++ky) {
        for (std::size_t kx = 0; kx < kw; ++kx) {
          std::size_t oy = (y + ph - ky) / sy;
          std::size_t ox = (x + pw - kx) / sx;
          if ((y + ph - ky) % sy == 0 && (x + pw - kx) % sx == 0 &&
              between(0, oy, col_h) && between(0, ox, col_w)) {
            v += xt::view(col, xt::all(), xt::all(), ky, kx, oy, ox);
          }
        }
      }

      CLOSE(v - u, epsilon);
    }
  }
}
