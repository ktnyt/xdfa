#define XTENSOR_USE_XSIMD
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xview.hpp"

#include "xnn/datasets/cifar10.hpp"
#include "xnn/xnn.hpp"

namespace F = xnn::functions;
namespace L = xnn::layers;
namespace O = xnn::optimizers;
namespace D = xnn::datasets;

auto expand(xt::xarray<float>&& in) {
  in.reshape({in.size(), 1});
  return in;
}

auto matrixify(const xt::xarray<int>& labels, std::size_t n_class) {
  xt::xarray<float> matrix = xt::zeros<float>({labels.shape()[0], n_class});
  auto indices = xnn::utils::to_index(labels);
  xt::index_view(matrix, indices) = 1.0;
  return matrix;
}

template <class E>
void print_shape(E&& exp) {
  auto& shape = exp.shape();
  std::cerr << "{ ";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      std::cerr << " ";
    }
    std::cerr << shape[i];
  }
  std::cerr << " }" << std::endl;
}

xt::xarray<float> similarity_matrix(
    xt::xarray<float> x, bool use_similarity_std = true) {
  if (x.dimension() == 4) {
    if (use_similarity_std && x.shape()[1] > 3 && x.shape()[2] > 1) {
      x.reshape({x.shape()[0], x.shape()[1], x.shape()[2] * x.shape()[3]});
      x = xt::stddev(x, {2});
    } else {
      x.reshape({x.shape()[0], x.shape()[1] * x.shape()[2] * x.shape()[3]});
    }
  }
  auto xc = x - expand(xt::mean(x, 1));
  auto xd = expand(xt::sqrt(xt::sum(xc * xc, 1)));
  xt::xarray<float> xn = xc / (1e-8 + xd);
  auto R = xt::linalg::dot(xn, xt::transpose(xn, {1, 0}));
  return xt::clip(R, -1.0f, 1.0f);
}

class LocalLossConv {
 public:
  LocalLossConv(
      std::size_t out_channel, std::size_t n_class, std::size_t pool_size)
      : n_class(n_class),
        conv(out_channel, 3, 1, 1, O::Adam()),
        f(conv, relu),
        pred_pool(pool_size, pool_size, 0),
        pred_fc(n_class, O::Adam()),
        pred(pred_pool, pred_flat, pred_fc),
        sim_conv(out_channel, 3, 1, 1, O::Adam()) {}

  xt::xarray<float> operator()(const xt::xarray<float>& x, xt::xarray<int>& y) {
    // Compute layer output
    xt::xarray<float> h = f(x);

    // Compute local prediction loss
    pred_loss.with(y)(pred(h));

    // Compute similarity loss
    xt::xarray<float> sh = sim_conv(h);
    xt::xarray<float> Rh = similarity_matrix(sh);
    xt::xarray<float> Ry = similarity_matrix(matrixify(y, n_class));
    sim_loss.with(Rh)(Ry);
    print_shape(sh);

    auto sim_grads = beta * sim_loss.grads();
    auto pred_grads = (1.0 - beta) * pred_loss.grads();
    print_shape(sim_grads);
    print_shape(pred_grads);
    xt::xarray<float> grads = sim_grads + pred_grads;

    f.backward(pred.backward(grads));
    f.update();
    pred.update();

    return x;
  }

 private:
  std::size_t n_class;
  L::connection::Convolution2D conv;
  L::activation::ReLU relu;
  L::network::Serial<float> f;

  float beta;

  L::pooling::AveragePooling2D pred_pool;
  L::manipulation::Flatten<float> pred_flat;
  L::connection::Linear pred_fc;
  L::network::Serial<float> pred;
  L::loss::SoftmaxCrossEntropy pred_loss;

  L::connection::Convolution2D sim_conv;
  L::loss::MeanSquaredError sim_loss;
};

int main() {
  std::random_device rd;
  D::cifar10::Training<float, int> dataset("cifar10", false, rd());
  dataset.x_data() /= 255.0f;

  std::size_t n_train = dataset.size();
  std::size_t n_class = 10;
  std::size_t n_epochs = 20;
  std::size_t batchsize = 100;

  LocalLossConv layer1(64, n_class, 8);

  xt::xarray<float> x_batch = xt::view(dataset.x_data(), xt::range(0, 10));
  xt::xarray<int> t_batch = xt::view(dataset.t_data(), xt::range(0, 10));
  layer1(x_batch, t_batch);

  return 0;
}
