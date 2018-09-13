#define XTENSOR_USE_XSIMD
#include <cmath>
#include <iostream>
#include <vector>

#include "xnn/functions.hpp"
#include "xnn/optimizers.hpp"

#include "xnn/loaders/mnist.hpp"

namespace F = xnn::functions;
namespace O = xnn::optimizers;

using namespace xnn::loaders;

int main() {
  auto train_image_path = "mnist/train-images-idx3-ubyte";
  auto train_label_path = "mnist/train-labels-idx1-ubyte";

  xt::xarray<float> x_train = mnist::read_images<float>(train_image_path, true);
  xt::xarray<int> t_train = mnist::read_labels<int>(train_label_path);

  auto x_shape = x_train.shape();
  std::size_t n_train = x_shape[0];
  std::size_t n_dims = x_shape[1];

  std::size_t n_epochs = 20;
  std::size_t batchsize = 100;

  F::Linear<float> l0(n_dims, 240);
  F::Linear<float> l1(240, 10);
  F::Sigmoid<float> a0;
  F::SoftmaxCrossEntropy<float, int> error;

  O::AdamRule<float> o0;
  O::AdamRule<float> o1;

  for (std::size_t epoch = 0; epoch < n_epochs; ++epoch) {
    std::cout << "Epoch " << epoch << std::flush;

    xt::xarray<float> loss = 0.0;
    xt::xarray<float> acc = 0.0;

    xt::random::seed(epoch);
    xt::random::shuffle(x_train);

    t_train.resize({n_train, 1});
    xt::random::seed(epoch);
    xt::random::shuffle(t_train);
    t_train.resize({n_train});

    for (std::size_t i = 0; i < n_train; i += batchsize) {
      xt::xarray<float> x = xt::view(x_train, xt::range(i, i + batchsize));
      xt::xarray<int> t = xt::view(t_train, xt::range(i, i + batchsize));

      xt::xarray<float> a = l0(x);
      xt::xarray<float> h = a0(a);
      xt::xarray<float> y = l1(h);

      loss += error.with(t)(y) * batchsize;
      acc += F::accuracy(t, y) * batchsize;

      xt::xarray<float> dy = error.grads();
      xt::xarray<float> dh = l1.derivative(dy);
      xt::xarray<float> da = a0.derivative(dh);
      l0.update(o0.diff(da));
      l1.update(o1.diff(dy));
    }

    std::cout << " Loss: " << loss / n_train << " Accuracy: " << acc / n_train
              << std::endl;
  }

  return 0;
}
