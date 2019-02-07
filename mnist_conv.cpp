#define XTENSOR_USE_XSIMD
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "xtensor/xarray.hpp"

#include "xnn/datasets/mnist.hpp"
#include "xnn/xnn.hpp"

namespace F = xnn::functions;
namespace L = xnn::layers;
namespace O = xnn::optimizers;
namespace D = xnn::datasets;

int main() {
  std::size_t batchsize = 100;
  D::mnist::Training<float, int> dataset("mnist", batchsize, false);
  dataset.x_data() /= 255.0f;

  std::size_t n_train = dataset.x_data().shape()[0];
  std::size_t n_input = dataset.x_data().shape()[1];

  std::size_t n_epochs = 20;

  L::connection::Convolution2D conv1(1, 20, 5, 1, 0, O::Adam());
  L::activation::ReLU a1;
  L::pooling::MaxPooling2D pool1(2, 2, 0);
  L::connection::Convolution2D conv2(20, 50, 5, 1, 0, O::Adam());
  L::activation::ReLU a2;
  L::pooling::MaxPooling2D pool2(2, 2, 0);
  L::connection::Linear fc1(4 * 4 * 50, 500, O::Adam());
  L::activation::ReLU a3;
  L::connection::Linear fc2(500, 10, O::Adam());

  L::miscellaneous::Serial<float> network(
      conv1, a1, pool1, conv2, a2, pool2, fc1, a3, fc2);

  L::loss::SoftmaxCrossEntropy error;

  for (std::size_t epoch = 0; epoch < n_epochs; ++epoch) {
    xt::xarray<float> loss = 0.0;
    xt::xarray<float> acc = 0.0;

    std::size_t batchnum = 0;

    auto train = [&](xt::xarray<float>& x, xt::xarray<int>& t) {
      std::cout << "\rEpoch " << std::right << std::setfill(' ') << std::setw(2)
                << epoch + 1 << " " << std::right << std::setfill('0')
                << std::setw(5) << ++batchnum * batchsize << " / " << n_train
                << std::flush;
      xt::xarray<float> y = network.forward(x);
      loss += error.with(t).forward(y) * batchsize;
      acc += F::evaluation::accuracy(t, y) * batchsize;

      network.backward(error.grads());
      network.update();
    };

    dataset.shuffle();
    dataset.for_each(batchsize, train);

    std::cout << "\rEpoch " << std::right << std::setfill(' ') << std::setw(2)
              << epoch + 1 << " Loss: " << std::scientific << loss / n_train
              << " Accuracy: " << acc / n_train << std::endl;
  }

  return 0;
}
