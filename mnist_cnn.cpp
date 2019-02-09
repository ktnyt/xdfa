#define XTENSOR_USE_XSIMD
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "xtensor/xarray.hpp"

#include "xnn/datasets/mnist.hpp"
#include "xnn/xnn.hpp"

namespace F = xnn::functions;
namespace L = xnn::layers;
namespace O = xnn::optimizers;
namespace D = xnn::datasets;

using F::evaluation::accuracy;
using L::activation::ReLU;
using L::connection::Convolution2D;
using L::connection::Linear;
using L::loss::SoftmaxCrossEntropy;
using L::manipulation::Flatten;
using L::network::Serial;
using L::pooling::MaxPooling2D;

int main() {
  std::random_device rd;
  D::mnist::Training<float, int> dataset("mnist", false, rd());
  dataset.x_data() /= 255.0f;

  std::size_t n_train = dataset.size();
  std::size_t n_epochs = 20;
  std::size_t batchsize = 100;

  Convolution2D conv1(20, 5, 1, 0, O::Adam());
  ReLU a1;
  MaxPooling2D pool1(2, 2, 0);
  Convolution2D conv2(50, 5, 1, 0, O::Adam());
  ReLU a2;
  MaxPooling2D pool2(2, 2, 0);
  Flatten<float> flat;
  Linear fc1(500, O::Adam());
  ReLU a3;
  Linear fc2(10, O::Adam());

  Serial<float> network(conv1, a1, pool1, conv2, a2, pool2, flat, fc1, a3, fc2);

  SoftmaxCrossEntropy error;

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
      loss += error.with(t).forward(y) * static_cast<float>(batchsize);
      acc += accuracy(t, y) * static_cast<float>(batchsize);

      network.backward(error.grads());
      network.update();
    };

    dataset.shuffle();
    dataset.for_each(batchsize, train);

    std::cout << "\rEpoch " << std::right << std::setfill(' ') << std::setw(2)
              << epoch + 1 << " Loss: " << std::scientific
              << loss / static_cast<float>(n_train)
              << " Accuracy: " << acc / static_cast<float>(n_train)
              << std::endl;
  }

  return 0;
}
