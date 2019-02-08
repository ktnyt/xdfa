#define XTENSOR_USE_XSIMD
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "xtensor/xarray.hpp"

#include "xnn/datasets/cifar10.hpp"
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
using L::noise::Dropout;
using L::pooling::MaxPooling2D;

int main() {
  std::random_device rd;
  D::cifar10::Training<float, int> dataset("cifar10", false, rd());
  dataset.x_data() /= 255.0f;

  std::size_t n_train = dataset.size();
  std::size_t n_epochs = 20;
  std::size_t batchsize = 100;

  Convolution2D conv1_1(3, 32, 3, 1, 0, O::Adam());
  ReLU a1_1;
  Convolution2D conv1_2(32, 32, 3, 1, 0, O::Adam());
  ReLU a1_2;
  MaxPooling2D pool1(2, 2, 0);
  Dropout d1(0.25);

  Convolution2D conv2_1(32, 64, 3, 1, 0, O::Adam());
  ReLU a2_1;
  Convolution2D conv2_2(64, 64, 3, 1, 0, O::Adam());
  ReLU a2_2;
  MaxPooling2D pool2(2, 2, 0);
  Dropout d2(0.25);

  Flatten<float> flat;

  Linear fc1(5 * 5 * 64, 512, O::Adam());
  ReLU a3;
  Linear fc2(512, 10, O::Adam());

  Serial<float> conv1(conv1_1, a1_1, conv1_2, a1_2, pool1, d1);
  Serial<float> conv2(conv2_1, a2_1, conv2_2, a2_2, pool2, d2);
  Serial<float> fc(fc1, a3, fc2);
  Serial<float> network(conv1, conv2, flat, fc);

  SoftmaxCrossEntropy error;

  for (std::size_t epoch = 0; epoch < n_epochs; ++epoch) {
    xt::xarray<float> loss = 0.0;
    xt::xarray<float> acc = 0.0;

    std::size_t batchnum = 1;

    auto train = [&](xt::xarray<float>& x, xt::xarray<int>& t) {
      xt::xarray<float> y = network(x);
      loss += error.with(t)(y) * static_cast<float>(batchsize);
      acc += accuracy(t, y) * static_cast<float>(batchsize);

      network.backward(error.grads());
      network.update();

      std::cout << "\rEpoch " << std::right << std::setfill(' ') << std::setw(2)
                << epoch + 1 << " " << std::right << std::setfill('0')
                << std::setw(5) << batchnum * batchsize << " / " << n_train
                << std::flush;

      ++batchnum;
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
