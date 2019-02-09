#define XTENSOR_USE_XSIMD
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
using L::activation::Sigmoid;
using L::connection::Linear;
using L::connection::LinearFeedback;
using L::loss::SoftmaxCrossEntropy;
using L::network::DirectFeedback;

int main() {
  std::random_device rd;
  D::mnist::Training<float, int> dataset("mnist", true, rd());
  dataset.x_data() /= 255.0f;

  std::size_t n_train = dataset.size();
  std::size_t n_epochs = 20;
  std::size_t batchsize = 100;

  std::size_t n_hidden = 1000;
  std::size_t n_output = 10;

  Sigmoid a0;
  LinearFeedback l0(n_hidden, O::Adam(), a0);
  Linear l1(n_output, O::Adam());
  SoftmaxCrossEntropy error;
  DirectFeedback<float> network(l0, l1);

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
