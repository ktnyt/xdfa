#define XTENSOR_USE_XSIMD
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
  D::mnist::Training<float, int> dataset("mnist", batchsize, true);
  dataset.x_data() /= 255.0f;

  std::size_t n_train = dataset.x_data().shape()[0];
  std::size_t n_input = dataset.x_data().shape()[1];
  std::size_t n_hidden = 1000;
  std::size_t n_output = 10;

  std::size_t n_epochs = 20;

  L::activation::Sigmoid a0;
  L::connection::LinearFeedback l0(n_input, n_hidden, n_output, O::Adam(), a0);
  L::connection::Linear l1(n_hidden, n_output, O::Adam());
  L::loss::SoftmaxCrossEntropy error;
  L::miscellaneous::DirectFeedback<float> network(l0, l1);

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
      acc += F::evaluation::accuracy(t, y) * static_cast<float>(batchsize);

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
