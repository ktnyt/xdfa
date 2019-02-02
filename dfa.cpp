#define XTENSOR_USE_XSIMD
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "xtensor/xarray.hpp"

#include "xnn/xnn.hpp"
#include "xnn/loaders/mnist.hpp"

namespace F = xnn::functions;
namespace L = xnn::layers;
namespace O = xnn::optimizers;

int main() {
  auto train_images_path = "mnist/train-images-idx3-ubyte";
  auto train_labels_path = "mnist/train-labels-idx1-ubyte";

  xt::xarray<float> x_train =
      mnist::read_images<float>(train_images_path, true) / 255.0;
  xt::xarray<int> t_train = mnist::read_labels<int>(train_labels_path);

  auto x_shape = x_train.shape();
  auto t_shape = t_train.shape();

  std::size_t n_train = x_shape[0];
  std::size_t n_input = x_shape[1];
  std::size_t n_output = 10;

  std::size_t n_epochs = 20;
  std::size_t batchsize = 100;
  std::size_t n_hidden = 1000;

  L::activation::Sigmoid a0;
  L::connection::LinearFeedback l0(n_input, n_hidden, n_output, O::Adam(), a0);
  L::connection::Linear l1(n_hidden, 10, O::Adam());
  L::loss::SoftmaxCrossEntropy error;
  L::miscellaneous::DirectFeedback<float> network(l0, l1);

  for (std::size_t epoch = 0; epoch < n_epochs; ++epoch) {
    xt::xarray<float> loss = 0.0;
    xt::xarray<float> acc = 0.0;

    xt::random::seed(epoch);
    xt::random::shuffle(x_train);
    xt::random::seed(epoch);
    xt::random::shuffle(t_train);

    for (std::size_t i = 0; i < n_train; i += batchsize) {
      std::cout << "\rEpoch " << std::right << std::setfill(' ') << std::setw(2)
                << epoch + 1 << " " << std::right << std::setfill('0')
                << std::setw(5) << i + batchsize << " / " << n_train
                << std::flush;
      xt::xarray<float> x = xt::view(x_train, xt::range(i, i + batchsize));
      xt::xarray<float> t = xt::view(t_train, xt::range(i, i + batchsize));

      xt::xarray<float> y = network.forward(x);

      loss += error.with(t).forward(y) * batchsize;
      acc += F::evaluation::accuracy(t, y) * batchsize;

      network.backward(error.grads());
      network.update();
    }

    std::cout << "\rEpoch " << std::right << std::setfill(' ') << std::setw(2)
              << epoch + 1 << " Loss: " << std::scientific << loss / n_train
              << " Accuracy: " << acc / n_train << std::endl;
  }

  return 0;
}
