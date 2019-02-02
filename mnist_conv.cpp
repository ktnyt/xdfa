#define XTENSOR_USE_XSIMD
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "xtensor/xarray.hpp"

#include "xnn/loaders/mnist.hpp"
#include "xnn/xnn.hpp"

namespace F = xnn::functions;
namespace L = xnn::layers;
namespace O = xnn::optimizers;

int main() {
  auto train_images_path = "mnist/train-images-idx3-ubyte";
  auto train_labels_path = "mnist/train-labels-idx1-ubyte";

  xt::xarray<float> x_train =
      mnist::read_images<float>(train_images_path, false) / 255.0;
  xt::xarray<int> t_train = mnist::read_labels<int>(train_labels_path);

  auto x_shape = x_train.shape();
  auto t_shape = t_train.shape();

  std::size_t n_train = x_shape[0];
  std::size_t n_input = x_shape[1];
  std::size_t n_output = 10;

  std::size_t n_epochs = 20;
  std::size_t batchsize = 100;
  std::size_t n_hidden = 1000;

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
      xt::xarray<int> t = xt::view(t_train, xt::range(i, i + batchsize));

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
