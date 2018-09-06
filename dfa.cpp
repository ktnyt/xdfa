#define XTENSOR_USE_XSIMD
#include <cmath>
#include <iostream>
#include <vector>

#include "mnist.hpp"
#include "xdfa/xdfa.hpp"

using namespace xdfa;

int main() {
  auto train_images_path = "mnist/train-images-idx3-ubyte";
  auto train_labels_path = "mnist/train-labels-idx1-ubyte";

  Arrayf x_train = mnist::read_images<float>(train_images_path, true);
  Arrayi t_train = mnist::read_labels<int>(train_labels_path);

  auto x_shape = x_train.shape();
  std::size_t n_train = x_shape[0];
  std::size_t n_dims = x_shape[1];

  std::size_t n_epochs = 20;
  std::size_t batchsize = 100;

  Layer l0(n_dims, 240);
  Layer l1(240, 10);
  Sigmoid a0;
  SoftmaxCrossEntropy error;

  Combine network({&l0, &a0, &l1});

  for (std::size_t epoch = 0; epoch < n_epochs; ++epoch) {
    std::cout << "Epoch " << epoch << std::flush;

    Arrayf loss = 0.0;
    Arrayf acc = 0.0;

    xt::random::seed(epoch);
    xt::random::shuffle(x_train);

    t_train.resize({n_train, 1});
    xt::random::seed(epoch);
    xt::random::shuffle(t_train);
    t_train.resize({n_train});

    for (std::size_t i = 0; i < n_train; i += batchsize) {
      Arrayf x = xt::view(x_train, xt::range(i, i + batchsize));
      Arrayf t = xt::view(t_train, xt::range(i, i + batchsize));

      Arrayf y = network.forward(x);

      loss += error.with(t).forward(y) * batchsize;
      acc += accuracy(t, y) * batchsize;

      network.backward(error.grads());
    }

    std::cout << " Loss: " << loss / n_train << " Accuracy: " << acc / n_train
              << std::endl;
  }

  return 0;
}
