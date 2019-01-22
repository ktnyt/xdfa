#define XTENSOR_USE_XSIMD
#include <cmath>
#include <iostream>
#include <vector>

#include "xnn/functions.hpp"
#include "xnn/initializers/matrix.hpp"
#include "xnn/loaders/mnist.hpp"
#include "xnn/optimizers.hpp"

namespace xfa = xnn::functions::activation;
namespace xfc = xnn::functions::connection;
namespace xfe = xnn::functions::evaluation;
namespace xfl = xnn::functions::loss;
namespace xfm = xnn::functions::miscellaneous;

namespace xop = xnn::optimizers;

int main() {
  auto train_images_path = "mnist/train-images-idx3-ubyte";
  auto train_labels_path = "mnist/train-labels-idx1-ubyte";

  xt::xarray<float> x_train =
      mnist::read_images<float>(train_images_path, true);
  xt::xarray<int> t_train = mnist::read_labels<int>(train_labels_path);

  auto x_shape = x_train.shape();
  std::size_t n_train = x_shape[0];
  std::size_t n_dims = x_shape[1];

  std::size_t n_epochs = 20;
  std::size_t batchsize = 100;
  std::size_t n_hidden = 1000;

  xfc::Linear l0(n_dims, n_hidden, xop::Adam());
  xfc::Linear l1(n_hidden, 10, xop::Adam());
  xfa::Sigmoid a0;
  xfl::SoftmaxCrossEntropy error;
  xfm::Combine<float> network({&l0, &a0, &l1});

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
      xt::xarray<float> t = xt::view(t_train, xt::range(i, i + batchsize));

      xt::xarray<float> y = network.forward(x);

      loss += error.with(t).forward(y) * batchsize;
      acc += xfe::accuracy(t, y) * batchsize;

      network.backward(error.grads());
      network.update();
    }

    std::cout << " Loss: " << loss / n_train << " Accuracy: " << acc / n_train
              << std::endl;
  }

  return 0;
}
