#include <memory>
#include <type_traits>

#include "xnn/functions.hpp"

namespace L = xnn::links;
namespace O = xnn::optimizers;

int main() {
  Variable x, t;

  L::Serial layer0({L::Linear(n_input, n_hidden), L::Sigmoid()});
  L::Linear layer1(n_hidden, n_output);

  // For direct feedback alignment.
  L::Serial feedback({L::Feedback(n_output, n_hidden), layer0});

  L::Serial forward({layer0, layer1});
  L::Parallel backward({feedback, layer1});

  L::SoftmaxCrossEntropy softmax_cross_entropy;

  // Asymmetrical backward calculation.
  O::SGD optimizer(forward, backward);

  Variable y = forward(x);
  Variable e = softmax_cross_entropy(t)(y);
  optimizer.update(e);

  return 0;
}
