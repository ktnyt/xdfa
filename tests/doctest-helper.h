#ifndef DOCTEST_HELPER_H
#define DOCTEST_HELPER_H

#ifndef CLOSE
#define CLOSE(a, e) REQUIRE(xt::abs(xt::sum(a))() < e)
#endif

#include <limits>

static constexpr float epsilon = std::numeric_limits<float>::epsilon();

#endif  // DOCTEST_HELPER_H
