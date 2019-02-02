if (BUILD_XNN_TESTS)
  ENABLE_TESTING()
  INCLUDE_DIRECTORIES(.)
  ADD_EXECUTABLE(
    main_tests
    main.cpp
    utils/convolution.cpp
    functions/connection/convolution.cpp
    functions/pooling/average_pooling.cpp
    functions/pooling/max_pooling.cpp
    layers/pooling/average_pooling.cpp
    layers/pooling/max_pooling.cpp
  )
  TARGET_LINK_LIBRARIES(main_tests blas)
  ADD_TEST(NAME main COMMAND main_tests)
endif()
