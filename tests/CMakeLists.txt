if (BUILD_XNN_TESTS)
  ENABLE_TESTING()
  INCLUDE_DIRECTORIES(.)
  ADD_EXECUTABLE(
    main_tests
    main.cpp
    functions/connection/convolution.cpp
    functions/pooling/average_pooling.cpp
    utils/convolution.cpp
  )
  TARGET_LINK_LIBRARIES(main_tests blas)
  ADD_TEST(NAME main COMMAND main_tests)
endif()
