if (BUILD_XNN_TESTS)
  ENABLE_TESTING()
  INCLUDE_DIRECTORIES(tests)
  ADD_EXECUTABLE(
    main_tests
    main.cpp
    utils/convolution.cpp
  )
  ADD_TEST(NAME main COMMAND main_tests)
endif()
