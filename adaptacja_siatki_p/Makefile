CXX ?= c++
CC ?= cc

CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pedantic
CFLAGS ?= -O2 -Wall -Wextra -DTE_NAT_LOG -DTE_POW_FROM_RIGHT

TARGET = fem_solver
TEST_TARGET = fem_tests
EQUATION_TEST_TARGET = fem_equation_tests
OBJS = main.o terminal_ui.o third_party/tinyexpr/tinyexpr.o

.PHONY: all clean test unit-test equation-test

all: $(TARGET)

test: unit-test equation-test

unit-test: $(TEST_TARGET)
	./$(TEST_TARGET)

equation-test: $(EQUATION_TEST_TARGET)
	./$(EQUATION_TEST_TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET)

$(TEST_TARGET): tests.cpp main.cpp fem_visualizer.h problem_config.h
	$(CXX) $(CXXFLAGS) tests.cpp -o $(TEST_TARGET)

$(EQUATION_TEST_TARGET): equation_tests.cpp main.cpp fem_visualizer.h problem_config.h
	$(CXX) $(CXXFLAGS) equation_tests.cpp -o $(EQUATION_TEST_TARGET)

main.o: main.cpp fem_visualizer.h problem_config.h terminal_ui.h
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

terminal_ui.o: terminal_ui.cpp terminal_ui.h problem_config.h third_party/tinyexpr/tinyexpr.h
	$(CXX) $(CXXFLAGS) -c terminal_ui.cpp -o terminal_ui.o

third_party/tinyexpr/tinyexpr.o: third_party/tinyexpr/tinyexpr.c third_party/tinyexpr/tinyexpr.h
	$(CC) $(CFLAGS) -c third_party/tinyexpr/tinyexpr.c -o third_party/tinyexpr/tinyexpr.o

clean:
	rm -f $(TARGET) $(TEST_TARGET) $(EQUATION_TEST_TARGET) $(OBJS)
