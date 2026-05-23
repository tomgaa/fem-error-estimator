#include "terminal_ui.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "third_party/tinyexpr/tinyexpr.h"

namespace {

class CompiledExpression {
public:
    explicit CompiledExpression(const std::string& text)
        : text_(text)
    {
        te_variable vars[] = {
            {"x", &x_, TE_VARIABLE, nullptr}
        };

        int error = 0;
        expr_ = te_compile(text_.c_str(), vars, 1, &error);

        if (!expr_) {
            std::ostringstream msg;
            msg << "Could not parse expression near position " << error << ".";
            throw std::runtime_error(msg.str());
        }
    }

    ~CompiledExpression() {
        te_free(expr_);
    }

    CompiledExpression(const CompiledExpression&) = delete;
    CompiledExpression& operator=(const CompiledExpression&) = delete;

    double eval(double x) const {
        x_ = x;
        return te_eval(expr_);
    }

private:
    std::string text_;
    mutable double x_ = 0.0;
    te_expr* expr_ = nullptr;
};

Function makeFunction(const std::string& text) {
    auto expr = std::make_shared<CompiledExpression>(text);

    return [expr](double x) {
        return expr->eval(x);
    };
}

std::string trim(const std::string& value) {
    std::size_t first = 0;
    while (first < value.size() &&
           std::isspace(static_cast<unsigned char>(value[first]))) {
        ++first;
    }

    std::size_t last = value.size();
    while (last > first &&
           std::isspace(static_cast<unsigned char>(value[last - 1]))) {
        --last;
    }

    return value.substr(first, last - first);
}

std::string askLine(const std::string& prompt, const std::string& defaultValue) {
    while (true) {
        std::cout << prompt;

        if (!defaultValue.empty()) {
            std::cout << " [" << defaultValue << "]";
        }

        std::cout << ": ";

        std::string line;
        if (!std::getline(std::cin, line)) {
            throw std::runtime_error("Input stream closed.");
        }

        line = trim(line);

        if (!line.empty()) {
            return line;
        }

        if (!defaultValue.empty()) {
            return defaultValue;
        }

        std::cout << "Please enter a value.\n";
    }
}

bool parseInt(const std::string& text, int& out) {
    std::istringstream in(text);
    int value = 0;
    in >> value;

    if (!in || !in.eof()) {
        return false;
    }

    out = value;
    return true;
}

bool parseDouble(const std::string& text, double& out) {
    std::istringstream in(text);
    double value = 0.0;
    in >> value;

    if (!in || !in.eof() || !std::isfinite(value)) {
        return false;
    }

    out = value;
    return true;
}

int askInt(
    const std::string& prompt,
    int defaultValue,
    int minValue,
    int maxValue
) {
    while (true) {
        int value = 0;
        std::string text = askLine(prompt, std::to_string(defaultValue));

        if (!parseInt(text, value)) {
            std::cout << "Please enter a whole number.\n";
            continue;
        }

        if (value < minValue || value > maxValue) {
            std::cout << "Please enter a value from "
                      << minValue << " to " << maxValue << ".\n";
            continue;
        }

        return value;
    }
}

double askDouble(
    const std::string& prompt,
    double defaultValue,
    double minValue,
    double maxValue
) {
    while (true) {
        double value = 0.0;

        std::ostringstream defaultText;
        defaultText << std::setprecision(12) << defaultValue;

        std::string text = askLine(prompt, defaultText.str());

        if (!parseDouble(text, value)) {
            std::cout << "Please enter a finite number.\n";
            continue;
        }

        if (value < minValue || value > maxValue) {
            std::cout << "Please enter a value from "
                      << minValue << " to " << maxValue << ".\n";
            continue;
        }

        return value;
    }
}

bool askYesNo(const std::string& prompt, bool defaultValue) {
    while (true) {
        std::string defaultText = defaultValue ? "Y/n" : "y/N";
        std::string answer = askLine(prompt, defaultText);

        std::transform(
            answer.begin(),
            answer.end(),
            answer.begin(),
            [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            }
        );

        if (answer == "y" || answer == "yes" || answer == "Y/n") {
            return true;
        }

        if (answer == "n" || answer == "no" || answer == "y/N") {
            return false;
        }

        if (answer == "y/n") {
            return defaultValue;
        }

        std::cout << "Please answer y or n.\n";
    }
}

std::string askExpression(
    const std::string& prompt,
    const std::string& defaultExpression,
    double testX
) {
    while (true) {
        std::string text = askLine(prompt, defaultExpression);

        try {
            Function f = makeFunction(text);
            double value = f(testX);

            if (!std::isfinite(value)) {
                std::cout << "This expression evaluates to NaN or Inf at x = "
                          << testX << ". Please try another expression.\n";
                continue;
            }

            return text;
        }
        catch (const std::exception& ex) {
            std::cout << ex.what() << "\n";
        }
    }
}

BoundaryType askBoundaryType(
    const std::string& side,
    BoundaryType defaultType
) {
    int defaultChoice = defaultType == DIRICHLET ? 1 : 2;

    std::cout << "\n" << side << " boundary condition:\n";
    std::cout << "  1. Dirichlet: u = value\n";
    std::cout << "  2. Neumann:   a*u' = value\n";

    int choice = askInt("Choice", defaultChoice, 1, 2);
    return choice == 1 ? DIRICHLET : NEUMANN;
}

BoundaryCondition askBoundaryCondition(
    const std::string& side,
    double xBoundary,
    BoundaryType defaultType,
    const std::string& defaultExpression,
    std::string& expressionOut
) {
    BoundaryCondition condition;
    condition.type = askBoundaryType(side, defaultType);

    expressionOut = askExpression(
        side + " boundary value/expression",
        defaultExpression,
        xBoundary
    );

    condition.value = makeFunction(expressionOut)(xBoundary);
    return condition;
}

std::vector<double> sampleGrid(double left, double right, int count) {
    std::vector<double> xs;
    xs.reserve(count);

    for (int i = 0; i < count; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(count - 1);
        xs.push_back(left + (right - left) * t);
    }

    return xs;
}

void warnAboutCoefficient(
    const std::string& name,
    const Function& f,
    double left,
    double right,
    bool shouldBePositive,
    bool shouldBeNonnegative
) {
    for (double x : sampleGrid(left, right, 41)) {
        double value = f(x);

        if (!std::isfinite(value)) {
            std::cout << "Warning: " << name
                      << " is NaN or Inf near x = " << x << ".\n";
            return;
        }

        if (shouldBePositive && value <= 0.0) {
            std::cout << "Warning: " << name
                      << " is <= 0 near x = " << x
                      << ". The FEM matrix may be invalid.\n";
            return;
        }

        if (shouldBeNonnegative && value < 0.0) {
            std::cout << "Warning: " << name
                      << " is < 0 near x = " << x
                      << ". The error estimator may be less robust.\n";
            return;
        }
    }
}

const char* boundaryTypeName(BoundaryType type) {
    return type == DIRICHLET ? "Dirichlet" : "Neumann";
}

void printSummary(const ProblemConfig& config) {
    std::cout << "\n============================================\n";
    std::cout << "Problem summary\n";
    std::cout << "Equation:\n";
    std::cout << "  -(a(x) u'(x))' + p(x) u'(x) + q(x) u(x) = f(x)\n";
    std::cout << "a(x): " << config.aExpression << "\n";
    std::cout << "p(x): " << config.pExpression << "\n";
    std::cout << "q(x): " << config.qExpression << "\n";
    std::cout << "f(x): " << config.fExpression << "\n";
    std::cout << "Domain: [" << config.domainLeft
              << ", " << config.domainRight << "]\n";
    std::cout << "Initial elements: " << config.initialElements << "\n";
    std::cout << "Polynomial degree: " << config.p_deg << "\n";
    std::cout << "Max steps: " << config.maxSteps << "\n";
    std::cout << "Tolerance: " << config.TOL << "\n";
    std::cout << "Refinement alpha: " << config.alpha << "\n";
    std::cout << "Left BC: " << boundaryTypeName(config.left.type)
              << ", value = " << config.left.value
              << " from " << config.leftBoundaryExpression << "\n";
    std::cout << "Right BC: " << boundaryTypeName(config.right.type)
              << ", value = " << config.right.value
              << " from " << config.rightBoundaryExpression << "\n";

    if (config.hasExactSolution) {
        std::cout << "Exact solution: " << config.exactExpression << "\n";
    }
    else {
        std::cout << "Exact solution: disabled\n";
    }

    std::cout << "Visualization: "
              << (config.visualizationEnabled ? "enabled" : "disabled")
              << "\n";
    std::cout << "============================================\n";
}

ProblemConfig tanhDefaults() {
    ProblemConfig config;
    config.p_deg = 2;
    config.domainLeft = 0.0;
    config.domainRight = 1.0;
    config.initialElements = 2;
    config.maxSteps = 20;
    config.TOL = 0.01;
    config.alpha = 0.3;
    config.aExpression = "1";
    config.pExpression = "0";
    config.qExpression = "1";
    config.fExpression =
        "tanh(10*(x-0.5))*(1+200/(cosh(10*(x-0.5))^2))";
    config.leftBoundaryExpression = "tanh(10*(x-0.5))";
    config.rightBoundaryExpression = "tanh(10*(x-0.5))";
    config.hasExactSolution = true;
    config.exactExpression = "tanh(10*(x-0.5))";

    return config;
}

ProblemConfig simpleDefaults() {
    ProblemConfig config;
    config.p_deg = 1;
    config.domainLeft = 0.0;
    config.domainRight = 1.0;
    config.initialElements = 4;
    config.maxSteps = 10;
    config.TOL = 0.01;
    config.alpha = 0.3;
    config.aExpression = "1";
    config.pExpression = "0";
    config.qExpression = "0";
    config.fExpression = "1";
    config.leftBoundaryExpression = "0";
    config.rightBoundaryExpression = "0";
    config.hasExactSolution = false;

    return config;
}

} // namespace

ProblemConfig askProblemConfigFromTerminal() {
    std::cout << "============================================\n";
    std::cout << "1D FEM solver setup\n";
    std::cout << "Equation form:\n";
    std::cout << "  -(a(x) u'(x))' + p(x) u'(x) + q(x) u(x) = f(x)\n";
    std::cout << "Expressions may use x, pi, e, +, -, *, /, ^, and functions like\n";
    std::cout << "sin, cos, tanh, exp, sqrt, log, cosh.\n";
    std::cout << "============================================\n";

    ProblemConfig defaults =
        askYesNo("Start from the built-in tanh-layer example?", true)
            ? tanhDefaults()
            : simpleDefaults();

    ProblemConfig config = defaults;

    std::cout << "\nDomain and mesh\n";
    config.domainLeft = askDouble(
        "Domain left endpoint",
        defaults.domainLeft,
        -std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max()
    );
    config.domainRight = askDouble(
        "Domain right endpoint",
        defaults.domainRight,
        -std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max()
    );

    while (config.domainRight <= config.domainLeft) {
        std::cout << "The right endpoint must be greater than the left endpoint.\n";
        config.domainRight = askDouble(
            "Domain right endpoint",
            defaults.domainRight,
            -std::numeric_limits<double>::max(),
            std::numeric_limits<double>::max()
        );
    }

    config.initialElements = askInt(
        "Initial number of elements",
        defaults.initialElements,
        1,
        1000000
    );

    std::cout << "\nEquation coefficients\n";
    double testX = 0.5 * (config.domainLeft + config.domainRight);
    config.aExpression = askExpression("a(x)", defaults.aExpression, testX);
    config.pExpression = askExpression("p(x)", defaults.pExpression, testX);
    config.qExpression = askExpression("q(x)", defaults.qExpression, testX);
    config.fExpression = askExpression("f(x)", defaults.fExpression, testX);

    config.a_fun = makeFunction(config.aExpression);
    config.p_fun = makeFunction(config.pExpression);
    config.q_fun = makeFunction(config.qExpression);
    config.f_fun = makeFunction(config.fExpression);

    std::cout << "\nBoundary conditions\n";
    config.left = askBoundaryCondition(
        "Left",
        config.domainLeft,
        defaults.left.type,
        defaults.leftBoundaryExpression,
        config.leftBoundaryExpression
    );
    config.right = askBoundaryCondition(
        "Right",
        config.domainRight,
        defaults.right.type,
        defaults.rightBoundaryExpression,
        config.rightBoundaryExpression
    );

    std::cout << "\nFEM and adaptation settings\n";
    config.p_deg = askInt("Polynomial degree, 1 or 2", defaults.p_deg, 1, 2);
    config.maxSteps = askInt("Maximum adaptation steps", defaults.maxSteps, 1, 10000);
    config.TOL = askDouble(
        "Relative tolerance",
        defaults.TOL,
        0.0,
        std::numeric_limits<double>::max()
    );
    config.alpha = askDouble("Refinement alpha", defaults.alpha, 0.0, 1.0);

    std::cout << "\nExact solution comparison\n";
    config.hasExactSolution = askYesNo(
        "Compare against an exact solution?",
        defaults.hasExactSolution
    );

    if (config.hasExactSolution) {
        config.exactExpression = askExpression(
            "u_exact(x)",
            defaults.exactExpression,
            testX
        );
        config.u_exact = makeFunction(config.exactExpression);
    }

    std::cout << "\nVisualization\n";
    config.visualizationEnabled = askYesNo(
        "Enable gnuplot visualization?",
        defaults.visualizationEnabled
    );

    std::cout << "\nChecking coefficient samples on the domain...\n";
    warnAboutCoefficient("a(x)", config.a_fun, config.domainLeft, config.domainRight, true, false);
    warnAboutCoefficient("q(x)", config.q_fun, config.domainLeft, config.domainRight, false, true);
    warnAboutCoefficient("p(x)", config.p_fun, config.domainLeft, config.domainRight, false, false);
    warnAboutCoefficient("f(x)", config.f_fun, config.domainLeft, config.domainRight, false, false);

    printSummary(config);

    if (!askYesNo("Run solver?", true)) {
        std::cout << "Cancelled.\n";
        std::exit(0);
    }

    return config;
}

