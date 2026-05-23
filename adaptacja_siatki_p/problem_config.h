#pragma once

#include <functional>
#include <string>
#include <vector>

using Function = std::function<double(double)>;

enum BoundaryType {
    DIRICHLET,
    NEUMANN
};

struct BoundaryCondition {
    BoundaryType type;
    double value;
};

struct ProblemConfig {
    int p_deg = 2;

    double domainLeft = 0.0;
    double domainRight = 1.0;
    int initialElements = 2;

    int maxSteps = 20;
    double TOL = 0.01;
    double alpha = 0.3;

    std::string aExpression = "1";
    std::string pExpression = "0";
    std::string qExpression = "1";
    std::string fExpression = "1";

    Function a_fun;
    Function p_fun;
    Function q_fun;
    Function f_fun;

    BoundaryCondition left = {DIRICHLET, 0.0};
    BoundaryCondition right = {DIRICHLET, 0.0};

    std::string leftBoundaryExpression = "0";
    std::string rightBoundaryExpression = "0";

    bool hasExactSolution = false;
    std::string exactExpression;
    Function u_exact;

    bool visualizationEnabled = true;
};

