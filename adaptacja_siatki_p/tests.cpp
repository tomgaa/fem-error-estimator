#define FEM_NO_MAIN
#include "main.cpp"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

namespace {

constexpr double kStrictTol = 1e-8;
constexpr double kSolverTol = 5e-3;

int failures = 0;

void fail(const std::string& testName, const std::string& message) {
    ++failures;
    std::cerr << "[FAIL] " << testName << ": " << message << "\n";
}

void expectTrue(
    const std::string& testName,
    bool condition,
    const std::string& message
) {
    if (!condition) {
        fail(testName, message);
    }
}

void expectNear(
    const std::string& testName,
    double actual,
    double expected,
    double tol,
    const std::string& label
) {
    if (std::abs(actual - expected) > tol) {
        std::ostringstream msg;
        msg << label << " expected " << expected
            << " got " << actual
            << " with tolerance " << tol;
        fail(testName, msg.str());
    }
}

void runTest(
    const std::string& name,
    const std::function<void(const std::string&)>& test
) {
    int before = failures;

    try {
        test(name);
    }
    catch (const std::exception& ex) {
        fail(name, ex.what());
    }

    if (failures == before) {
        std::cout << "[PASS] " << name << "\n";
    }
}

void testShapeInterpolation(const std::string& name) {
    for (int degree = 1; degree <= 3; ++degree) {
        for (int i = 0; i <= degree; ++i) {
            for (int j = 0; j <= degree; ++j) {
                double expected = (i == j) ? 1.0 : 0.0;
                double actual = shapeValue(i, degree, referenceNode(j, degree));
                expectNear(name, actual, expected, kStrictTol, "shape interpolation");
            }
        }

        const double samplePoints[] = {0.0, 0.17, 0.5, 0.83, 1.0};

        for (double s : samplePoints) {
            double sum = 0.0;

            for (int i = 0; i <= degree; ++i) {
                sum += shapeValue(i, degree, s);
            }

            expectNear(name, sum, 1.0, kStrictTol, "partition of unity");
        }
    }
}

void testHierarchicalEndpointValues(const std::string& name) {
    for (int degree = 2; degree <= 3; ++degree) {
        expectNear(name, hierShapeValue(0, degree, 0.0), 1.0, kStrictTol, "left endpoint");
        expectNear(name, hierShapeValue(0, degree, 1.0), 0.0, kStrictTol, "left endpoint");
        expectNear(name, hierShapeValue(1, degree, 0.0), 0.0, kStrictTol, "right endpoint");
        expectNear(name, hierShapeValue(1, degree, 1.0), 1.0, kStrictTol, "right endpoint");

        for (int i = 2; i <= degree; ++i) {
            expectNear(name, hierShapeValue(i, degree, 0.0), 0.0, kStrictTol, "bubble left endpoint");
            expectNear(name, hierShapeValue(i, degree, 1.0), 0.0, kStrictTol, "bubble right endpoint");
        }
    }
}

void testDofCoordinates(const std::string& name) {
    std::vector<double> mesh = {0.0, 0.5, 1.0};

    std::vector<double> linear = buildDofCoords(mesh, 1);
    std::vector<double> expectedLinear = {0.0, 0.5, 1.0};

    expectTrue(name, linear.size() == expectedLinear.size(), "linear DOF count");

    for (int i = 0; i < static_cast<int>(expectedLinear.size()); ++i) {
        expectNear(name, linear[i], expectedLinear[i], kStrictTol, "linear DOF coordinate");
    }

    std::vector<double> quadratic = buildDofCoords(mesh, 2);
    std::vector<double> expectedQuadratic = {0.0, 0.25, 0.5, 0.75, 1.0};

    expectTrue(name, quadratic.size() == expectedQuadratic.size(), "quadratic DOF count");

    for (int i = 0; i < static_cast<int>(expectedQuadratic.size()); ++i) {
        expectNear(name, quadratic[i], expectedQuadratic[i], kStrictTol, "quadratic DOF coordinate");
    }
}

void expectLinearSolution(
    const std::string& name,
    int pDeg,
    const BoundaryCondition& left,
    const BoundaryCondition& right
) {
    std::vector<double> mesh = {0.0, 0.25, 0.5, 0.75, 1.0};

    Function one = [](double) { return 1.0; };
    Function zero = [](double) { return 0.0; };

    Eigen::VectorXd d = solveFEM(
        mesh,
        pDeg,
        one,
        zero,
        zero,
        zero,
        left,
        right
    );

    std::vector<double> dofCoords = buildDofCoords(mesh, pDeg);

    for (int i = 0; i < d.size(); ++i) {
        expectNear(name, d(i), dofCoords[i], kSolverTol, "linear exact solution");
    }
}

void testLinearDirichletSolution(const std::string& name) {
    BoundaryCondition left = {DIRICHLET, 0.0};
    BoundaryCondition right = {DIRICHLET, 1.0};

    expectLinearSolution(name, 1, left, right);
    expectLinearSolution(name, 2, left, right);
}

void testLinearNeumannSolution(const std::string& name) {
    BoundaryCondition left = {DIRICHLET, 0.0};
    BoundaryCondition right = {NEUMANN, 1.0};

    expectLinearSolution(name, 1, left, right);
    expectLinearSolution(name, 2, left, right);
}

void testEstimatorSanity(const std::string& name) {
    std::vector<double> mesh = {0.0, 0.5, 1.0};

    Function one = [](double) { return 1.0; };
    Function zero = [](double) { return 0.0; };
    Function source = [](double x) { return 2.0 + x; };

    BoundaryCondition left = {DIRICHLET, 0.0};
    BoundaryCondition right = {DIRICHLET, 0.0};

    FEMResult result = solveAndEstimate(
        mesh,
        2,
        one,
        zero,
        one,
        source,
        left,
        right
    );

    expectTrue(name, std::isfinite(result.etaGlobal2), "global estimator is finite");
    expectTrue(name, result.etaGlobal2 >= 0.0, "global estimator is non-negative");
    expectTrue(name, result.eta2List.size() == mesh.size() - 1, "element estimator count");
    expectTrue(name, result.equilibriumList.size() == mesh.size() - 1, "equilibrium count");

    for (int e = 0; e < static_cast<int>(result.eta2List.size()); ++e) {
        expectTrue(name, std::isfinite(result.eta2List[e]), "element estimator is finite");
        expectTrue(name, result.eta2List[e] >= -1e-10, "element estimator is non-negative");
        expectNear(name, result.equilibriumList[e], 0.0, 1e-6, "local equilibrium");
    }
}

void testRefineMesh(const std::string& name) {
    std::vector<double> mesh = {0.0, 0.5, 1.0};
    std::vector<double> refined = refineMesh(mesh, {100.0, 1.0}, 0.3);
    std::vector<double> expected = {0.0, 0.25, 0.5, 1.0};

    expectTrue(name, refined.size() == expected.size(), "refined mesh size");

    for (int i = 0; i < static_cast<int>(expected.size()); ++i) {
        expectNear(name, refined[i], expected[i], kStrictTol, "refined mesh coordinate");
    }

    std::vector<double> fallback = refineMesh(mesh, {0.0, 0.0}, 0.3);

    expectTrue(name, fallback.size() == 4, "fallback refinement size");

    for (int i = 1; i < static_cast<int>(fallback.size()); ++i) {
        expectTrue(name, fallback[i] > fallback[i - 1], "refined mesh is strictly increasing");
    }
}

} // namespace

int main() {
    runTest("shape interpolation", testShapeInterpolation);
    runTest("hierarchical endpoint values", testHierarchicalEndpointValues);
    runTest("DOF coordinates", testDofCoordinates);
    runTest("linear Dirichlet solution", testLinearDirichletSolution);
    runTest("linear Neumann solution", testLinearNeumannSolution);
    runTest("estimator sanity", testEstimatorSanity);
    runTest("mesh refinement", testRefineMesh);

    if (failures != 0) {
        std::cerr << failures << " test assertion(s) failed.\n";
        return EXIT_FAILURE;
    }

    std::cout << "All tests passed.\n";
    return EXIT_SUCCESS;
}
