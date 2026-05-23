#define FEM_NO_MAIN
#include "main.cpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

struct EquationCase {
    std::string name;
    int pDeg = 2;
    std::vector<double> mesh;
    Function a;
    Function p;
    Function q;
    Function f;
    Function exact;
    BoundaryCondition left;
    BoundaryCondition right;
    double maxDofError = 0.0;
};

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

std::vector<double> uniformMesh(int nElem) {
    std::vector<double> mesh;
    mesh.reserve(nElem + 1);

    for (int i = 0; i <= nElem; ++i) {
        mesh.push_back(static_cast<double>(i) / static_cast<double>(nElem));
    }

    return mesh;
}

double maxDofError(
    const Eigen::VectorXd& d,
    const std::vector<double>& dofCoords,
    const Function& exact
) {
    double maxError = 0.0;

    for (int i = 0; i < d.size(); ++i) {
        maxError = std::max(maxError, std::abs(d(i) - exact(dofCoords[i])));
    }

    return maxError;
}

void expectEstimatorIsSane(
    const std::string& name,
    const FEMResult& result,
    int nElem
) {
    expectTrue(name, std::isfinite(result.etaGlobal2), "global estimator is finite");
    expectTrue(name, result.etaGlobal2 >= -1e-10, "global estimator is non-negative");
    expectTrue(name, static_cast<int>(result.eta2List.size()) == nElem, "element estimator count");

    for (int e = 0; e < nElem; ++e) {
        expectTrue(name, std::isfinite(result.eta2List[e]), "element estimator is finite");
        expectTrue(name, result.eta2List[e] >= -1e-10, "element estimator is non-negative");
    }
}

void solveManufacturedCase(const EquationCase& eq) {
    FEMResult result = solveAndEstimate(
        eq.mesh,
        eq.pDeg,
        eq.a,
        eq.p,
        eq.q,
        eq.f,
        eq.left,
        eq.right
    );

    double error = maxDofError(result.d, result.dofCoords, eq.exact);
    expectTrue(eq.name, std::isfinite(error), "DOF error is finite");
    expectTrue(eq.name, error < eq.maxDofError, "DOF error is within tolerance");
    expectEstimatorIsSane(eq.name, result, static_cast<int>(eq.mesh.size()) - 1);
}

void testPoissonSineEquation(const std::string&) {
    EquationCase eq;
    eq.name = "Poisson sine equation";
    eq.pDeg = 2;
    eq.mesh = uniformMesh(16);
    eq.a = [](double) { return 1.0; };
    eq.p = [](double) { return 0.0; };
    eq.q = [](double) { return 0.0; };
    eq.f = [](double x) { return kPi * kPi * std::sin(kPi * x); };
    eq.exact = [](double x) { return std::sin(kPi * x); };
    eq.left = {DIRICHLET, 0.0};
    eq.right = {DIRICHLET, 0.0};
    eq.maxDofError = 8e-3;

    solveManufacturedCase(eq);
}

void testReactionDiffusionPolynomialEquation(const std::string&) {
    EquationCase eq;
    eq.name = "reaction-diffusion polynomial equation";
    eq.pDeg = 2;
    eq.mesh = uniformMesh(6);
    eq.a = [](double) { return 1.0; };
    eq.p = [](double) { return 0.0; };
    eq.q = [](double) { return 1.0; };
    eq.f = [](double x) { return 2.0 + x * (1.0 - x); };
    eq.exact = [](double x) { return x * (1.0 - x); };
    eq.left = {DIRICHLET, 0.0};
    eq.right = {DIRICHLET, 0.0};
    eq.maxDofError = 2e-3;

    solveManufacturedCase(eq);
}

void testAdvectionReactionPolynomialEquation(const std::string&) {
    EquationCase eq;
    eq.name = "advection-reaction polynomial equation";
    eq.pDeg = 2;
    eq.mesh = uniformMesh(8);
    eq.a = [](double) { return 1.0; };
    eq.p = [](double) { return 2.0; };
    eq.q = [](double) { return 1.0; };
    eq.f = [](double x) {
        double u = x * (1.0 - x);
        double du = 1.0 - 2.0 * x;
        return 2.0 + 2.0 * du + u;
    };
    eq.exact = [](double x) { return x * (1.0 - x); };
    eq.left = {DIRICHLET, 0.0};
    eq.right = {DIRICHLET, 0.0};
    eq.maxDofError = 3e-3;

    solveManufacturedCase(eq);
}

void testVariableDiffusionEquation(const std::string&) {
    EquationCase eq;
    eq.name = "variable-diffusion equation";
    eq.pDeg = 2;
    eq.mesh = uniformMesh(8);
    eq.a = [](double x) { return 1.0 + x; };
    eq.p = [](double) { return 0.0; };
    eq.q = [](double) { return 1.0; };
    eq.f = [](double x) {
        double u = x * (1.0 - x);
        return 1.0 + 4.0 * x + u;
    };
    eq.exact = [](double x) { return x * (1.0 - x); };
    eq.left = {DIRICHLET, 0.0};
    eq.right = {DIRICHLET, 0.0};
    eq.maxDofError = 3e-3;

    solveManufacturedCase(eq);
}

void testAdaptiveEstimatorDropsForSmoothEquation(const std::string& name) {
    Function one = [](double) { return 1.0; };
    Function zero = [](double) { return 0.0; };
    Function q = [](double) { return 1.0; };
    Function f = [](double x) {
        return kPi * kPi * std::sin(kPi * x) + std::sin(kPi * x);
    };

    BoundaryCondition left = {DIRICHLET, 0.0};
    BoundaryCondition right = {DIRICHLET, 0.0};

    std::vector<double> mesh = uniformMesh(4);
    FEMResult coarse = solveAndEstimate(mesh, 2, one, zero, q, f, left, right);

    std::vector<double> refined = refineMesh(mesh, coarse.eta2List, 0.3);
    FEMResult fine = solveAndEstimate(refined, 2, one, zero, q, f, left, right);

    expectEstimatorIsSane(name, coarse, static_cast<int>(mesh.size()) - 1);
    expectEstimatorIsSane(name, fine, static_cast<int>(refined.size()) - 1);
    expectTrue(name, refined.size() > mesh.size(), "adaptive refinement adds nodes");
    expectTrue(name, fine.etaGlobal2 < coarse.etaGlobal2, "global estimator decreases after refinement");
}

} // namespace

int main() {
    runTest("Poisson sine equation", testPoissonSineEquation);
    runTest("reaction-diffusion polynomial equation", testReactionDiffusionPolynomialEquation);
    runTest("advection-reaction polynomial equation", testAdvectionReactionPolynomialEquation);
    runTest("variable-diffusion equation", testVariableDiffusionEquation);
    runTest("adaptive estimator decrease", testAdaptiveEstimatorDropsForSmoothEquation);

    if (failures != 0) {
        std::cerr << failures << " whole-equation assertion(s) failed.\n";
        return EXIT_FAILURE;
    }

    std::cout << "All whole-equation tests passed.\n";
    return EXIT_SUCCESS;
}
