// ==========================================================
// main.cpp
//
// FEM 1D dla równania:
//
//     -u'' + p(x) u' + q(x) u = f(x)
//
// Obsługiwane:
//     p_deg = 1  -> liniowe funkcje kształtu
//     p_deg = 2  -> kwadratowe funkcje kształtu
//
// Estymator:
//     p_err = p_deg + 1
//
// Czyli:
//     p_deg = 1 -> estymator p_err = 2
//     p_deg = 2 -> estymator p_err = 3
//
// Ta wersja NIE używa kwadratury Gaussa.
// Pochodne są liczone numerycznie, tak jak w pierwotnym main.cpp.
// ==========================================================

#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iomanip>

#include "../Eigen/Dense"

#include "fem_visualizer.h"

using Function = std::function<double(double)>;

// ==========================================================
// Parametry numeryczne
// ==========================================================

const double DERIV_STEP = 1e-4;
const double INTEGRATION_STEP = 1e-3;

// ==========================================================
// Pochodna numeryczna i całkowanie prostokątami
// ==========================================================

double pochodna(
    double x,
    const Function& func,
    double h = DERIV_STEP
) {
    return (func(x + h) - func(x)) / h;
}

double calka(
    double start,
    double end,
    const Function& func,
    double delta = INTEGRATION_STEP
) {
    double res = 0.0;

    for (double x = start; x <= end; x += delta) {
        res += func(x) * delta;
    }

    return res;
}

// ==========================================================
// Stopnie swobody i indeksowanie
// ==========================================================

void validateDegree(int degree) {
    if (degree < 1 || degree > 3) {
        throw std::runtime_error(
            "Obsługiwane są stopnie degree = 1, 2, 3. "
            "Solver używa p_deg = 1 lub 2, a estymator p_err = p_deg + 1."
        );
    }
}

int nLocalDofs(int degree) {
    validateDegree(degree);
    return degree + 1;
}

int nGlobalDofs(int nElem, int p_deg) {
    validateDegree(p_deg);
    return nElem * p_deg + 1;
}

// Mapowanie lokalnego indeksu na globalny DOF.
// To jest kluczowe dla p_deg = 2.
int globalDof(int elem, int local, int p_deg) {
    return elem * p_deg + local;
}

// ==========================================================
// Węzły referencyjne i funkcje Lagrange'a na [0,1]
// ==========================================================

std::vector<double> referenceNodes(int degree) {
    validateDegree(degree);

    std::vector<double> nodes(degree + 1);

    for (int i = 0; i <= degree; ++i) {
        nodes[i] = static_cast<double>(i) / static_cast<double>(degree);
    }

    return nodes;
}

double shapeValue(int i, int degree, double s) {
    validateDegree(degree);

    if (i < 0 || i > degree) {
        throw std::runtime_error("shapeValue: zły lokalny indeks funkcji kształtu.");
    }

    auto nodes = referenceNodes(degree);

    double value = 1.0;

    for (int j = 0; j <= degree; ++j) {
        if (j == i) continue;

        value *= (s - nodes[j]) / (nodes[i] - nodes[j]);
    }

    return value;
}

// Pochodna funkcji kształtu liczona numerycznie po zmiennej s.
// Pochodną po x otrzymujemy później jako:
//
//     dN/dx = (dN/ds) / h
//
double shapeDerivNumerical(int i, int degree, double s) {
    Function Ni = [=](double ss) {
        return shapeValue(i, degree, ss);
    };

    return pochodna(s, Ni, DERIV_STEP);
}

// ==========================================================
// Globalne współrzędne stopni swobody
// ==========================================================

std::vector<double> buildDofCoords(
    const std::vector<double>& meshNodes,
    int p_deg
) {
    validateDegree(p_deg);

    int nElem = static_cast<int>(meshNodes.size()) - 1;
    int nDof = nGlobalDofs(nElem, p_deg);

    std::vector<double> dofCoords(nDof);

    for (int e = 0; e < nElem; ++e) {
        double a = meshNodes[e];
        double b = meshNodes[e + 1];
        double h = b - a;

        for (int i = 0; i <= p_deg; ++i) {
            int I = globalDof(e, i, p_deg);
            double s = static_cast<double>(i) / static_cast<double>(p_deg);

            dofCoords[I] = a + h * s;
        }
    }

    return dofCoords;
}

// ==========================================================
// Wartość u_h i pochodnej u_h' na elemencie
// ==========================================================

double uhValueOnElement(
    int elem,
    double s,
    int p_deg,
    const Eigen::VectorXd& d
) {
    validateDegree(p_deg);

    double value = 0.0;

    for (int i = 0; i <= p_deg; ++i) {
        int I = globalDof(elem, i, p_deg);
        value += d(I) * shapeValue(i, p_deg, s);
    }

    return value;
}

double duhDxOnElement(
    int elem,
    double s,
    double h,
    int p_deg,
    const Eigen::VectorXd& d
) {
    validateDegree(p_deg);

    double value = 0.0;

    for (int i = 0; i <= p_deg; ++i) {
        int I = globalDof(elem, i, p_deg);

        double dN_ds = shapeDerivNumerical(i, p_deg, s);
        double dN_dx = dN_ds / h;

        value += d(I) * dN_dx;
    }

    return value;
}

// ==========================================================
// Macierz elementowa solvera głównego
//
// B(u,v) = ∫ [ a u' v' + p u' v + q u v ] dx
// ==========================================================

Eigen::MatrixXd elementMatrix(
    double a,
    double b,
    int p_deg,
    const Function& a_fun,
    const Function& p_fun,
    const Function& q_fun
) {
    validateDegree(p_deg);

    int nLoc = nLocalDofs(p_deg);
    double h = b - a;

    Eigen::MatrixXd Ke = Eigen::MatrixXd::Zero(nLoc, nLoc);

    for (int i = 0; i < nLoc; ++i) {
        for (int j = 0; j < nLoc; ++j) {
            Function integrand = [&](double s) {
                double x = a + h * s;

                double v_i = shapeValue(i, p_deg, s);
                double u_j = shapeValue(j, p_deg, s);

                double dv_i_dx = shapeDerivNumerical(i, p_deg, s) / h;
                double du_j_dx = shapeDerivNumerical(j, p_deg, s) / h;

                return
                    a_fun(x) * du_j_dx * dv_i_dx
                    + p_fun(x) * du_j_dx * v_i
                    + q_fun(x) * u_j * v_i;
            };

            Ke(i, j) = h * calka(0.0, 1.0, integrand);
        }
    }

    return Ke;
}

// ==========================================================
// Wektor prawej strony elementu
//
// L(v) = ∫ f v dx
// ==========================================================

Eigen::VectorXd elementVector(
    double a,
    double b,
    int p_deg,
    const Function& f_fun
) {
    validateDegree(p_deg);

    int nLoc = nLocalDofs(p_deg);
    double h = b - a;

    Eigen::VectorXd Fe = Eigen::VectorXd::Zero(nLoc);

    for (int i = 0; i < nLoc; ++i) {
        Function integrand = [&](double s) {
            double x = a + h * s;
            return f_fun(x) * shapeValue(i, p_deg, s);
        };

        Fe(i) = h * calka(0.0, 1.0, integrand);
    }

    return Fe;
}

// ==========================================================
// Składanie globalnego układu
// ==========================================================

void assembleSystem(
    const std::vector<double>& meshNodes,
    int p_deg,
    const Function& a_fun,
    const Function& p_fun,
    const Function& q_fun,
    const Function& f_fun,
    Eigen::MatrixXd& K,
    Eigen::VectorXd& F
) {
    validateDegree(p_deg);

    int nElem = static_cast<int>(meshNodes.size()) - 1;
    int nDof = nGlobalDofs(nElem, p_deg);

    K = Eigen::MatrixXd::Zero(nDof, nDof);
    F = Eigen::VectorXd::Zero(nDof);

    for (int e = 0; e < nElem; ++e) {
        double a = meshNodes[e];
        double b = meshNodes[e + 1];

        Eigen::MatrixXd Ke = elementMatrix(
            a,
            b,
            p_deg,
            a_fun,
            p_fun,
            q_fun
        );

        Eigen::VectorXd Fe = elementVector(
            a,
            b,
            p_deg,
            f_fun
        );

        for (int i = 0; i <= p_deg; ++i) {
            int I = globalDof(e, i, p_deg);

            F(I) += Fe(i);

            for (int j = 0; j <= p_deg; ++j) {
                int J = globalDof(e, j, p_deg);

                K(I, J) += Ke(i, j);
            }
        }
    }
}

// ==========================================================
// Warunki brzegowe
// ==========================================================

void applyDirichlet(
    Eigen::MatrixXd& K,
    Eigen::VectorXd& F,
    int dof,
    double value
) {
    for (int i = 0; i < K.rows(); ++i) {
        F(i) -= K(i, dof) * value;
    }

    K.row(dof).setZero();
    K.col(dof).setZero();

    K(dof, dof) = 1.0;
    F(dof) = value;
}

void addRightNeumann(
    Eigen::VectorXd& F,
    double rightFlux
) {
    int rightDof = static_cast<int>(F.size()) - 1;
    F(rightDof) += rightFlux;
}

// ==========================================================
// Rozwiązanie układu FEM
// ==========================================================

Eigen::VectorXd solveFEM(
    const std::vector<double>& meshNodes,
    int p_deg,
    const Function& a_fun,
    const Function& p_fun,
    const Function& q_fun,
    const Function& f_fun,
    double leftDirichletValue,
    double rightNeumannFlux
) {
    Eigen::MatrixXd K;
    Eigen::VectorXd F;

    assembleSystem(
        meshNodes,
        p_deg,
        a_fun,
        p_fun,
        q_fun,
        f_fun,
        K,
        F
    );

    addRightNeumann(F, rightNeumannFlux);
    applyDirichlet(K, F, 0, leftDirichletValue);

    Eigen::VectorXd d = K.partialPivLu().solve(F);

    return d;
}

// ==========================================================
// Lokalna forma B_K(u_h, v)
//
// u_h jest w przestrzeni p_deg.
// v może być w przestrzeni p_deg albo p_err.
// ==========================================================

double localBuhv(
    const std::vector<double>& meshNodes,
    const Eigen::VectorXd& d,
    int elem,
    int p_deg,
    int testDegree,
    int testLocal,
    const Function& a_fun,
    const Function& p_fun,
    const Function& q_fun
) {
    validateDegree(p_deg);
    validateDegree(testDegree);

    double a = meshNodes[elem];
    double b = meshNodes[elem + 1];
    double h = b - a;

    Function integrand = [&](double s) {
        double x = a + h * s;

        double uh = uhValueOnElement(
            elem,
            s,
            p_deg,
            d
        );

        double duh = duhDxOnElement(
            elem,
            s,
            h,
            p_deg,
            d
        );

        double v = shapeValue(
            testLocal,
            testDegree,
            s
        );

        double dv_dx =
            shapeDerivNumerical(testLocal, testDegree, s) / h;

        return
            a_fun(x) * duh * dv_dx
            + p_fun(x) * duh * v
            + q_fun(x) * uh * v;
    };

    return h * calka(0.0, 1.0, integrand);
}

double localLv(
    const std::vector<double>& meshNodes,
    int elem,
    int testDegree,
    int testLocal,
    const Function& f_fun
) {
    validateDegree(testDegree);

    double a = meshNodes[elem];
    double b = meshNodes[elem + 1];
    double h = b - a;

    Function integrand = [&](double s) {
        double x = a + h * s;

        double v = shapeValue(
            testLocal,
            testDegree,
            s
        );

        return f_fun(x) * v;
    };

    return h * calka(0.0, 1.0, integrand);
}

// ==========================================================
// Dane brzegowe elementu do estymatora
// ==========================================================

struct ElementBoundaryData {
    double tLeft = 0.0;
    double tRight = 0.0;

    double thetaLeft = 0.0;
    double thetaRight = 0.0;
};

double averagedFluxOnElementBoundary(
    const std::vector<double>& meshNodes,
    const Eigen::VectorXd& d,
    int elem,
    int p_deg,
    int side,
    const Function& a_fun,
    double rightNeumannFlux,
    bool useRightNeumannFlux
) {
    int nElem = static_cast<int>(meshNodes.size()) - 1;

    double a = meshNodes[elem];
    double b = meshNodes[elem + 1];
    double h = b - a;

    if (side == -1) {
        double x = a;

        double duK = duhDxOnElement(
            elem,
            0.0,
            h,
            p_deg,
            d
        );

        double fluxK = a_fun(x) * duK;

        if (elem > 0) {
            double aL = meshNodes[elem - 1];
            double bL = meshNodes[elem];
            double hL = bL - aL;

            double duL = duhDxOnElement(
                elem - 1,
                1.0,
                hL,
                p_deg,
                d
            );

            double fluxL = a_fun(x) * duL;

            return 0.5 * (fluxK + fluxL) * (-1.0);
        }

        return fluxK * (-1.0);
    }

    if (side == +1) {
        double x = b;

        double duK = duhDxOnElement(
            elem,
            1.0,
            h,
            p_deg,
            d
        );

        double fluxK = a_fun(x) * duK;

        if (elem < nElem - 1) {
            double aR = meshNodes[elem + 1];
            double bR = meshNodes[elem + 2];
            double hR = bR - aR;

            double duR = duhDxOnElement(
                elem + 1,
                0.0,
                hR,
                p_deg,
                d
            );

            double fluxR = a_fun(x) * duR;

            return 0.5 * (fluxK + fluxR) * (+1.0);
        }

        if (useRightNeumannFlux) {
            return rightNeumannFlux;
        }

        return fluxK * (+1.0);
    }

    throw std::runtime_error(
        "averagedFluxOnElementBoundary: side musi być -1 albo +1."
    );
}

ElementBoundaryData computeElementBoundaryData(
    const std::vector<double>& meshNodes,
    const Eigen::VectorXd& d,
    int elem,
    int p_deg,
    const Function& a_fun,
    const Function& p_fun,
    const Function& q_fun,
    const Function& f_fun,
    double rightNeumannFlux
) {
    ElementBoundaryData data;

    data.tLeft = averagedFluxOnElementBoundary(
        meshNodes,
        d,
        elem,
        p_deg,
        -1,
        a_fun,
        rightNeumannFlux,
        true
    );

    data.tRight = averagedFluxOnElementBoundary(
        meshNodes,
        d,
        elem,
        p_deg,
        +1,
        a_fun,
        rightNeumannFlux,
        true
    );

    int leftLocal = 0;
    int rightLocal = p_deg;

    double B_left = localBuhv(
        meshNodes,
        d,
        elem,
        p_deg,
        p_deg,
        leftLocal,
        a_fun,
        p_fun,
        q_fun
    );

    double L_left = localLv(
        meshNodes,
        elem,
        p_deg,
        leftLocal,
        f_fun
    );

    double B_right = localBuhv(
        meshNodes,
        d,
        elem,
        p_deg,
        p_deg,
        rightLocal,
        a_fun,
        p_fun,
        q_fun
    );

    double L_right = localLv(
        meshNodes,
        elem,
        p_deg,
        rightLocal,
        f_fun
    );

    data.thetaLeft = B_left - L_left + data.tLeft;
    data.thetaRight = B_right - L_right + data.tRight;

    return data;
}

// ==========================================================
// Macierz lokalnego problemu błędu
//
// A_K(phi, v) = ∫ [ a phi' v' + q phi v ] dx
//
// Pochodne phi i v też liczone numerycznie.
// ==========================================================

Eigen::MatrixXd errorMatrix_AK(
    double a,
    double b,
    int p_err,
    const Function& a_fun,
    const Function& q_fun
) {
    validateDegree(p_err);

    int nErr = nLocalDofs(p_err);
    double h = b - a;

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(nErr, nErr);

    for (int i = 0; i < nErr; ++i) {
        for (int j = 0; j < nErr; ++j) {
            Function integrand = [&](double s) {
                double x = a + h * s;

                double v_i = shapeValue(i, p_err, s);
                double phi_j = shapeValue(j, p_err, s);

                double dv_i_dx =
                    shapeDerivNumerical(i, p_err, s) / h;

                double dphi_j_dx =
                    shapeDerivNumerical(j, p_err, s) / h;

                return
                    a_fun(x) * dphi_j_dx * dv_i_dx
                    + q_fun(x) * phi_j * v_i;
            };

            A(i, j) = h * calka(0.0, 1.0, integrand);
        }
    }

    return A;
}

// ==========================================================
// Wektor lokalnego residuum w przestrzeni p_err
// ==========================================================

Eigen::VectorXd computeLocalResidual(
    const std::vector<double>& meshNodes,
    const Eigen::VectorXd& d,
    int elem,
    int p_deg,
    int p_err,
    const Function& a_fun,
    const Function& p_fun,
    const Function& q_fun,
    const Function& f_fun,
    const ElementBoundaryData& bd
) {
    validateDegree(p_deg);
    validateDegree(p_err);

    int nErr = nLocalDofs(p_err);

    Eigen::VectorXd r = Eigen::VectorXd::Zero(nErr);

    for (int m = 0; m < nErr; ++m) {
        double B = localBuhv(
            meshNodes,
            d,
            elem,
            p_deg,
            p_err,
            m,
            a_fun,
            p_fun,
            q_fun
        );

        double L = localLv(
            meshNodes,
            elem,
            p_err,
            m,
            f_fun
        );

        double vLeft = shapeValue(m, p_err, 0.0);
        double vRight = shapeValue(m, p_err, 1.0);

        r(m) =
            B - L
            - bd.thetaLeft * vLeft
            - bd.thetaRight * vRight
            + bd.tLeft * vLeft
            + bd.tRight * vRight;
    }

    return r;
}

// ==========================================================
// Warunek zerowej średniej dla lokalnego problemu błędu
// przy q = 0.
// ==========================================================

Eigen::VectorXd meanConstraintVector(
    double a,
    double b,
    int degree
) {
    validateDegree(degree);

    int n = nLocalDofs(degree);
    double h = b - a;

    Eigen::VectorXd c = Eigen::VectorXd::Zero(n);

    for (int i = 0; i < n; ++i) {
        Function integrand = [&](double s) {
            return shapeValue(i, degree, s);
        };

        c(i) = h * calka(0.0, 1.0, integrand);
    }

    return c;
}

Eigen::VectorXd solveLocalErrorFunction(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& r,
    double a,
    double b,
    int p_err
) {
    int n = static_cast<int>(A.rows());

    Eigen::FullPivLU<Eigen::MatrixXd> rankCheck(A);
    rankCheck.setThreshold(1e-10);

    if (rankCheck.rank() == n) {
        return A.fullPivLu().solve(r);
    }

    Eigen::VectorXd c = meanConstraintVector(a, b, p_err);

    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n + 1, n + 1);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n + 1);

    M.block(0, 0, n, n) = A;

    for (int i = 0; i < n; ++i) {
        M(i, n) = c(i);
        M(n, i) = c(i);
    }

    rhs.head(n) = r;
    rhs(n) = 0.0;

    Eigen::VectorXd sol = M.fullPivLu().solve(rhs);

    return sol.head(n);
}

double computeEta2(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& phi
) {
    double eta2 = phi.transpose() * A * phi;

    if (!std::isfinite(eta2)) {
        return 0.0;
    }

    if (eta2 < 0.0 && std::abs(eta2) < 1e-10) {
        return 0.0;
    }

    return eta2;
}

// ==========================================================
// Norma energetyczna rozwiązania
// ==========================================================

double computeEnergyNormUh2(
    const std::vector<double>& meshNodes,
    const Eigen::VectorXd& d,
    int p_deg,
    const Function& a_fun,
    const Function& q_fun
) {
    validateDegree(p_deg);

    int nElem = static_cast<int>(meshNodes.size()) - 1;

    double norm2 = 0.0;

    for (int e = 0; e < nElem; ++e) {
        double a = meshNodes[e];
        double b = meshNodes[e + 1];
        double h = b - a;

        Function integrand = [&](double s) {
            double x = a + h * s;

            double uh = uhValueOnElement(
                e,
                s,
                p_deg,
                d
            );

            double duh = duhDxOnElement(
                e,
                s,
                h,
                p_deg,
                d
            );

            return a_fun(x) * duh * duh + q_fun(x) * uh * uh;
        };

        norm2 += h * calka(0.0, 1.0, integrand);
    }

    return norm2;
}

// ==========================================================
// Wynik jednego rozwiązania + estymacji
// ==========================================================

struct FEMResult {
    Eigen::VectorXd d;

    std::vector<double> dofCoords;

    std::vector<Eigen::VectorXd> residualList;
    std::vector<Eigen::VectorXd> phiList;
    std::vector<double> eta2List;

    double etaGlobal2 = 0.0;
    double uhEnergyNorm2 = 0.0;
};

FEMResult solveAndEstimate(
    const std::vector<double>& meshNodes,
    int p_deg,
    const Function& a_fun,
    const Function& p_fun,
    const Function& q_fun,
    const Function& f_fun,
    double leftDirichletValue,
    double rightNeumannFlux
) {
    validateDegree(p_deg);

    int p_err = p_deg + 1;
    validateDegree(p_err);

    int nElem = static_cast<int>(meshNodes.size()) - 1;

    FEMResult result;

    result.d = solveFEM(
        meshNodes,
        p_deg,
        a_fun,
        p_fun,
        q_fun,
        f_fun,
        leftDirichletValue,
        rightNeumannFlux
    );

    result.dofCoords = buildDofCoords(meshNodes, p_deg);

    result.residualList.resize(nElem);
    result.phiList.resize(nElem);
    result.eta2List.assign(nElem, 0.0);

    double etaGlobal2 = 0.0;

    for (int e = 0; e < nElem; ++e) {
        double a = meshNodes[e];
        double b = meshNodes[e + 1];

        ElementBoundaryData bd = computeElementBoundaryData(
            meshNodes,
            result.d,
            e,
            p_deg,
            a_fun,
            p_fun,
            q_fun,
            f_fun,
            rightNeumannFlux
        );

        Eigen::MatrixXd Aerr = errorMatrix_AK(
            a,
            b,
            p_err,
            a_fun,
            q_fun
        );

        Eigen::VectorXd r = computeLocalResidual(
            meshNodes,
            result.d,
            e,
            p_deg,
            p_err,
            a_fun,
            p_fun,
            q_fun,
            f_fun,
            bd
        );

        Eigen::VectorXd phi = solveLocalErrorFunction(
            Aerr,
            r,
            a,
            b,
            p_err
        );

        double eta2 = computeEta2(Aerr, phi);

        result.residualList[e] = r;
        result.phiList[e] = phi;
        result.eta2List[e] = eta2;

        etaGlobal2 += eta2;
    }

    result.etaGlobal2 = etaGlobal2;

    result.uhEnergyNorm2 = computeEnergyNormUh2(
        meshNodes,
        result.d,
        p_deg,
        a_fun,
        q_fun
    );

    return result;
}

// ==========================================================
// Adaptacja siatki h-adaptacyjna
// ==========================================================

std::vector<double> refineMesh(
    const std::vector<double>& meshNodes,
    const std::vector<double>& eta2List,
    double alpha
) {
    int nElem = static_cast<int>(meshNodes.size()) - 1;

    double etaMax = 0.0;
    int maxIndex = 0;

    for (int e = 0; e < nElem; ++e) {
        double eta = 0.0;

        if (std::isfinite(eta2List[e]) && eta2List[e] > 0.0) {
            eta = std::sqrt(eta2List[e]);
        }

        if (eta > etaMax) {
            etaMax = eta;
            maxIndex = e;
        }
    }

    double threshold = alpha * etaMax;

    std::vector<double> newMesh;
    newMesh.reserve(meshNodes.size() * 2);

    bool anyRefined = false;

    for (int e = 0; e < nElem; ++e) {
        double a = meshNodes[e];
        double b = meshNodes[e + 1];
        double mid = 0.5 * (a + b);

        double eta = 0.0;

        if (std::isfinite(eta2List[e]) && eta2List[e] > 0.0) {
            eta = std::sqrt(eta2List[e]);
        }

        newMesh.push_back(a);

        if (eta > threshold && etaMax > 0.0) {
            newMesh.push_back(mid);
            anyRefined = true;
        }
    }

    newMesh.push_back(meshNodes.back());

    if (!anyRefined && nElem > 0) {
        newMesh.clear();

        for (int e = 0; e < nElem; ++e) {
            double a = meshNodes[e];
            double b = meshNodes[e + 1];
            double mid = 0.5 * (a + b);

            newMesh.push_back(a);

            if (e == maxIndex) {
                newMesh.push_back(mid);
            }
        }

        newMesh.push_back(meshNodes.back());
    }

    return newMesh;
}

// ==========================================================
// Wypisywanie wyników
// ==========================================================

void printDofs(
    const std::vector<double>& dofCoords,
    const Eigen::VectorXd& d
) {
    std::cout << "Globalne DOF-y: x, u_h(x)\n";

    for (int i = 0; i < d.size(); ++i) {
        std::cout << "  dof " << std::setw(3) << i
                  << "  x = " << std::setw(14) << std::setprecision(10) << dofCoords[i]
                  << "  u = " << std::setw(14) << std::setprecision(10) << d(i)
                  << "\n";
    }
}

void printEta(
    const std::vector<double>& meshNodes,
    const std::vector<double>& eta2List
) {
    std::cout << "Estymatory elementowe eta_K:\n";

    int nElem = static_cast<int>(meshNodes.size()) - 1;

    for (int e = 0; e < nElem; ++e) {
        double eta = 0.0;

        if (std::isfinite(eta2List[e]) && eta2List[e] > 0.0) {
            eta = std::sqrt(eta2List[e]);
        }

        std::cout << "  elem " << std::setw(3) << e
                  << "  [" << std::setprecision(8) << meshNodes[e]
                  << ", " << std::setprecision(8) << meshNodes[e + 1] << "]"
                  << "  eta = " << std::setprecision(12) << eta
                  << "\n";
    }
}

// ==========================================================
// main
// ==========================================================

int main() {
    try {
        // --------------------------------------------------
        // Wybór stopnia funkcji kształtu:
        //
        //     p_deg = 1 -> funkcje liniowe
        //     p_deg = 2 -> funkcje kwadratowe
        // --------------------------------------------------

        const int p_deg = 2; // zmień na 2, żeby użyć funkcji kwadratowych
        const int p_err = p_deg + 1;

        validateDegree(p_deg);
        validateDegree(p_err);

        std::cout << "============================================\n";
        std::cout << "FEM 1D z p_deg = " << p_deg
                  << ", estymator p_err = " << p_err << "\n";
        std::cout << "Pochodne liczone numerycznie.\n";
        std::cout << "Calkowanie metoda prostokatow.\n";
        std::cout << "============================================\n";

        // --------------------------------------------------
        // Przykład:
        //
        //     u'' + 6x^2 = 0
        //
        // czyli:
        //
        //     -u'' = 6x^2
        //
        // Warunki:
        //
        //     u(0) = 1
        //     u'(1) = -1/2
        //
        // Rozwiązanie analityczne:
        //
        //     u(x) = -0.5 x^4 + 1.5 x + 1
        // --------------------------------------------------

        Function a_fun = [](double /*x*/) {
            return 1.0;
        };

        const double k  = 10.0;
        const double x0 = 0.5;

        // q > 0 sprawia ze macierz A_K lokalnego problemu bledu
        // jest dodatnio okreslona (bez osobliwosci).
        auto q_fun = [](double x) {
            return 1.0;
        };

        auto p_fun = [](double x) {
        return 0.0;
        };
        
        // Rownanie: -u'' + u = f
        // u(x)   = tanh(k*(x-x0))
        // u''(x) = -2k^2 * tanh(k*(x-x0)) * sech^2(k*(x-x0))
        // f(x)   = -u'' + u = tanh(k*(x-x0)) * (1 + 2k^2 * sech^2(k*(x-x0)))
        auto f_fun = [k, x0](double x) {
            double t     = std::tanh(k * (x - x0));
            double sech2 = 1.0 / (std::cosh(k * (x - x0)) * std::cosh(k * (x - x0)));
            return t * (1.0 + 2.0 * k * k * sech2);
        };

        Function u_exact = [&](double x) {
            return std::tanh(k*(x-x0));
        };

        const double leftDirichletValue = u_exact(0.0);
        const double rightNeumannFlux =
            k / (std::cosh(k * (1.0 - x0)) * std::cosh(k * (1.0 - x0)));

        std::vector<double> meshNodes = {
            0.0,
            0.5,
            1.0
        };

        const int maxSteps = 20;
        const double TOL = 0.01;
        const double alpha = 0.3;

        FEMVisualizer viz;
        for (int step = 0; step < maxSteps; ++step) {
            FEMResult result = solveAndEstimate(
                meshNodes,
                p_deg,
                a_fun,
                p_fun,
                q_fun,
                f_fun,
                leftDirichletValue,
                rightNeumannFlux
            );

            double etaGlobal = std::sqrt(std::max(0.0, result.etaGlobal2));
            double uhEnergyNorm = std::sqrt(std::max(0.0, result.uhEnergyNorm2));
            double toleranceLevel = TOL * uhEnergyNorm;

            viz.plotStep(
                step,
                meshNodes,
                result.dofCoords,
                result.d,
                result.eta2List,
                etaGlobal,
                uhEnergyNorm,
                TOL,
                alpha,
                p_deg
            );

            std::cout << "\n--------------------------------------------\n";
            std::cout << "Krok adaptacji: " << step << "\n";
            std::cout << "Liczba elementow: " << meshNodes.size() - 1 << "\n";
            std::cout << "Liczba DOF: " << result.d.size() << "\n";
            std::cout << "etaGlobal = " << std::setprecision(12) << etaGlobal << "\n";
            std::cout << "||u_h||_E = " << std::setprecision(12) << uhEnergyNorm << "\n";
            std::cout << "TOL * ||u_h||_E = " << std::setprecision(12) << toleranceLevel << "\n";

            printDofs(result.dofCoords, result.d);
            printEta(meshNodes, result.eta2List);

            std::cout << "Porownanie z rozwiazaniem analitycznym w DOF-ach:\n";

            for (int i = 0; i < result.d.size(); ++i) {
                double x = result.dofCoords[i];
                double err = result.d(i) - u_exact(x);

                std::cout << "  x = " << std::setw(12) << std::setprecision(8) << x
                          << "  uh = " << std::setw(14) << std::setprecision(10) << result.d(i)
                          << "  u_exact = " << std::setw(14) << std::setprecision(10) << u_exact(x)
                          << "  err = " << std::setw(14) << std::setprecision(10) << err
                          << "\n";
            }

            if (etaGlobal <= toleranceLevel) {
                std::cout << "\nWarunek stopu spelniony.\n";
                break;
            }

            if (step + 1 < maxSteps) {
                meshNodes = refineMesh(
                    meshNodes,
                    result.eta2List,
                    alpha
                );
            }
        }

        return 0;
    }
    catch (const std::exception& ex) {
        std::cerr << "Blad: " << ex.what() << "\n";
        return 1;
    }
}