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
// Solver główny używa funkcji Lagrange'a.
// Lokalny estymator błędu używa hierarchicznych funkcji kształtu:
//     psi_1(s) = 1 - s,
//     psi_2(s) = s,
//     psi_i(0) = psi_i(1) = 0 dla i >= 3.
//
//
// Warunki brzegowe są elastyczne:
//
//     BoundaryCondition left  = {DIRICHLET, value};
//     BoundaryCondition right = {DIRICHLET, value};
//
// albo:
//
//     BoundaryCondition right = {NEUMANN, value};
//
// Uwaga:
// Dla NEUMANN value oznacza naturalny strumień a(x)u'(x)
// na brzegu. Dla a(x)=1 jest to po prostu u'(x).
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
const double INTEGRATION_STEP = 0.0005;

// ==========================================================
// Warunki brzegowe
// ==========================================================

enum BoundaryType {
    DIRICHLET,
    NEUMANN
};

struct BoundaryCondition {
    BoundaryType type;
    double value;
};

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

double referenceNode(int i, int degree) {
    validateDegree(degree);

    if (i < 0 || i > degree) {
        throw std::runtime_error("referenceNode: zły lokalny indeks.");
    }

    return static_cast<double>(i) / static_cast<double>(degree);
}

double shapeValue(int i, int degree, double s) {
    validateDegree(degree);

    if (i < 0 || i > degree) {
        throw std::runtime_error("shapeValue: zły lokalny indeks funkcji kształtu.");
    }

    double si = referenceNode(i, degree);
    double value = 1.0;

    for (int j = 0; j <= degree; ++j) {
        if (j == i) continue;

        double sj = referenceNode(j, degree);
        value *= (s - sj) / (si - sj);
    }

    return value;
}

// Pochodna funkcji kształtu liczona numerycznie po zmiennej s.
// Pochodną po x otrzymujemy później jako:
//
//     dN/dx = (dN/ds) / h
//
double shapeDerivNumerical(int i, int degree, double s) {
    double h = DERIV_STEP;

    if (s - h < 0.0) {
        return (shapeValue(i, degree, s + h)
              - shapeValue(i, degree, s)) / h;
    }

    if (s + h > 1.0) {
        return (shapeValue(i, degree, s)
              - shapeValue(i, degree, s - h)) / h;
    }

    return (shapeValue(i, degree, s + h)
          - shapeValue(i, degree, s - h)) / (2.0 * h);
}

// ==========================================================
// Hierarchiczne funkcje kształtu dla lokalnego estymatora błędu
//
// Ważne:
//   - solver główny nadal używa funkcji Lagrange'a,
//   - estymator błędu używa poniższej bazy hierarchicznej.
//
// Dwie pierwsze funkcje są liniowe i spełniają:
//
//     psi_0(s) + psi_1(s) = 1.
//
// Kolejne funkcje są funkcjami wewnętrznymi, czyli znikają na
// końcach elementu. Dzięki temu warunek równowagi elementu
// z note_mgr może być sprawdzany przez r(psi_0) + r(psi_1) = 0.
// ==========================================================

double hierShapeValue(int i, int degree, double s) {
    validateDegree(degree);

    if (i < 0 || i > degree) {
        throw std::runtime_error("hierShapeValue: zły lokalny indeks funkcji kształtu.");
    }

    if (i == 0) {
        return 1.0 - s;
    }

    if (i == 1) {
        return s;
    }

    if (i == 2) {
        return s * (1.0 - s);
    }

    if (i == 3) {
        return s * (1.0 - s) * (2.0 * s - 1.0);
    }

    throw std::runtime_error("hierShapeValue: nieobsługiwany indeks funkcji.");
}

// Pochodna hierarchicznej funkcji kształtu liczona numerycznie
// po zmiennej referencyjnej s. Pochodną po x otrzymujemy jako:
//
//     dpsi/dx = (dpsi/ds) / h.
//
double hierShapeDerivNumerical(int i, int degree, double s) {
    double h = DERIV_STEP;

    if (s - h < 0.0) {
        return (hierShapeValue(i, degree, s + h)
              - hierShapeValue(i, degree, s)) / h;
    }

    if (s + h > 1.0) {
        return (hierShapeValue(i, degree, s)
              - hierShapeValue(i, degree, s - h)) / h;
    }

    return (hierShapeValue(i, degree, s + h)
          - hierShapeValue(i, degree, s - h)) / (2.0 * h);
}

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
// Warunki brzegowe: część naturalna i podstawowa
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

void applyNaturalBoundaryConditions(
    Eigen::VectorXd& F,
    const BoundaryCondition& left,
    const BoundaryCondition& right
) {
    int rightDof = static_cast<int>(F.size()) - 1;

    // Dla równania:
    //
    //     -(a u')' + p u' + q u = f
    //
    // słaba forma ma po prawej stronie wkład:
    //
    //     + (a u')(1) v(1) - (a u')(0) v(0)
    //
    // Dlatego:
    //   lewy Neumann  -> F(0)        -= value
    //   prawy Neumann -> F(rightDof) += value
    //
    // value oznacza naturalny strumień a(x)u'(x).

    if (left.type == NEUMANN) {
        F(0) -= left.value;
    }

    if (right.type == NEUMANN) {
        F(rightDof) += right.value;
    }
}

void applyEssentialBoundaryConditions(
    Eigen::MatrixXd& K,
    Eigen::VectorXd& F,
    const BoundaryCondition& left,
    const BoundaryCondition& right
) {
    int rightDof = static_cast<int>(F.size()) - 1;

    if (left.type == DIRICHLET) {
        applyDirichlet(K, F, 0, left.value);
    }

    if (right.type == DIRICHLET) {
        applyDirichlet(K, F, rightDof, right.value);
    }
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
    const BoundaryCondition& left,
    const BoundaryCondition& right
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

    applyNaturalBoundaryConditions(F, left, right);
    applyEssentialBoundaryConditions(K, F, left, right);

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
// Lokalne formy z hierarchiczną funkcją testową.
//
// Te funkcje są używane wyłącznie w estymatorze błędu.
// u_h nadal jest rozwiązaniem z solvera głównego, więc jest
// rozwinięte w bazie Lagrange'a stopnia p_deg.
// ==========================================================

double localBuhvHierTest(
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

        double v = hierShapeValue(
            testLocal,
            testDegree,
            s
        );

        double dv_dx =
            hierShapeDerivNumerical(testLocal, testDegree, s) / h;

        return
            a_fun(x) * duh * dv_dx
            + p_fun(x) * duh * v
            + q_fun(x) * uh * v;
    };

    return h * calka(0.0, 1.0, integrand);
}

double localLvHierTest(
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

        double v = hierShapeValue(
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
    const BoundaryCondition& left,
    const BoundaryCondition& right
) {
    int nElem = static_cast<int>(meshNodes.size()) - 1;

    double a = meshNodes[elem];
    double b = meshNodes[elem + 1];
    double h = b - a;

    if (side == -1) {
        double x = a;

        // Lewy brzeg globalny.
        if (elem == 0 && left.type == NEUMANN) {
            // Normalna zewnętrzna po lewej stronie to n = -1.
            // left.value oznacza naturalny strumień a u'.
            return -left.value;
        }

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

        // Lewy Dirichlet: strumień nie jest zadany,
        // więc bierzemy jednostronny strumień z elementu.
        return fluxK * (-1.0);
    }

    if (side == +1) {
        double x = b;

        // Prawy brzeg globalny.
        if (elem == nElem - 1 && right.type == NEUMANN) {
            // Normalna zewnętrzna po prawej stronie to n = +1.
            // right.value oznacza naturalny strumień a u'.
            return right.value;
        }

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

        // Prawy Dirichlet: strumień nie jest zadany,
        // więc bierzemy jednostronny strumień z elementu.
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
    int p_err,
    const Function& a_fun,
    const Function& p_fun,
    const Function& q_fun,
    const Function& f_fun,
    const BoundaryCondition& left,
    const BoundaryCondition& right
) {
    validateDegree(p_err);

    ElementBoundaryData data;

    data.tLeft = averagedFluxOnElementBoundary(
        meshNodes,
        d,
        elem,
        p_deg,
        -1,
        a_fun,
        left,
        right
    );

    data.tRight = averagedFluxOnElementBoundary(
        meshNodes,
        d,
        elem,
        p_deg,
        +1,
        a_fun,
        left,
        right
    );

    // Warunek prolongacji/równowagi estymatora liczony jest dla
    // dwóch pierwszych funkcji hierarchicznych:
    //   psi_0 = 1 - s,
    //   psi_1 = s.
    // Nie używamy tutaj funkcji Lagrange'a stopnia p_deg.
    int leftLocal = 0;
    int rightLocal = 1;

    double B_left = localBuhvHierTest(
        meshNodes,
        d,
        elem,
        p_deg,
        p_err,
        leftLocal,
        a_fun,
        p_fun,
        q_fun
    );

    double L_left = localLvHierTest(
        meshNodes,
        elem,
        p_err,
        leftLocal,
        f_fun
    );

    double B_right = localBuhvHierTest(
        meshNodes,
        d,
        elem,
        p_deg,
        p_err,
        rightLocal,
        a_fun,
        p_fun,
        q_fun
    );

    double L_right = localLvHierTest(
        meshNodes,
        elem,
        p_err,
        rightLocal,
        f_fun
    );

    // Poprawiona konwencja znaków zgodna z:
    //   lambda_K(v) = theta_left v(left) + theta_right v(right)
    //               + t_left v(left) + t_right v(right),
    //   r_K(v) = B_K(u_h,v) - L_K(v) - lambda_K(v).
    // Z warunku r_K(psi_i)=0 dostajemy theta_i = B_i - L_i - t_i.
    data.thetaLeft = B_left - L_left - data.tLeft;
    data.thetaRight = B_right - L_right - data.tRight;

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

                double v_i = hierShapeValue(i, p_err, s);
                double phi_j = hierShapeValue(j, p_err, s);

                double dv_i_dx =
                    hierShapeDerivNumerical(i, p_err, s) / h;

                double dphi_j_dx =
                    hierShapeDerivNumerical(j, p_err, s) / h;

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
        double B = localBuhvHierTest(
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

        double L = localLvHierTest(
            meshNodes,
            elem,
            p_err,
            m,
            f_fun
        );

        double vLeft = hierShapeValue(m, p_err, 0.0);
        double vRight = hierShapeValue(m, p_err, 1.0);

        r(m) =
            B - L
            - bd.thetaLeft * vLeft
            - bd.thetaRight * vRight
            - bd.tLeft * vLeft
            - bd.tRight * vRight;
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
            return hierShapeValue(i, degree, s);
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
    std::vector<double> equilibriumList;

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
    const BoundaryCondition& left,
    const BoundaryCondition& right
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
        left,
        right
    );

    result.dofCoords = buildDofCoords(meshNodes, p_deg);

    result.residualList.resize(nElem);
    result.phiList.resize(nElem);
    result.eta2List.assign(nElem, 0.0);
    result.equilibriumList.assign(nElem, 0.0);

    double etaGlobal2 = 0.0;

    for (int e = 0; e < nElem; ++e) {
        double a = meshNodes[e];
        double b = meshNodes[e + 1];

        ElementBoundaryData bd = computeElementBoundaryData(
            meshNodes,
            result.d,
            e,
            p_deg,
            p_err,
            a_fun,
            p_fun,
            q_fun,
            f_fun,
            left,
            right
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

        // Dla hierarchicznej bazy estymatora pierwsze dwie funkcje
        // spełniają psi_0 + psi_1 = 1. Dlatego r(psi_0)+r(psi_1)
        // jest diagnostyką lokalnej równowagi r_K(1)=0.
        if (r.size() >= 2) {
            result.equilibriumList[e] = r(0) + r(1);
        }

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
    const std::vector<double>& eta2List,
    const std::vector<double>& equilibriumList = {}
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
                  << "  eta = " << std::setprecision(12) << eta;

        if (static_cast<int>(equilibriumList.size()) == nElem) {
            std::cout << "  r(psi0)+r(psi1) = "
                      << std::setprecision(4) << std::scientific
                      << equilibriumList[e]
                      << std::defaultfloat;
        }

        std::cout << "\n";
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

        const int p_deg = 2;
        const int p_err = p_deg + 1;

        validateDegree(p_deg);
        validateDegree(p_err);

        std::cout << "============================================\n";
        std::cout << "FEM 1D z p_deg = " << p_deg
                  << ", estymator p_err = " << p_err << "\n";
        std::cout << "Pochodne liczone numerycznie.\n";
        std::cout << "Calkowanie metoda prostokatow.\n";
        std::cout << "Estymator uzywa hierarchicznych funkcji ksztaltu.\n";
        std::cout << "============================================\n";

        // --------------------------------------------------
        // Przykład:
        //
        //     -u'' + u = f
        //
        // z rozwiązaniem dokładnym:
        //
        //     u(x) = tanh(k * (x - x0))
        // --------------------------------------------------

        Function a_fun = [](double /*x*/) {
            return 1.0;
        };

        const double k  = 10.0;
        const double x0 = 0.5;

        // q > 0 sprawia, że macierz A_K lokalnego problemu błędu
        // jest dodatnio określona.
        Function q_fun = [](double /*x*/) {
            return 1.0;
        };

        Function p_fun = [](double /*x*/) {
            return 0.0;
        };

        Function f_fun = [k, x0](double x) {
            double z = k * (x - x0);
            double t = std::tanh(z);
            double c = std::cosh(z);
            double sech2 = 1.0 / (c * c);

            return t * (1.0 + 2.0 * k * k * sech2);
        };

        Function u_exact = [k, x0](double x) {
            return std::tanh(k * (x - x0));
        };

        Function du_exact = [k, x0](double x) {
            double z = k * (x - x0);
            double c = std::cosh(z);
            double sech2 = 1.0 / (c * c);

            return k * sech2;
        };

        // --------------------------------------------------
        // Elastyczne warunki brzegowe.
        //
        // Dla DIRICHLET:
        //     value = u(x_boundary)
        //
        // Dla NEUMANN:
        //     value = a(x_boundary) * u'(x_boundary)
        //
        // Dla tego przykładu a(x)=1, więc NEUMANN value = u'(x).
        // --------------------------------------------------

        BoundaryCondition left = {
            DIRICHLET,
            u_exact(0.0)
        };

        BoundaryCondition right = {
            DIRICHLET,
            u_exact(1.0)
        };

        // Alternatywny wariant: lewy Dirichlet, prawy Neumann.
        //
        // BoundaryCondition left = {
        //     DIRICHLET,
        //     u_exact(0.0)
        // };
        //
        // BoundaryCondition right = {
        //     NEUMANN,
        //     a_fun(1.0) * du_exact(1.0)
        // };

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
                left,
                right
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
            printEta(meshNodes, result.eta2List, result.equilibriumList);

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