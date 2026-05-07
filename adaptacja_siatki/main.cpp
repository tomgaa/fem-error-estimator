#include <iostream>
#include <vector>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include "../Eigen/Dense"
#include "fem_visualizer.h"

// ==========================================================
// Typy warunków brzegowych
// ==========================================================
enum BCType { DIRICHLET, NEUMANN };

struct BoundaryCondition {
    BCType type;
    double value;
};

// ==========================================================
// Prosta całka numeryczna — metoda prostokątów ze środkiem
// ==========================================================
double calka(
    double a,
    double b,
    std::function<double(double)> f,
    double dx = 1e-4
) {
    if (b <= a) {
        return 0.0;
    }

    double res = 0.0;

    for (double x = a; x < b; x += dx) {
        double xMid = x + 0.5 * dx;

        if (xMid > b) {
            xMid = 0.5 * (x + b);
        }

        double localDx = std::min(dx, b - x);
        res += f(xMid) * localDx;
    }

    return res;
}

// ==========================================================
// Funkcje kształtu liniowe na elemencie [a,b]
// ==========================================================
double N1(double x, double a, double b) {
    return (b - x) / (b - a);
}

double N2(double x, double a, double b) {
    return (x - a) / (b - a);
}

double dN1_dx(double a, double b) {
    return -1.0 / (b - a);
}

double dN2_dx(double a, double b) {
    return 1.0 / (b - a);
}

// ==========================================================
// Funkcje bazowe kwadratowe na elemencie referencyjnym s in [0,1]
// Węzły: s = 0, s = 0.5, s = 1
// ==========================================================
double N1_B(double s) {
    return 2.0 * (s - 0.5) * (s - 1.0);
}

double N2_B(double s) {
    return 4.0 * s * (1.0 - s);
}

double N3_B(double s) {
    return 2.0 * s * (s - 0.5);
}

double dN1_B(double s) {
    return 4.0 * s - 3.0;
}

double dN2_B(double s) {
    return 4.0 - 8.0 * s;
}

double dN3_B(double s) {
    return 4.0 * s - 1.0;
}

double NB(int i, double s) {
    switch (i) {
        case 0: return N1_B(s);
        case 1: return N2_B(s);
        case 2: return N3_B(s);
        default:
            throw std::runtime_error("Niepoprawny indeks funkcji kwadratowej NB.");
    }
}

double dNB_ds(int i, double s) {
    switch (i) {
        case 0: return dN1_B(s);
        case 1: return dN2_B(s);
        case 2: return dN3_B(s);
        default:
            throw std::runtime_error("Niepoprawny indeks pochodnej funkcji kwadratowej dNB.");
    }
}

// ==========================================================
// Wartość rozwiązania MES na elemencie
// ==========================================================
double uhValue(
    double x,
    double a,
    double b,
    double ui,
    double uj
) {
    return ui * N1(x, a, b) + uj * N2(x, a, b);
}

// ==========================================================
// Pochodna rozwiązania MES na elemencie
// ==========================================================
double duOnElement(
    const std::vector<double>& nodes,
    const Eigen::VectorXd& d,
    int k
) {
    double h = nodes[k + 1] - nodes[k];
    return (d(k + 1) - d(k)) / h;
}

// ==========================================================
// Średni strumień t_K na lewym i prawym brzegu elementu
// Dla a(x)=1.
// ==========================================================
double tBar(double duK, double duNeighbor, double normal) {
    return 0.5 * (duK + duNeighbor) * normal;
}

// ==========================================================
// Macierz elementowa standardowego MES, funkcje liniowe
//
// Dla równania:
// -u'' + p u' + q u = f
// ==========================================================
std::vector<std::vector<double>> K_e(
    double a,
    double b,
    std::function<double(double)> p,
    std::function<double(double)> q
) {
    double h = b - a;

    std::vector<std::vector<double>> K(2, std::vector<double>(2, 0.0));

    // Część od -u'', czyli int u' v' dx
    K[0][0] +=  1.0 / h;
    K[0][1] += -1.0 / h;
    K[1][0] += -1.0 / h;
    K[1][1] +=  1.0 / h;

    auto Ni = [&](int i, double x) {
        return (i == 0) ? N1(x, a, b) : N2(x, a, b);
    };

    auto dNj = [&](int j) {
        return (j == 0) ? dN1_dx(a, b) : dN2_dx(a, b);
    };

    // Część od p(x) * u'
    // int p(x) u' v dx
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            auto integrand = [&](double x) {
                return p(x) * Ni(i, x) * dNj(j);
            };

            K[i][j] += calka(a, b, integrand);
        }
    }

    // Część od q(x) * u
    // int q(x) u v dx
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            auto integrand = [&](double x) {
                return q(x) * Ni(i, x) * Ni(j, x);
            };

            K[i][j] += calka(a, b, integrand);
        }
    }

    return K;
}

// ==========================================================
// Wektor prawej strony elementu standardowego MES
// ==========================================================
std::vector<double> p_e(
    double a,
    double b,
    std::function<double(double)> f
) {
    std::vector<double> p(2, 0.0);

    p[0] = calka(a, b, [&](double x) {
        return f(x) * N1(x, a, b);
    });

    p[1] = calka(a, b, [&](double x) {
        return f(x) * N2(x, a, b);
    });

    return p;
}

// ==========================================================
// Assembly globalnego układu MES
// ==========================================================
void assemble(
    const std::vector<double>& nodes,
    std::function<double(double)> p_fun,
    std::function<double(double)> q_fun,
    std::function<double(double)> f_fun,
    Eigen::MatrixXd& K,
    Eigen::VectorXd& P
) {
    int n = static_cast<int>(nodes.size());

    K = Eigen::MatrixXd::Zero(n, n);
    P = Eigen::VectorXd::Zero(n);

    for (int e = 0; e < n - 1; e++) {
        double a = nodes[e];
        double b = nodes[e + 1];

        auto Ke = K_e(a, b, p_fun, q_fun);
        auto pe = p_e(a, b, f_fun);

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                K(e + i, e + j) += Ke[i][j];
            }

            P(e + i) += pe[i];
        }
    }
}

// ==========================================================
// Warunki brzegowe
// ==========================================================
void applyBC(
    Eigen::MatrixXd& K,
    Eigen::VectorXd& P,
    BoundaryCondition left,
    BoundaryCondition right
) {
    int n = static_cast<int>(K.rows());

    // Lewy brzeg
    if (left.type == DIRICHLET) {
        double val = left.value;

        for (int i = 0; i < n; i++) {
            P(i) -= K(i, 0) * val;
        }

        K.row(0).setZero();
        K.col(0).setZero();

        K(0, 0) = 1.0;
        P(0) = val;
    }

    if (left.type == NEUMANN) {
        P(0) += -left.value;
    }

    // Prawy brzeg
    if (right.type == DIRICHLET) {
        double val = right.value;

        for (int i = 0; i < n; i++) {
            P(i) -= K(i, n - 1) * val;
        }

        K.row(n - 1).setZero();
        K.col(n - 1).setZero();

        K(n - 1, n - 1) = 1.0;
        P(n - 1) = val;
    }

    if (right.type == NEUMANN) {
        P(n - 1) += right.value;
    }
}

// ==========================================================
// Macierz lokalnego problemu błędu A_K
//
// Szukamy phi_K w przestrzeni kwadratowej:
// B_K(phi_K, v) = r_K(v)
//
// Tutaj B_K(phi,v) = int_K phi' v' dx + int_K q phi v dx.
//
// Uwaga dydaktyczna:
// Dla q = 0 macierz A_K jest osobliwa, bo stała funkcja ma zerową
// normę energetyczną na elemencie. To nie jest błąd programu,
// tylko cecha problemu. Rozwiązanie phi_K jest wtedy wyznaczone
// z dokładnością do stałej, a eta_K pozostaje dobrze określone.
// ==========================================================
Eigen::Matrix3d errorMatrix_AK(
    double a,
    double b,
    std::function<double(double)> q
) {
    double h = b - a;

    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            auto integrand = [&](double s) {
                double x = a + h * s;

                double dNi_dx = dNB_ds(i, s) / h;
                double dNj_dx = dNB_ds(j, s) / h;

                double Ni = NB(i, s);
                double Nj = NB(j, s);

                return dNi_dx * dNj_dx + q(x) * Ni * Nj;
            };

            A(i, j) = h * calka(0.0, 1.0, integrand);
        }
    }

    return A;
}

// ==========================================================
// Struktura przechowująca dane brzegowe elementu potrzebne
// do samo-zrównoważonego residuum
// ==========================================================
struct ElementBoundaryData {
    double tLeft;
    double tRight;
    double theta1;
    double theta2;
};

// ==========================================================
// Obliczanie tLeft, tRight dla elementu k
// ==========================================================
std::pair<double, double> computeTForElement(
    const std::vector<double>& nodes,
    const Eigen::VectorXd& d,
    int k,
    BoundaryCondition leftBC,
    BoundaryCondition rightBC
) {
    int nElem = static_cast<int>(nodes.size()) - 1;

    double duK = duOnElement(nodes, d, k);

    double duLeft = duK;
    double duRight = duK;

    // Sąsiad z lewej
    if (k > 0) {
        duLeft = duOnElement(nodes, d, k - 1);
    } else {
        if (leftBC.type == NEUMANN) {
            duLeft = leftBC.value;
        }
    }

    // Sąsiad z prawej
    if (k < nElem - 1) {
        duRight = duOnElement(nodes, d, k + 1);
    } else {
        if (rightBC.type == NEUMANN) {
            duRight = rightBC.value;
        }
    }

    double tLeft = tBar(duK, duLeft, -1.0);
    double tRight = tBar(duK, duRight, 1.0);

    return {tLeft, tRight};
}

// ==========================================================
// Obliczanie theta1, theta2
//
// theta1 = B_K(u_h, psi1) - L_K(psi1) + tLeft
// theta2 = B_K(u_h, psi2) - L_K(psi2) + tRight
//
// gdzie psi1, psi2 to liniowe funkcje kształtu elementu.
// ==========================================================
ElementBoundaryData computeElementBoundaryData(
    const std::vector<double>& nodes,
    const Eigen::VectorXd& d,
    int k,
    std::function<double(double)> q_fun,
    std::function<double(double)> f_fun,
    BoundaryCondition leftBC,
    BoundaryCondition rightBC
) {
    double a = nodes[k];
    double b = nodes[k + 1];
    double h = b - a;

    double ui = d(k);
    double uj = d(k + 1);

    double duK = duOnElement(nodes, d, k);

    auto [tLeft, tRight] = computeTForElement(nodes, d, k, leftBC, rightBC);

    double dPsi1_dx = -1.0 / h;
    double dPsi2_dx =  1.0 / h;

    // B_K(u_h, psi1)
    double B1 = calka(a, b, [&](double x) {
        double uh = uhValue(x, a, b, ui, uj);
        return duK * dPsi1_dx + q_fun(x) * uh * N1(x, a, b);
    });

    // B_K(u_h, psi2)
    double B2 = calka(a, b, [&](double x) {
        double uh = uhValue(x, a, b, ui, uj);
        return duK * dPsi2_dx + q_fun(x) * uh * N2(x, a, b);
    });

    // L_K(psi1)
    double L1 = calka(a, b, [&](double x) {
        return f_fun(x) * N1(x, a, b);
    });

    // L_K(psi2)
    double L2 = calka(a, b, [&](double x) {
        return f_fun(x) * N2(x, a, b);
    });

    ElementBoundaryData data;

    data.tLeft = tLeft;
    data.tRight = tRight;

    data.theta1 = B1 - L1 + tLeft;
    data.theta2 = B2 - L2 + tRight;

    return data;
}

// ==========================================================
// Obliczanie lokalnego residuum r_K w przestrzeni kwadratowej
//
// Uwaga znakowa:
// korzystamy z wersji zgodnej z computeElementBoundaryData:
// r_K(v) = B_K(u_h,v) - L_K(v)
//          - theta1*v(0) - theta2*v(1)
//          + tLeft*v(0) + tRight*v(1)
//
// Dla v = psi1 oraz v = psi2 residuum powinno zanikać.
// ==========================================================
Eigen::Vector3d computeLocalResidual(
    const std::vector<double>& nodes,
    const Eigen::VectorXd& d,
    int k,
    std::function<double(double)> q_fun,
    std::function<double(double)> f_fun,
    const ElementBoundaryData& data
) {
    double a = nodes[k];
    double b = nodes[k + 1];
    double h = b - a;

    double ui = d(k);
    double uj = d(k + 1);

    double duK = duOnElement(nodes, d, k);

    Eigen::Vector3d r = Eigen::Vector3d::Zero();

    for (int m = 0; m < 3; m++) {
        // B_K(u_h, NB_m)
        double Bm = h * calka(0.0, 1.0, [&](double s) {
            double x = a + h * s;

            double uh = uhValue(x, a, b, ui, uj);
            double v = NB(m, s);
            double dv_dx = dNB_ds(m, s) / h;

            return duK * dv_dx + q_fun(x) * uh * v;
        });

        // L_K(NB_m)
        double Lm = h * calka(0.0, 1.0, [&](double s) {
            double x = a + h * s;
            return f_fun(x) * NB(m, s);
        });

        // Wartości funkcji testowej na końcach elementu
        double vLeft = NB(m, 0.0);
        double vRight = NB(m, 1.0);

        double thetaPart =
            data.theta1 * vLeft +
            data.theta2 * vRight;

        double tPart =
            data.tLeft * vLeft +
            data.tRight * vRight;

        r(m) = Bm - Lm - thetaPart + tPart;
    }

    return r;
}

// ==========================================================
// Obliczanie współczynników lokalnej funkcji błędu phi_K
// A_K * b_K = r_K
// ==========================================================
Eigen::Vector3d computeLocalErrorFunction(
    const std::vector<double>& nodes,
    const Eigen::VectorXd& d,
    int k,
    std::function<double(double)> q_fun,
    std::function<double(double)> f_fun,
    BoundaryCondition leftBC,
    BoundaryCondition rightBC,
    ElementBoundaryData& boundaryDataOut,
    Eigen::Matrix3d& AOut,
    Eigen::Vector3d& rOut
) {
    double a = nodes[k];
    double b = nodes[k + 1];

    boundaryDataOut = computeElementBoundaryData(
        nodes,
        d,
        k,
        q_fun,
        f_fun,
        leftBC,
        rightBC
    );

    AOut = errorMatrix_AK(a, b, q_fun);

    rOut = computeLocalResidual(
        nodes,
        d,
        k,
        q_fun,
        f_fun,
        boundaryDataOut
    );

    // colPivHouseholderQr radzi sobie także z osobliwością przy q = 0.
    // Otrzymana funkcja phi_K może różnić się o stałą, ale eta_K nie zależy
    // od tej stałej, bo jest liczona w normie energetycznej.
    Eigen::Vector3d bK = AOut.colPivHouseholderQr().solve(rOut);

    return bK;
}

// ==========================================================
// Norma energetyczna lokalnej funkcji błędu
//
// eta_K^2 = B_K(phi_K, phi_K)
//         = b_K^T A_K b_K
// ==========================================================
double computeEta2(
    const Eigen::Vector3d& bK,
    const Eigen::Matrix3d& A
) {
    double eta2 = (bK.transpose() * A * bK)(0, 0);

    if (eta2 < 0.0 && std::abs(eta2) < 1e-12) {
        eta2 = 0.0;
    }

    return eta2;
}

// ==========================================================
// Norma energetyczna rozwiązania MES
//
// ||u_h||_E^2 = suma_K int_K (u_h')^2 + q u_h^2 dx
//
// Ta norma jest potrzebna do kryterium:
// eta <= TOL * ||u_h||_E
// ==========================================================
double computeEnergyNormUh2(
    const std::vector<double>& nodes,
    const Eigen::VectorXd& d,
    std::function<double(double)> q_fun
) {
    int nElem = static_cast<int>(nodes.size()) - 1;

    double norm2 = 0.0;

    for (int k = 0; k < nElem; k++) {
        double a = nodes[k];
        double b = nodes[k + 1];

        double ui = d(k);
        double uj = d(k + 1);
        double duK = duOnElement(nodes, d, k);

        norm2 += calka(a, b, [&](double x) {
            double uh = uhValue(x, a, b, ui, uj);
            return duK * duK + q_fun(x) * uh * uh;
        });
    }

    if (norm2 < 0.0 && std::abs(norm2) < 1e-12) {
        norm2 = 0.0;
    }

    return norm2;
}

// ==========================================================
// Wyniki jednego kroku obliczeń: rozwiązanie + estymatory
// ==========================================================
struct FEMResult {
    Eigen::VectorXd d;
    std::vector<Eigen::Vector3d> bKList;
    std::vector<double> eta2List;
    double etaGlobal2;
    double uhEnergyNorm2;
};

// ==========================================================
// Jeden pełny krok:
// 1. assembly,
// 2. warunki brzegowe,
// 3. rozwiązanie,
// 4. estymator błędu.
// ==========================================================
FEMResult solveAndEstimate(
    const std::vector<double>& nodes,
    std::function<double(double)> p_fun,
    std::function<double(double)> q_fun,
    std::function<double(double)> f_fun,
    BoundaryCondition left,
    BoundaryCondition right,
    bool printSolutionDetails,
    bool printErrorDetails
) {
    Eigen::MatrixXd K;
    Eigen::VectorXd P;

    assemble(nodes, p_fun, q_fun, f_fun, K, P);
    applyBC(K, P, left, right);

    Eigen::VectorXd d = K.colPivHouseholderQr().solve(P);

    int nElem = static_cast<int>(nodes.size()) - 1;

    std::vector<Eigen::Vector3d> bKList;
    std::vector<double> eta2List;

    double etaGlobal2 = 0.0;

    if (printSolutionDetails) {
        std::cout << "Rozwiazanie MES:\n";
        for (int i = 0; i < d.size(); i++) {
            std::cout << "u" << i << " = " << d(i) << "\n";
        }

        std::cout << "\n";
        std::cout << "Obliczanie estymatora bledu:\n";
    }

    for (int k = 0; k < nElem; k++) {
        ElementBoundaryData boundaryData;
        Eigen::Matrix3d A;
        Eigen::Vector3d rK;

        Eigen::Vector3d bK = computeLocalErrorFunction(
            nodes,
            d,
            k,
            q_fun,
            f_fun,
            left,
            right,
            boundaryData,
            A,
            rK
        );

        double eta2 = computeEta2(bK, A);

        bKList.push_back(bK);
        eta2List.push_back(eta2);

        etaGlobal2 += eta2;

        if (printErrorDetails) {
            std::cout << "Element K" << k << " = ["
                      << nodes[k] << ", " << nodes[k + 1] << "]\n";

            std::cout << "  duK    = " << duOnElement(nodes, d, k) << "\n";
            std::cout << "  tLeft  = " << boundaryData.tLeft << "\n";
            std::cout << "  tRight = " << boundaryData.tRight << "\n";

            std::cout << "  theta1 = " << boundaryData.theta1 << "\n";
            std::cout << "  theta2 = " << boundaryData.theta2 << "\n";

            std::cout << "  r_K:\n";
            for (int i = 0; i < 3; i++) {
                std::cout << "    r[" << i << "] = " << rK(i) << "\n";
            }

            std::cout << "  b_K, czyli wspolczynniki phi_K:\n";
            for (int i = 0; i < 3; i++) {
                std::cout << "    b[" << i << "] = " << bK(i) << "\n";
            }

            std::cout << "  eta_K^2 = " << eta2 << "\n";
            std::cout << "  eta_K   = " << std::sqrt(eta2) << "\n";
            std::cout << "\n";
        }
    }

    double uhEnergyNorm2 = computeEnergyNormUh2(nodes, d, q_fun);

    FEMResult result;
    result.d = d;
    result.bKList = bKList;
    result.eta2List = eta2List;
    result.etaGlobal2 = etaGlobal2;
    result.uhEnergyNorm2 = uhEnergyNorm2;

    return result;
}

// ==========================================================
// Podział elementów wybranych przez kryterium:
// eta_K > alpha * etaMax
//
// Element spełniający kryterium jest dzielony w połowie.
// ==========================================================
std::vector<double> refineMesh(
    const std::vector<double>& nodes,
    const std::vector<double>& eta2List,
    double alpha
) {
    int nElem = static_cast<int>(nodes.size()) - 1;

    double etaMax = 0.0;

    for (int k = 0; k < nElem; k++) {
        etaMax = std::max(etaMax, std::sqrt(eta2List[k]));
    }

    std::vector<double> newNodes;
    newNodes.push_back(nodes[0]);

    for (int k = 0; k < nElem; k++) {
        double a = nodes[k];
        double b = nodes[k + 1];

        double etaK = std::sqrt(eta2List[k]);

        if (etaK > alpha * etaMax) {
            double mid = 0.5 * (a + b);
            newNodes.push_back(mid);
        }

        newNodes.push_back(b);
    }

    return newNodes;
}

// ==========================================================
// Wypisanie aktualnej siatki
// ==========================================================
void printNodes(const std::vector<double>& nodes) {
    std::cout << "Siatka: ";
    for (double x : nodes) {
        std::cout << x << " ";
    }
    std::cout << "\n";
}

// ==========================================================
// MAIN
// ==========================================================
int main() {
    std::cout << std::fixed << std::setprecision(10);

    // ------------------------------------------------------
    // Siatka poczatkowa
    // ------------------------------------------------------

    // ------------------------------------------------------
    // PRZYKLAD: warstwa wewnetrzna (interior layer)
    //
    // Rozwiazanie analityczne:
    //   u(x) = tanh(k * (x - 0.5)),  k = 10
    //
    // Rownanie: -u'' + u = f(x)
    //   f(x) = tanh(k*(x-0.5)) * (1 + 2k^2 * sech^2(k*(x-0.5)))
    //
    // Steep gradient przy x = 0.5 — siatka powinna sie tam
    // dramatycznie zageszczac w kolejnych krokach adaptacji.
    // ------------------------------------------------------
    std::vector<double> nodes = {0.0, 0.25, 0.5, 0.75, 1.0};

    const double k  = 10.0;
    const double x0 = 0.5;

    // ------------------------------------------------------
    // Rownanie
    //
    // Solver MES obsluguje:
    // -u'' + p u' + q u = f
    //
    // Estymator bledu ponizej jest zgodny dla:
    // -u'' + q u = f
    //
    // Dlatego na tym etapie trzymaj p_fun = 0.
    // ------------------------------------------------------
    auto p_fun = [](double x) {
        return 0.0;
    };

    // q > 0 sprawia ze macierz A_K lokalnego problemu bledu
    // jest dodatnio okreslona (bez osobliwosci).
    auto q_fun = [](double x) {
        return 1.0;
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

    // ------------------------------------------------------
    // Warunki brzegowe
    //
    // Dirichlet z wartosciami analitycznymi:
    // u(0) = tanh(-k*x0),  u(1) = tanh(k*(1-x0))
    // ------------------------------------------------------
    BoundaryCondition left  = {DIRICHLET, std::tanh(-k * x0)};
    BoundaryCondition right = {DIRICHLET, std::tanh( k * (1.0 - x0))};

    // ------------------------------------------------------
    // Parametry adaptacji siatki
    //
    // TOL   - tolerancja wzgledna: eta <= TOL * ||u_h||_E
    // alpha - dzielimy elementy, dla ktorych eta_K > alpha * etaMax
    // ------------------------------------------------------
    double TOL = 0.01;
    double alpha = 0.5;
    int maxAdaptSteps = 20;

    // ------------------------------------------------------
    // Procedura adaptacji siatki
    // ------------------------------------------------------
    FEMResult lastResult;
    FEMVisualizer viz;

    for (int step = 0; step < maxAdaptSteps; step++) {
        std::cout << "==================================================\n";
        std::cout << "Krok adaptacji: " << step << "\n";
        printNodes(nodes);

        bool printSolutionDetails = true;
        bool printErrorDetails = false;

        lastResult = solveAndEstimate(
            nodes,
            p_fun,
            q_fun,
            f_fun,
            left,
            right,
            printSolutionDetails,
            printErrorDetails
        );

        double etaGlobal = std::sqrt(lastResult.etaGlobal2);
        double uhEnergyNorm = std::sqrt(lastResult.uhEnergyNorm2);

        std::cout << "Globalny estymator bledu:\n";
        std::cout << "  eta^2       = " << lastResult.etaGlobal2 << "\n";
        std::cout << "  eta         = " << etaGlobal << "\n";
        std::cout << "  ||u_h||_E^2 = " << lastResult.uhEnergyNorm2 << "\n";
        std::cout << "  ||u_h||_E   = " << uhEnergyNorm << "\n";
        std::cout << "  TOL*||u_h||_E = " << TOL * uhEnergyNorm << "\n";

        viz.plotStep(step, nodes, lastResult.d, lastResult.eta2List,
                     etaGlobal, uhEnergyNorm, TOL, alpha);

        // Kryterium stopu:
        // eta <= TOL * ||u_h||_E
        if (etaGlobal <= TOL * uhEnergyNorm) {
            std::cout << "\nSTOP: osiagnieto wymagana tolerancje bledu.\n";
            break;
        }

        std::vector<double> newNodes = refineMesh(
            nodes,
            lastResult.eta2List,
            alpha
        );

        if (newNodes.size() == nodes.size()) {
            std::cout << "\nSTOP: siatka nie zostala zmieniona.\n";
            break;
        }

        nodes = newNodes;

        std::cout << "\nSiatka zostala zageszczona.\n\n";

        if (step == maxAdaptSteps - 1) {
            std::cout << "\nSTOP: osiagnieto maksymalna liczbe krokow adaptacji.\n";
        }
    }

    std::cout << "\n==================================================\n";
    std::cout << "Siatka koncowa:\n";
    printNodes(nodes);

    return 0;
}
