#include <iostream>
#include <vector>
#include <functional>
#include "../Eigen/Dense"

// =======================
// Typy warunków brzegowych
// =======================
enum BCType { DIRICHLET, NEUMANN };

struct BoundaryCondition {
    BCType type;
    double value;
};

// =======================
// Prosta całka numeryczna
// =======================
double calka(double a, double b, std::function<double(double)> f, double dx = 1e-4) {
    double res = 0.0;
    for (double x = a; x < b; x += dx)
        res += f(x) * dx;
    return res;
}

// =======================
// Funkcje kształtu
// =======================
double N1(double x, double a, double b) {
    return (b - x) / (b - a);
}

double N2(double x, double a, double b) {
    return (x - a) / (b - a);
}

double t_K(double a, double duLeft, double duRight, double n){
    return 0.5*(a*duLeft+a*duRight) * n;
}


// funkcje bazowe dla funkcji bledu
double N1_B(double s) {
    return 2.0*(s-0.5)*(s-1);
}

double N2_B(double s) {
    return 4.0*s*(1-s);
}

double N3_B(double s) {
    return 2.0*s*(s-0.5);
}

// pochodne dla funkcji bazowych dla funkcji bledu
double dN1_B(double s) { return 4*s-3;}
double dN2_B(double s) { return 4-8*s;}
double dN3_B(double s) { return 4*s-1;}

double dN1B_N1B(double s) { return (4*s - 3)*(4*s - 3); }   // 16s² -24s +9
double dN1B_N2B(double s) { return (4*s - 3)*(4 - 8*s); }   // -32s² +40s -12
double dN1B_N3B(double s) { return (4*s - 3)*(4*s - 1); }   // 16s² -16s +3

double dN2B_N1B(double s) { return dN1B_N2B(s); }
double dN2B_N2B(double s) { return (4 - 8*s)*(4 - 8*s); }   // 64s² -64s +16
double dN2B_N3B(double s) { return (4 - 8*s)*(4*s - 1); }   // -32s² +24s -4

double dN3B_N1B(double s) { return dN1B_N3B(s); }
double dN3B_N2B(double s) { return dN2B_N3B(s); }
double dN3B_N3B(double s) { return (4*s - 1)*(4*s - 1); }   // 16s² -8s +1;

std::vector<std::vector<double>> A_K(
    double a, double b,
    std::function<double(double)> p = [](double){return 0;},
    std::function<double(double)> q = [](double){return 0;})
{
    double h = b - a;

    std::vector<std::vector<double>> A(3, std::vector<double>(3, 0.0));

    // (N_B)'^T * N_B, dla calki 0 do 1 uzywajc funkcji dN1B_N1B, dN1B_N2B ...
    A[0][0] += (7.0/3.0)  / h;
    A[0][1] += (-8.0/3.0) / h;
    A[0][2] += (1.0/3.0)  / h;

    A[1][0] += (-8.0/3.0) / h;
    A[1][1] += (16.0/3.0) / h;
    A[1][2] += (-8.0/3.0) / h;

    A[2][0] += (1.0/3.0)  / h;
    A[2][1] += (-8.0/3.0) / h;
    A[2][2] += (7.0/3.0)  / h;

    // funkcje lokalne
    auto Ni = [&](int i, double x) {
        switch (i) {
            case 0:
                return N1_B(x);
                break;
            case 1:
                return N2_B(x);
                break;
            case 2:
                return N3_B(x);
                break;
            default:
                std::cout << "niepoprawny indeks funkcji lokalnej\n";
        }
    };

    // część od q(x) * u
    for (int i = 0; i < 3; i++) {
        for  (int j = 0; j < 3; j++) {
            auto integrand = [&](double x) {
                return q(x) * Ni(i, x) * Ni(j, x);
            };
            A[i][j] += h * calka(0.0, 1.0, integrand); // h bierze sie z Jacobianu, bo integral(q(x)N_i*N_j dx = h integral_0_1(q(a + hs) N_i*N_j ds))
            // x = a + hs, h = b - a
            // dx = h*ds
        }
    }

    return A;
}

// =======================
// Macierz elementowa
// =======================
std::vector<std::vector<double>> K_e(
    double a, double b,
    std::function<double(double)> p = [](double){return 0;},
    std::function<double(double)> q = [](double){return 0;})
{
    double h = b - a;

    std::vector<std::vector<double>> K(2, std::vector<double>(2, 0.0));

    // część od u''
    K[0][0] += 1.0 / h;
    K[0][1] += -1.0 / h;
    K[1][0] += -1.0 / h;
    K[1][1] += 1.0 / h;

    // funkcje lokalne
    auto Ni = [&](int i, double x) {
        return (i == 0) ? N1(x, a, b) : N2(x, a, b);
    };

    auto dNj = [&](int j) {
        return (j == 0) ? -1.0 / h : 1.0 / h;
    };

    // część od p(x) * u'
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            auto integrand = [&](double x) {
                return p(x) * Ni(i, x) * dNj(j);
            };
            K[i][j] += calka(a, b, integrand);
        }
    }

    // część od q(x) * u
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

// =======================
// Wektor prawej strony
// =======================
std::vector<double> p_e(
    double a, double b,
    std::function<double(double)> f)
{
    std::vector<double> p(2, 0.0);

    auto f1 = [&](double x) { return f(x) * N1(x, a, b); };
    auto f2 = [&](double x) { return f(x) * N2(x, a, b); };

    p[0] = calka(a, b, f1);
    p[1] = calka(a, b, f2);

    return p;
}

// =======================
// Assembly
// =======================
void assemble(
    const std::vector<double>& nodes,
    std::function<double(double)> p_fun,
    std::function<double(double)> q_fun,
    std::function<double(double)> f_fun,
    Eigen::MatrixXd& K,
    Eigen::VectorXd& P)
{
    int n = nodes.size();
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

// =======================
// Warunki brzegowe
// =======================
void applyBC(
    Eigen::MatrixXd& K,
    Eigen::VectorXd& P,
    BoundaryCondition left,
    BoundaryCondition right)
{
    int n = K.rows();

    // LEFT
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

    // RIGHT
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

// =======================
// MAIN
// =======================
int main() {

    // siatka
    std::vector<double> nodes = {0, 0.25, 0.5, 0.75, 0.85, 1};

    // równanie
    auto p_fun = [](double x) { return 0.0; };
    auto q_fun = [](double x) { return 0.0; };
    auto f_fun = [](double x) { return 6.0 * x * x; };

    // warunki brzegowe
    BoundaryCondition left  = {DIRICHLET, 1.0};
    BoundaryCondition right = {NEUMANN, -0.5};

    Eigen::MatrixXd K;
    Eigen::VectorXd P;

    assemble(nodes, p_fun, q_fun, f_fun, K, P);

    applyBC(K, P, left, right);

    // rozwiązanie
    Eigen::VectorXd d = K.colPivHouseholderQr().solve(P);

    std::cout << "Rozwiazanie:\n";
    for (int i = 0; i < d.size(); i++) {
        std::cout << "u" << i << " = " << d(i) << "\n";
    }

    // szukanie błędu
    std::vector<std::vector<double>> t(0, std::vector<double>(0));

    // u(0) = 1
    // u(L) = -0.5
    // alfa = 1, beta = 1, gamma_0 = 1, gamma_L = -0.5 a(l), a(0) = 1
    for (int i=0; i < nodes.size()-1; i++) {
        // B_K (u_h, psi_K) - L_K(psi_K) + t_K(-1/+1)
        double a = nodes[i], b = nodes[i+1];
        double h = b - a;
        double u_approx = d[i];

        // N - funkcja testowa
        // integral(N' * N*d) + N(l)*d*N(l) - N(0)*d*N(0)

        double dN1 = -1;
        double dN2 = 1;

        // czesc t_K
        double du_left = d[i]*dN1 + d[i+1]*dN2;
        double du_right = 0;
        if (i == (int)nodes.size() - 2) {
            double du_right = d[i+1]*dN1 + d[i+1]*dN2;
        } else {
            double du_right = d[i+1]*dN1 + d[i+2]*dN2;
        }

        double t_left = t_K(1, du_left, du_right, -1);
        double t_right = t_K(1, du_left, du_right, 1);

        auto d_u_d_v_1 = [a,b,u_approx, dN1](double x) {
            return N1(x,a,b)*u_approx*dN1;
        };

        auto d_u_d_v_2 = [a,b, u_approx,dN2](double x) {
            return N2(x,a,b)*u_approx*dN2;
        };

        double B_u_v_1 = calka(a,b,d_u_d_v_1) + N1(b,a,b)*u_approx*N1(b,a,b) - N1(a,a,b)*u_approx*N1(a,a,b);
        double B_u_v_2 = calka(a,b,d_u_d_v_2) + N2(b,a,b)*u_approx*N2(b,a,b) - N2(a,a,b)*u_approx*N2(a,a,b);

        double L_v_1 = calka(a,b,f_fun) + -0.5*N1(b,a,b) - 1*N1(a,a,b);
        double L_v_2 = calka(a,b,f_fun) + -0.5*N2(b,a,b) - 1*N2(a,a,b);

        double theta_1 = B_u_v_1 - L_v_1 + t_left;
        double theta_2 = B_u_v_2 - L_v_2 + t_right;

        std::vector<double> theta {theta_1, theta_2};

        t.push_back(theta);
    }
    

    std::cout << "ok0\n";

    // szukanie funkcji błędu
    // B(sigma_K, v) = r_K(v), rozszerzona przestrzen (p + 1) -> p = 2, trzy funkcje

    // A_K * b_K = r_K
    std::vector<Eigen::VectorXd> b_K;
    for (int k=0; k < nodes.size()-1; k++)
    {
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3, 3);

        size_t ILOSC_FUNKCJI_TESTOWYCH = 3;

        // to matrixxd
        auto A_std = A_K(0.0,1.0, [](double){return 0.0;}, [](double){return 0.0;});
        for (int i=0; i < 3; i++){
            for (int j=0; j < 3; j++) {
                A(i,j) += A_std[i][j];
            }
        }

        // r_K
        Eigen::Vector3d r_K;
        for (int p=0; p < ILOSC_FUNKCJI_TESTOWYCH; p++) {
            
            double a = nodes[k];
            double b = nodes[k+1];
            double h = nodes[k+1] - nodes[k];
            double duh = (d[k+1]-d[k])/h;

            std::vector<std::function<double(double)>> N_i_B {N1_B, N2_B, N3_B};
            std::vector<std::function<double(double)>> dN_i_B {dN1_B, dN2_B, dN3_B};


            auto B_K_1 = [&](double s){
                double a = nodes[k];
                double x = a + h*s;

                // 1 * d/dx = h * d/ds
                // dx = (1/h) * ds

                return duh*dN_i_B[p](x)*1/h;
            };

            auto B_K_2 = [&](double s){
                double a = nodes[k];
                double x = a+h*s;

                return (d[k]*N1(x,a,b) + d[k+1]*N2(x,a,b))*N_i_B[p](x);
            };

            double term1 = h * calka(0,1,B_K_1);
            double term2 = h * calka(0,1,B_K_2);

            Eigen::Vector3d B_K;
            B_K(p) = term1 + term2;

            Eigen::Vector3d L_K;

            auto L_int_1 = [&](double x){
                return f_fun(x) * N1_B(x);
            };

            auto L_int_2 = [&](double x){
                return f_fun(x) * N2_B(x);
            };

            auto L_int_3 = [&](double x){
                return f_fun(x) * N3_B(x);
            };

            double L_v_1 = calka(0.0,1.0,L_int_1);
            double L_v_2 = calka(0.0,1.0,L_int_2);
            double L_v_3 = calka(0.0,1.0,L_int_3);

            L_K(0) += L_v_1;
            L_K(1) += L_v_2;
            L_K(2) += L_v_3;

            double dN1 = -1;
            double dN2 = 1;

            double du_left = d(k)*dN1 + d(k+1)*dN2;
            double du_right = 0;
            if (k == (int)nodes.size() - 2) {
                du_right = d[k+1]*dN1 + d[k+1]*dN2;
            } else {
                du_right = d[k+1]*dN1 + d[k+2]*dN2;
            }

            double delta_1 = t[k][0]*N1_B(0) + t[k][1]*N1_B(1) +  t_K(1, du_left, du_right, -1) + t_K(1, du_left, du_right, 1);
            double delta_2 = t[k][0]*N2_B(0) + t[k][1]*N2_B(1) +  t_K(1, du_left, du_right, -1) + t_K(1, du_left, du_right, 1);
            double delta_3 = t[k][0]*N3_B(0) + t[k][1]*N3_B(1) +  t_K(1, du_left, du_right, -1) + t_K(1, du_left, du_right, 1);

            B_K(0) += delta_1;
            B_K(1) += delta_2;
            B_K(2) += delta_3;

            r_K(0) = B_K(0) - L_K(0) - delta_1;
            r_K(1) = B_K(1) - L_K(1) - delta_2;
            r_K(2) = B_K(2) - L_K(2) - delta_1;
        }

        b_K.push_back(A.colPivHouseholderQr().solve(r_K));
    }

    std::cout << "ok\n";

    // obliczenie bledy i wyswietlanie go
    for (int i=0; i < b_K.size(); i++) {
        std::cout << "b_k_" << i << "wspolczynnik: \n";
        for (int j=0; j < b_K[i].size(); j++){
            std::cout << b_K[i][j] << "\n";
        }

        auto A_std = A_K(0.0,1.0, [](double){return 0.0;}, [](double){return 0.0;});
        Eigen::Matrix3d A;
        for (int k=0; k < 3; k++){
            for (int l=0; l < 3; l++){
                A(k,l) += A_std[k][l];
            }
        }

        double eta2 = b_K[i].transpose()*A*b_K[i];
        std::cout << "eta2: " << eta2 << "\n";
    }

    return 0;
}