// rozwiazujemy rownanie
// u''(x) + 6x^2 = 0
// 
// zaczynamy od funkcji ksztaltu
// N_i = (x[i+1] - x) / h, N_j = (x - x[i]) / h
// h = [x+1] - x[i]
// 
// f(x) = 6x^2
//
// Skladanie (assembly)

#include <iostream>
#include <vector>
#include <functional>
#include "Eigen/Dense"

double pochodna(double x, std::function<double(double)> func, double h) {
    return (func(x+h) - func(x)) / h;
}

double calka(double start, double end, std::function<double(double)> func, double delta = 0.001) {
    double res = 0;
    for (double a = start; a <= end; a += delta) {
        res += func(a) * delta;
    }

    return res;
}

double N_1(double x,  double x_cur, double x_next) {
    return (x_next - x) / (x_next-x_cur);
}

double N_2(double x, double x_cur, double x_next) {
    return (x - x_cur) / (x_next-x_cur);
}

std::vector<std::vector<double> > K_e(double start, double end) {
    std::vector<double (*)(double,double,double)> funkcje_kstaltu {N_1, N_2};
    std::vector<std::vector<double>> res(2, std::vector<double>(2));

    for (int i=0; i<funkcje_kstaltu.size(); i++)
    {
        for (int j=0; j<funkcje_kstaltu.size(); j++) {
            auto N_i = [start, end, i, funkcje_kstaltu](double x) {
                return funkcje_kstaltu[i](x, start, end);
            };

            auto N_j = [start, end, j, funkcje_kstaltu](double x) {
                return funkcje_kstaltu[j](x, start, end);
            };

            auto N_i_N_j = [N_i,N_j](double x) {
                double precyzja = 0.000001;
                return pochodna(x, N_i, precyzja)*pochodna(x, N_j, precyzja);
            };

            res[i][j] = calka(start, end, N_i_N_j);
        }
    }

    return res;
}

std::vector<double> p_e(double start, double end, std::function<double(double)> f)  {
    std::vector<double> res(2,0);
    auto p_1 = [f, start, end](double x){
        return f(x) * N_1(x, start, end);
    };

    auto p_2 = [f, start, end](double x){
        return f(x) * N_2(x, start, end);
    };

    res[0] = calka(start,end, p_1, 0.000001);
    res[1] = calka(start,end,p_2, 0.000001);

    return res;
}

std::vector<std::vector<double>>  skladanie_K(std::vector<std::vector<std::vector<double>>> K) {
    std::vector<std::vector<double>> res (K.size()+1, std::vector<double>(K.size()+1));

    // zerowanie macierzy
    for (int i=0; i < res.size(); i++) {
        for (int j=0; j < res[0].size(); j++) {
            res[i][j] = 0;
        }
    }

    // skladanie
    for (int k=0; k < K.size(); k++){
        for (int i=0; i < K[k].size(); i++) {
            for (int j=0; j < K[k][0].size(); j++) {
                res[i+k][j+k] += K[k][i][j];
            }
        }
    }

    return res;
}

std::vector<double> skladanie_p(std::vector<std::vector<double>> p) {
    std::vector<double> res (p.size()+1);

    // zerowanie wektora
    for (int i=0; i < res.size(); i++) {
        res[i] = 0;
    }

    for (int e=0; e < p.size(); e++){
        for (int i=0; i < p[e].size(); i++){
            res[i+e] += p[e][i];
        }
    }
    
    return res;
}

int main() {
    auto K_1 = K_e(0, 0.5);
    auto K_2 = K_e(0.5,1);

    // skladanie K
    auto wszystkie_K = std::vector<std::vector<std::vector<double>>> {K_1, K_2};
    auto K = skladanie_K(wszystkie_K);

    std::cout << "K zlozone:\n";
    for (auto vec : K) {
        for (auto num : vec) {
            std::cout << num << ' ';
        }
        std::cout << '\n';
    }

    auto f = [](double x) {
        return 6.0 * x * x;
    };
    auto p_1 = p_e(0, 0.5, f);
    auto p_2 = p_e(0.5,1,f);

    std::vector<double> p_1b = {0.0, 0.0};
    std::vector<double> p_2b = {0.0, -0.5};

    for (int i = 0; i < 2; i++) {
        p_1[i] += p_1b[i];
        p_2[i] += p_2b[i];
    }

    // skladanie p
    auto wszystkie_p = std::vector<std::vector<double>> {p_1,p_2};
    auto p = skladanie_p(wszystkie_p);

    std::cout << "p zlozone:\n";
    for (auto num : p) {
            std::cout << num << ' ';
        }
    std::cout << '\n';

    // konwersja do Eigen
    int n = K.size();

    Eigen::MatrixXd K_eigen(n, n);
    Eigen::VectorXd p_eigen(n);

    // przepisanie danych
    for (int i = 0; i < n; i++) {
        p_eigen(i) = p[i];
        for (int j = 0; j < n; j++) {
            K_eigen(i, j) = K[i][j];
        }
    }

    for (int i = 0; i < n; i++) {
        p_eigen(i) -= K_eigen(i, 0) * 1.0;
    }

    // teraz narzucenie warunku
    K_eigen.row(0).setZero();
    K_eigen.col(0).setZero();
    K_eigen(0,0) = 1.0;

    p_eigen(0) = 1.0;

    // rozwiązanie układu K * d = p
    Eigen::VectorXd d = K_eigen.partialPivLu().solve(p_eigen);

    // wypisanie wyniku
    std::cout << "Rozwiazanie d:\n";
    for (int i = 0; i < d.size(); i++) {
        std::cout << d(i) << " ";
    }
    std::cout << std::endl;

    return 0;
}