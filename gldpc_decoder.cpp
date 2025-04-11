#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <tuple>
#include <string>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/math/special_functions/expm1.hpp>

#define HARD(A) (((A) < 0) ? 1 : 0)

namespace mp = boost::multiprecision;
using big_float = mp::number<mp::cpp_dec_float<100>>;
using namespace std;
namespace py = pybind11;

void printVec(vector<double>& vec) {
    for (int i = 0; i < vec.size(); i++) {
        cout << " " << vec[i];
    }
    cout << "\n";
}

void printVecInt(vector<int> vec) {
    for (int i = 0; i < vec.size(); i++) {
        cout << " " << vec[i];
    }
    cout << "\n";
}

double safe_exp(double x)
{
    if (x > 700)
    {
        return 1.014232054735005e+304;
    }
    if (x < -700)
    {
        return 9.859676543759770e-305;
    }
    return exp(x);
}

double safe_log(double x)
{
    if (x < 9.859676543759770e-305)
    {
        return -700;
    }
    return log(x);
}

bool is_zero_mod2(vector<int> matrix) {
    for (int element : matrix) {
        if (element % 2 != 0) {
            return false;
        }
    }
    
    return true;
}
vector<int> multiply_vector_matrix(vector<int> v, vector<vector<int>> m) {
    int n = v.size();
    int p = m.size();
    int q = m[0].size();
    vector<int> result(q);
    for (int j = 0; j < q; j++) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += v[k] * m[k][j];
        }
        result[j] = sum;
    }
    return result;
}
vector<vector<int>> transpose(const vector<vector<int>> matrix) {
    const size_t rows = matrix.size();
    const size_t cols = matrix[0].size();
    vector<vector<int>> result(cols, vector<int>(rows));
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

vector<double> decode_bcjr_precise(
    const vector<vector<tuple<string, int, string>>> &edg_bpsk,
    const vector<double> &llr_in,
    double sigma2,
    bool useNormalization
) {
    double a_priori = 0.5;
    double constant_coef = a_priori * (1 / (2 * M_PI * sigma2));

    vector<vector<tuple<string, big_float, string>>> gammas;

    for (const auto& layer : edg_bpsk) {
        vector<tuple<string, big_float, string>> new_layer;
        for (const auto& row : layer) {
            string s1 = get<0>(row);
            int int_value = get<1>(row);
            string s2 = get<2>(row);

            big_float double_value = static_cast<big_float>(int_value);
            new_layer.emplace_back(s1, double_value, s2);
        }
        gammas.push_back(new_layer);
    }

    vector<unordered_map<string, big_float>> alphas(gammas.size() + 1);
    tuple<string, big_float, string> first_edge = gammas[0][0];
    alphas[0][get<0>(first_edge)] = 1.0;

    for (size_t i = 0; i < gammas.size(); i++) {
        for (size_t j = 0; j < gammas[i].size(); j++) {
            string prev_vex = get<0>(gammas[i][j]);
            int edge_value = get<1>(edg_bpsk[i][j]);
            string next_vex = get<2>(gammas[i][j]);

            big_float diff = pow(llr_in[i] - edge_value, 2) / (2 * sigma2);
            big_float cur_gamma = constant_coef * mp::exp2(-diff / log(2)); // original exp(-diff)

            if (cur_gamma == 0) {
                cur_gamma = 2.22507e-308;
            }
            gammas[i][j] = make_tuple(prev_vex, cur_gamma, next_vex);
            big_float new_alpha = cur_gamma * alphas[i][prev_vex];
            alphas[i + 1][next_vex] += new_alpha;
        }

        if (useNormalization) {
            big_float sum_alpha = 0;
            for (const auto& p : alphas[i + 1]) sum_alpha += p.second;
            if (sum_alpha != 0) {
                for (auto& p : alphas[i + 1]) p.second /= sum_alpha;
            }
        }
    }

    vector<unordered_map<string, big_float>> betas(gammas.size() + 1);
    betas[gammas.size()][get<0>(gammas[0][0])] = 1;
    vector<double> llr_out(llr_in.size(), 0);

    for (int i = gammas.size() - 1; i >= 0; i--) {
        big_float up = 0, down = 0;
        for (size_t j = 0; j < gammas[i].size(); j++) {
            string next_vex = get<0>(gammas[i][j]);
            big_float cur_gamma = get<1>(gammas[i][j]);
            string prev_vex = get<2>(gammas[i][j]);

            big_float new_beta = cur_gamma * betas[i + 1][prev_vex];
            betas[i][next_vex] += new_beta;

            big_float cur_alpha = alphas[i][next_vex];
            big_float cur_beta = betas[i + 1][prev_vex];
            big_float cur_sigma = cur_gamma * cur_alpha * cur_beta;

            if (get<1>(edg_bpsk[i][j]) == 1) {
                up += cur_sigma;
            } else {
                down += cur_sigma;
            }
        }

        llr_out[i] = static_cast<double>(mp::log(up / down));

        if (useNormalization) {
            big_float sum_beta = 0;
            for (const auto& p : betas[i]) sum_beta += p.second;
            if (sum_beta != 0) {
                for (auto& p : betas[i]) p.second /= sum_beta;
            }
        }
    }

    return llr_out;
}

vector<double> decode_bcjr(
    const vector<vector<tuple<string, int, string>>> &edg_bpsk,
    const vector<double> &llr_in,
    double sigma2,
    bool useNormalization
) {
    double a_priori = 0.5;
    double constant_coef = a_priori * (1 / (2 * M_PI * sigma2));

    vector<vector<tuple<string, long double, string>>> gammas;

    for (const auto& layer : edg_bpsk) {
        vector<tuple<string, long double, string>> new_layer;
        for (const auto& row : layer) {
            string s1 = get<0>(row);
            int int_value = get<1>(row);
            string s2 = get<2>(row);

            long double double_value = static_cast<long double>(int_value);
            new_layer.emplace_back(s1, double_value, s2);
        }
        gammas.push_back(new_layer);
    }

    vector<unordered_map<string, long double>> alphas(gammas.size() + 1);
    tuple<string, long double, string> first_edge = gammas[0][0];
    alphas[0][get<0>(first_edge)] = 1.0;

    for (size_t i = 0; i < gammas.size(); i++) {
        for (size_t j = 0; j < gammas[i].size(); j++) {
            string prev_vex = get<0>(gammas[i][j]);
            int edge_value = get<1>(edg_bpsk[i][j]);
            string next_vex = get<2>(gammas[i][j]);

            long double diff = pow(llr_in[i] - edge_value, 2) / (2 * sigma2);
            long double cur_gamma = constant_coef * safe_exp(-diff);

            // if (cur_gamma == 0) {
            //     return decode_bcjr_precise(edg_bpsk, llr_in, sigma2, useNormalization);
            // }

            gammas[i][j] = make_tuple(prev_vex, cur_gamma, next_vex);
            long double new_alpha = cur_gamma * alphas[i][prev_vex];
            alphas[i + 1][next_vex] += new_alpha;
        }

        if (useNormalization) {
            long double sum_alpha = 0;
            for (const auto& p : alphas[i + 1]) sum_alpha += p.second;
            if (sum_alpha != 0) {
                for (auto& p : alphas[i + 1]) p.second /= sum_alpha;
            }
        }
    }

    vector<unordered_map<string, long double>> betas(gammas.size() + 1);
    betas[gammas.size()][get<0>(gammas[0][0])] = 1;
    vector<double> llr_out(llr_in.size(), 0);

    for (int i = gammas.size() - 1; i >= 0; i--) {
        long double up = 0, down = 0;
        for (size_t j = 0; j < gammas[i].size(); j++) {
            string next_vex = get<0>(gammas[i][j]);
            long double cur_gamma = get<1>(gammas[i][j]);
            string prev_vex = get<2>(gammas[i][j]);

            long double new_beta = cur_gamma * betas[i + 1][prev_vex];
            betas[i][next_vex] += new_beta;

            long double cur_alpha = alphas[i][next_vex];
            long double cur_beta = betas[i + 1][prev_vex];
            long double cur_sigma = cur_gamma * cur_alpha * cur_beta;

            if (get<1>(edg_bpsk[i][j]) == 1) {
                up += cur_sigma;
            } else {
                down += cur_sigma;
            }
        }

        llr_out[i] = static_cast<double>(safe_log(up / down));


        if (useNormalization) {
            long double sum_beta = 0;
            for (const auto& p : betas[i]) sum_beta += p.second;
            if (sum_beta != 0) {
                for (auto& p : betas[i]) p.second /= sum_beta;
            }
        }
    }

    return llr_out;
}

vector<int> decode_gldpc(
    const vector<vector<int>>& H_GLDPC,
    const vector<vector<int>>& H_LDPC,
    const vector<vector<int>>& sorted_original_indexes,
    const vector<vector<tuple<string, int, string>>> &edg_bpsk,
    const vector<double>& llr,
    double sigma2,
    int maxIter,
    bool useNormalization
) {
    int m_ldpc = H_LDPC.size();
    int n_ldpc = H_LDPC[0].size();
    vector<int> x_hat(n_ldpc, 0);
    vector<vector<double>> H_gamma(m_ldpc, vector<double>(n_ldpc, 0.0));
    vector<vector<double>> H_q(m_ldpc, vector<double>(n_ldpc, 0.0));
    vector<double> llr_out(n_ldpc, 0.0);
    vector<vector<int>> H_T = transpose(H_GLDPC);
    // Инициализация H_q
    for (int i = 0; i < m_ldpc; i++) {
        for (int j = 0; j < n_ldpc; j++) {
            H_q[i][j] = H_LDPC[i][j] * llr[j];
        }
    }

    for (int iter = 0; iter < maxIter; iter++) {
        // Layer decoding
        for (int i = 0; i < m_ldpc; i++) {
            const vector<int>& sorted_indexes = sorted_original_indexes[i];
            vector<double> llr_in_layer_decoder;

            for (int idx : sorted_indexes) {
                llr_in_layer_decoder.push_back(H_q[i][idx]);
            }

            vector<double> llr_from_layer_decoder = decode_bcjr(
                edg_bpsk,
                llr_in_layer_decoder,
                sigma2,
                useNormalization
            );

            for (size_t k = 0; k < sorted_indexes.size(); k++) {
                int j = sorted_indexes[k];
                H_gamma[i][j] = llr_from_layer_decoder[k] - llr_in_layer_decoder[k];
            }
        }

        // Update H_q
        for (int i = 0; i < m_ldpc; i++) {
            for (int j = 0; j < n_ldpc; j++) {
                if (H_LDPC[i][j] == 1) {
                    double sum = 0.0;
                    for (int k = 0; k < m_ldpc; k++) {
                        if (k != i && H_LDPC[k][j] == 1) {
                            sum += H_gamma[k][j];
                        }
                    }
                    H_q[i][j] = llr[j] + sum;
                }
            }
        }

        // Update LLR_out
        for (int j = 0; j < n_ldpc; j++) {
            double sum = 0.0;
            for (int i = 0; i < m_ldpc; i++) {
                sum += H_gamma[i][j];
            }
            llr_out[j] = llr[j] + sum;
        }

        // Hard decision
        for (int j = 0; j < n_ldpc; j++) {
            x_hat[j] = HARD(llr_out[j]);
        }

        // Syndrome check
        cout << "СИНДРОМ:" << std::endl;
        printVecInt(multiply_vector_matrix(x_hat, H_T));
        if (is_zero_mod2(multiply_vector_matrix(x_hat, H_T))) {
            cout << "ХОРОШО" << std::endl;
            printVec(llr_out);
            return x_hat;
        }
    }
    cout << "ПЛОХО" << std::endl;
    printVec(llr_out);
    return x_hat;
}

// Оборачиваем функцию для Pybind11
PYBIND11_MODULE(gldpc_decoder, m) {
    m.def("decode_gldpc", &decode_gldpc, "gldpc decoder function");
    m.def("decode_bcjr", &decode_bcjr, "bcjr decoder function");
    m.def("decode_bcjr_precise", &decode_bcjr_precise, "bcjr decoder precise function");
}