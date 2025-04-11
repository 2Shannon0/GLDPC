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

using namespace std;
namespace py = pybind11;
namespace mp = boost::multiprecision;

vector<double> decode_bcjr_precise(
    const vector<vector<tuple<string, int, string>>> &edg_bpsk,
    const vector<double> &llr_in,
    double sigma2,
    bool useNormalization
) {
    double a_priori = 0.5;
    double constant_coef = a_priori * (1 / (2 * M_PI * sigma2));

    vector<vector<tuple<string, mp::cpp_dec_float_100, string>>> gammas;

    for (const auto& layer : edg_bpsk) {
        vector<tuple<string, mp::cpp_dec_float_100, string>> new_layer;
        for (const auto& row : layer) {
            string s1 = get<0>(row);
            int int_value = get<1>(row);
            string s2 = get<2>(row);

            mp::cpp_dec_float_100 double_value = static_cast<mp::cpp_dec_float_100>(int_value);
            new_layer.emplace_back(s1, double_value, s2);
        }
        gammas.push_back(new_layer);
    }

    vector<unordered_map<string, mp::cpp_dec_float_100>> alphas(gammas.size() + 1);
    tuple<string, mp::cpp_dec_float_100, string> first_edge = gammas[0][0];
    alphas[0][get<0>(first_edge)] = 1.0;

    for (size_t i = 0; i < gammas.size(); i++) {
        for (size_t j = 0; j < gammas[i].size(); j++) {
            string prev_vex = get<0>(gammas[i][j]);
            int edge_value = get<1>(edg_bpsk[i][j]);
            string next_vex = get<2>(gammas[i][j]);

            mp::cpp_dec_float_100 diff = pow(llr_in[i] - edge_value, 2) / (2 * sigma2);
            mp::cpp_dec_float_100 cur_gamma = constant_coef * mp::exp2(-diff / log(2)); // original exp(-diff)

            gammas[i][j] = make_tuple(prev_vex, cur_gamma, next_vex);
            mp::cpp_dec_float_100 new_alpha = cur_gamma * alphas[i][prev_vex];
            alphas[i + 1][next_vex] += new_alpha;
        }

        if (useNormalization) {
            mp::cpp_dec_float_100 sum_alpha = 0;
            for (const auto& p : alphas[i + 1]) sum_alpha += p.second;
            if (sum_alpha != 0) {
                for (auto& p : alphas[i + 1]) p.second /= sum_alpha;
            }
        }
    }

    vector<unordered_map<string, mp::cpp_dec_float_100>> betas(gammas.size() + 1);
    betas[gammas.size()][get<0>(gammas[0][0])] = 1;
    vector<double> llr_out(llr_in.size(), 0);

    for (int i = gammas.size() - 1; i >= 0; i--) {
        mp::cpp_dec_float_100 up = 0, down = 0;
        for (size_t j = 0; j < gammas[i].size(); j++) {
            string next_vex = get<0>(gammas[i][j]);
            mp::cpp_dec_float_100 cur_gamma = get<1>(gammas[i][j]);
            string prev_vex = get<2>(gammas[i][j]);

            mp::cpp_dec_float_100 new_beta = cur_gamma * betas[i + 1][prev_vex];
            betas[i][next_vex] += new_beta;

            mp::cpp_dec_float_100 cur_alpha = alphas[i][next_vex];
            mp::cpp_dec_float_100 cur_beta = betas[i + 1][prev_vex];
            mp::cpp_dec_float_100 cur_sigma = cur_gamma * cur_alpha * cur_beta;

            if (get<1>(edg_bpsk[i][j]) == 1) {
                up += cur_sigma;
            } else {
                down += cur_sigma;
            }
        }

        llr_out[i] = static_cast<double>(mp::log(up / down));

        if (useNormalization) {
            mp::cpp_dec_float_100 sum_beta = 0;
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
            long double cur_gamma = constant_coef * exp(-diff);

            if (cur_gamma == 0) {
                return decode_bcjr_precise(edg_bpsk, llr_in, sigma2, useNormalization);
            }

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

        llr_out[i] = static_cast<double>(log(up / down));


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
        bool valid = true;
        for (int i = 0; i < m_ldpc; i++) {
            int syndrome = 0;
            for (int j = 0; j < n_ldpc; j++) {
                syndrome += H_LDPC[i][j] * x_hat[j];
            }
            if (syndrome % 2 != 0) {
                valid = false;
                break;
            }
        }

        if (valid) {
            return x_hat;
        }
    }

    return x_hat;
}

// Оборачиваем функцию для Pybind11
PYBIND11_MODULE(gldpc_decoder, m) {
    m.def("decode_gldpc", &decode_gldpc, "gldpc decoder function");
    m.def("decode_bcjr", &decode_bcjr, "bcjr decoder function");
    m.def("decode_bcjr_precise", &decode_bcjr_precise, "bcjr decoder precise function");
}
