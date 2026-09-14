/* ----------------------------------------------------------------------------
 * Copyright (c) 2021, University of Leeds and Harbin Institute of Technology.
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   butterworth_filter.cpp
 *  @author Jun Li (junlileeds@gmail.com)
 *  @brief  Source file for ButterworthFilter class
 *  @date   June 16, 2022
 **/

#include "butterworth_filter.hpp"

namespace tssfa_se {

ButterworthFilter::ButterworthFilter(double dt, double cutoff, int N)
    : dt_(dt), cutoff_(cutoff), N_(N)
{
    for(int i = 0; i < MAX_FILTER_LENGTH; i++)
    {
        a_[i] = 0;
        b_[i] = 0;
    }

    init_ = false;
}

void ButterworthFilter::clearFilter()
{
    for(int i = 0; i < MAX_FILTER_LENGTH; i++)
    {
        a_[i] = 0;
        b_[i] = 0;
    }
}

void ButterworthFilter::butterworth(double dt, double cutoff, int N)
{
    // generate the filter coefficients
    this->clearFilter();
    double A;
    if (N > 4) N = 4;
    if (N == 0) N = 1;
    double C = 1 / tan(M_PI * cutoff * dt);
    switch (N) {
        case 1:
            A = 1 / (1 + C);
            a_[0] = A;
            a_[1] = A;
            b_[0] = 1;
            b_[1] = (1 - C) * A;
            break;
        case 2:
            A = 1 / (1 + 1.4142 * C + pow(C, 2));
            a_[0] = A;
            a_[1] = 2 * A;
            a_[2] = A;
            b_[0] = 1;
            b_[1] = (2 - 2 * pow(C, 2)) * A;
            b_[2] = (1 - 1.4142 * C + pow(C, 2)) * A;
            break;
        case 3:
            A = 1 / (1 + 2 * C + 2 * pow(C, 2) + pow(C, 3));
            a_[0] = A;
            a_[1] = 3 * A;
            a_[2] = 3 * A;
            a_[3] = A;
            b_[0] = 1;
            b_[1] = (3 + 2 * C - 2 * pow(C, 2) - 3 * pow(C, 3)) * A;
            b_[2] = (3 - 2 * C - 2 * pow(C, 2) + 3 * pow(C, 3)) * A;
            b_[3] = (1 - 2 * C + 2 * pow(C, 2) - pow(C, 3)) * A;
            break;
        case 4:
            A = 1 / (1 + 2.6131 * C + 3.4142 * pow(C, 2) + 2.6131 * pow(C, 3) + pow(C, 4));
            a_[0] = A;
            a_[1] = 4 * A;
            a_[2] = 6 * A;
            a_[3] = 4 * A;
            a_[4] = A;
            b_[0] = 1;
            b_[1] = (4 + 2 * 2.6131 * C - 2 * 2.6131 * pow(C, 3) - 4 * pow(C, 4)) * A;
            b_[2] = (6 * pow(C, 4) - 2 * 3.4142 * pow(C, 2) + 6) * A;
            b_[3] = (4 - 2 * 2.6131 * C + 2 * 2.6131 * pow(C, 3) - 4 * pow(C, 4)) * A;
            b_[4] = (1 - 2.6131 * C + 3.4142 * pow(C, 2) - 2.6131 * pow(C, 3) + pow(C, 4)) * A;
            break;
        default:
            break;
    }
}

std::vector<double> ButterworthFilter::applyFilter(const std::vector<double>& X)
{
    //assumes the filter was already run so that the coefficients are able
    //receive the new input- return the filter signal
    if (!init_) {
        Xvec_.clear();
        Yvec_.clear();
        Xvec_.resize(5, X);
        Yvec_.resize(5, X);
        init_ = true;
    }

    for (int i = 0; i < 4; i++) 
    {
        Xvec_[i] = Xvec_[i+1];
        Yvec_[i] = Yvec_[i+1];
    }
    Xvec_[4] = X;
    for (std::size_t i = 0; i < X.size(); i++) {
        Yvec_[4][i] = a_[0] * Xvec_[4][i] + a_[1] * Xvec_[3][i] + a_[2] * Xvec_[2][i] + 
                      a_[3] * Xvec_[1][i] + a_[4] * Xvec_[0][i] - b_[1] * Yvec_[3][i] - 
                      b_[2] * Yvec_[2][i] - b_[3] * Yvec_[1][i] - b_[4] * Yvec_[0][i];
    }

    return Yvec_[4];
}

ButterworthFilter::Vector ButterworthFilter::applyButterworth(ConstRefVector raw_data)
{
    if (!init_) {
        if (dt_ <= 0 || dt_ > 0.1) {
            std::cout << "Invalid Filter Sampling Time!" << std::endl;
            exit(0);
        }
        this->butterworth(dt_, cutoff_, N_);
        Xvec_.clear();
        Yvec_.clear();
        std::vector<double> temp_data(
            raw_data.data(), raw_data.data()+raw_data.size());
        Xvec_.resize(5, temp_data);
        Yvec_.resize(5, temp_data);
        init_ = true;
    }
    std::vector<double> temp_data(
        raw_data.data(), raw_data.data()+raw_data.size());
    temp_data = this->applyFilter(temp_data);

    return Eigen::Map<Vector, Eigen::Unaligned>(temp_data.data(), temp_data.size());
}

}  // end tssfa_se namespace