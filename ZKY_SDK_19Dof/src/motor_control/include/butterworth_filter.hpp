/* ----------------------------------------------------------------------------
 * Copyright (c) 2021, University of Leeds and Harbin Institute of Technology.
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   butterworth_filter.hpp
 *  @author Jun Li (junlileeds@gmail.com)
 *  @brief  Header file for ButterworthFilter class
 *  @date   June 16, 2022
 **/

#pragma once

#include <cmath>
#include <vector>
#include <iostream>

#include <Eigen/Dense>

#define MAX_FILTER_LENGTH 8

namespace tssfa_se {

class ButterworthFilter
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Ref<Vector>               RefVector;
    typedef const Eigen::Ref<const Vector>&  ConstRefVector;

    ButterworthFilter(double dt, double cutoff, int N);

    void clearFilter();

    /** Build a butterworth filter. T is the sample period,
    *  cutoff is the cutoff frequency in hertz, N is the order (1,2,3 or 4)
    */
    void butterworth(double T, double cutoff, int N);

    std::vector<double> applyFilter(const std::vector<double>& X);

    Vector applyButterworth(ConstRefVector raw_data);

private:
    bool init_;

    double dt_;  // sample time [s]
    double cutoff_;  // cutoff frequency [Hz]
    int N_;  // order of the filter

    double a_[MAX_FILTER_LENGTH];  // Y coefficient
    double b_[MAX_FILTER_LENGTH];  // X coefficient

    std::vector<std::vector<double>> Xvec_;
    std::vector<std::vector<double>> Yvec_;
};

}  // end tssfa_se namespace