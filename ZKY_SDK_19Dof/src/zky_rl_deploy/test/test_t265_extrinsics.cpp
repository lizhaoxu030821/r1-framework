#include <gtest/gtest.h>

#include <array>
#include <cmath>

#include "zky_rl_deploy/core/t265_extrinsics.hpp"

namespace zky_rl_deploy {
namespace {

using QuaternionXyzw = T265Extrinsics::QuaternionXyzw;
using Vector3 = T265Extrinsics::Vector3;

QuaternionXyzw Multiply(const QuaternionXyzw& lhs, const QuaternionXyzw& rhs) {
  return {
      lhs[3] * rhs[0] + lhs[0] * rhs[3] + lhs[1] * rhs[2] - lhs[2] * rhs[1],
      lhs[3] * rhs[1] - lhs[0] * rhs[2] + lhs[1] * rhs[3] + lhs[2] * rhs[0],
      lhs[3] * rhs[2] + lhs[0] * rhs[1] - lhs[1] * rhs[0] + lhs[2] * rhs[3],
      lhs[3] * rhs[3] - lhs[0] * rhs[0] - lhs[1] * rhs[1] - lhs[2] * rhs[2],
  };
}

QuaternionXyzw Conjugate(const QuaternionXyzw& quaternion) {
  return {-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]};
}

QuaternionXyzw Normalize(const QuaternionXyzw& quaternion) {
  const double norm =
      std::sqrt(quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1] +
                quaternion[2] * quaternion[2] + quaternion[3] * quaternion[3]);
  return {quaternion[0] / norm,
          quaternion[1] / norm,
          quaternion[2] / norm,
          quaternion[3] / norm};
}

QuaternionXyzw RelativeRotation(const QuaternionXyzw& start, const QuaternionXyzw& end) {
  return Normalize(Multiply(Conjugate(Normalize(start)), Normalize(end)));
}

QuaternionXyzw AxisAngleZ90() {
  return {0.0, 0.0, std::sqrt(0.5), std::sqrt(0.5)};
}

QuaternionXyzw AxisAngleX90() {
  return {std::sqrt(0.5), 0.0, 0.0, std::sqrt(0.5)};
}

QuaternionXyzw AxisAngleY90() {
  return {0.0, std::sqrt(0.5), 0.0, std::sqrt(0.5)};
}

void ExpectSameRotation(const QuaternionXyzw& actual, const QuaternionXyzw& expected) {
  const QuaternionXyzw normalized_actual = Normalize(actual);
  const QuaternionXyzw normalized_expected = Normalize(expected);
  const double dot = normalized_actual[0] * normalized_expected[0] +
                     normalized_actual[1] * normalized_expected[1] +
                     normalized_actual[2] * normalized_expected[2] +
                     normalized_actual[3] * normalized_expected[3];
  EXPECT_NEAR(std::abs(dot), 1.0, 1e-9);
}

TEST(T265ExtrinsicsTest, MapsRawAngularVelocityAxesIntoCanonicalFlu) {
  const T265ExtrinsicConfig extrinsic;

  EXPECT_EQ(T265Extrinsics::TransformImuOpticalAngularVelocityToPelvis({0.0, 0.0, 1.0}, extrinsic),
            (Vector3{1.0, 0.0, 0.0}));
  EXPECT_EQ(T265Extrinsics::TransformImuOpticalAngularVelocityToPelvis({1.0, 0.0, 0.0}, extrinsic),
            (Vector3{0.0, -1.0, 0.0}));
  EXPECT_EQ(T265Extrinsics::TransformImuOpticalAngularVelocityToPelvis({0.0, 1.0, 0.0}, extrinsic),
            (Vector3{0.0, 0.0, -1.0}));
}

TEST(T265ExtrinsicsTest, MapsRawOrientationAxesIntoExpectedRelativeBodyRotations) {
  const T265ExtrinsicConfig extrinsic;
  const QuaternionXyzw base =
      T265Extrinsics::TransformPoseFrameOrientationToPelvis({0.0, 0.0, 0.0, 1.0}, extrinsic);

  ExpectSameRotation(
      RelativeRotation(base,
                       T265Extrinsics::TransformPoseFrameOrientationToPelvis(AxisAngleX90(), extrinsic)),
      AxisAngleX90());
  ExpectSameRotation(
      RelativeRotation(base,
                       T265Extrinsics::TransformPoseFrameOrientationToPelvis(AxisAngleY90(), extrinsic)),
      AxisAngleY90());
  ExpectSameRotation(
      RelativeRotation(base,
                       T265Extrinsics::TransformPoseFrameOrientationToPelvis(AxisAngleZ90(), extrinsic)),
      AxisAngleZ90());
}

TEST(T265ExtrinsicsTest, AppliesPelvisTranslationInWorldUsingPelvisOrientation) {
  const T265ExtrinsicConfig extrinsic{{0.0, 0.0, 0.0, 1.0}, {0.1, 0.0, 0.0}};

  EXPECT_EQ(T265Extrinsics::TransformPoseFramePositionToPelvis({1.0, 2.0, 3.0},
                                                               {0.0, 0.0, 0.0, 1.0},
                                                               extrinsic),
            (Vector3{0.9, 2.0, 3.0}));

  const QuaternionXyzw yaw_90 = AxisAngleZ90();
  const Vector3 translated =
      T265Extrinsics::TransformPoseFramePositionToPelvis({1.0, 2.0, 3.0}, yaw_90, extrinsic);
  EXPECT_NEAR(translated[0], 1.0, 1e-9);
  EXPECT_NEAR(translated[1], 1.9, 1e-9);
  EXPECT_NEAR(translated[2], 3.0, 1e-9);
}

}  // namespace
}  // namespace zky_rl_deploy

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
