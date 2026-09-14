#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace zky_rl_deploy {

struct NpyArrayInfo {
  std::string entry_name;
  std::string dtype_descr;
  bool fortran_order{false};
  std::vector<std::size_t> shape;
};

// 这个工具只读取“stored/uncompressed”的 .npz 目录和 .npy header，
// 目的是在 Stage 0 不引入额外第三方依赖的前提下，先把 NPZ shape 契约锁定。
class StoredNpzArchive {
 public:
  static StoredNpzArchive Open(const std::string& npz_path);

  bool HasArray(const std::string& entry_name) const;
  const NpyArrayInfo& GetArrayInfo(const std::string& entry_name) const;

 private:
  std::unordered_map<std::string, NpyArrayInfo> arrays_;
};

}  // namespace zky_rl_deploy
