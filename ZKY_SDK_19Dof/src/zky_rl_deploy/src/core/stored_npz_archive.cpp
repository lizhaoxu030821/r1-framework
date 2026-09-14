#include "zky_rl_deploy/core/stored_npz_archive.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace zky_rl_deploy {
namespace {

constexpr std::uint32_t kZipLocalFileHeaderSignature = 0x04034b50U;
constexpr std::uint32_t kZipCentralDirectorySignature = 0x02014b50U;
constexpr std::uint32_t kZipEndOfCentralDirectorySignature = 0x06054b50U;
constexpr std::uint16_t kZip64ExtendedInformationExtraField = 0x0001U;

std::uint16_t ReadLe16(std::istream& stream) {
  std::uint8_t bytes[2] = {};
  stream.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
  if (!stream) {
    throw std::runtime_error("failed to read 16-bit little-endian value");
  }
  return static_cast<std::uint16_t>(bytes[0]) |
         (static_cast<std::uint16_t>(bytes[1]) << 8U);
}

std::uint32_t ReadLe32(std::istream& stream) {
  std::uint8_t bytes[4] = {};
  stream.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
  if (!stream) {
    throw std::runtime_error("failed to read 32-bit little-endian value");
  }
  return static_cast<std::uint32_t>(bytes[0]) |
         (static_cast<std::uint32_t>(bytes[1]) << 8U) |
         (static_cast<std::uint32_t>(bytes[2]) << 16U) |
         (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

std::uint16_t ReadLe16(const char* data) {
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(data);
  return static_cast<std::uint16_t>(bytes[0]) |
         (static_cast<std::uint16_t>(bytes[1]) << 8U);
}

std::uint64_t ReadLe64(const char* data) {
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(data);
  return static_cast<std::uint64_t>(bytes[0]) |
         (static_cast<std::uint64_t>(bytes[1]) << 8U) |
         (static_cast<std::uint64_t>(bytes[2]) << 16U) |
         (static_cast<std::uint64_t>(bytes[3]) << 24U) |
         (static_cast<std::uint64_t>(bytes[4]) << 32U) |
         (static_cast<std::uint64_t>(bytes[5]) << 40U) |
         (static_cast<std::uint64_t>(bytes[6]) << 48U) |
         (static_cast<std::uint64_t>(bytes[7]) << 56U);
}

std::string ReadBytesAsString(std::istream& stream, std::size_t byte_count) {
  std::string value(byte_count, '\0');
  stream.read(value.data(), static_cast<std::streamsize>(byte_count));
  if (!stream) {
    throw std::runtime_error("failed to read raw bytes");
  }
  return value;
}

std::vector<std::size_t> ParseShapeTuple(const std::string& header) {
  const std::regex shape_regex("'shape'\\s*:\\s*\\(([^\\)]*)\\)");
  std::smatch match;
  if (!std::regex_search(header, match, shape_regex)) {
    throw std::invalid_argument("npy header is missing shape tuple");
  }

  std::vector<std::size_t> shape;
  std::stringstream stream(match[1].str());
  std::string token;
  while (std::getline(stream, token, ',')) {
    token.erase(std::remove_if(token.begin(),
                               token.end(),
                               [](unsigned char ch) { return std::isspace(ch) != 0; }),
                token.end());
    if (token.empty()) {
      continue;
    }
    shape.push_back(static_cast<std::size_t>(std::stoull(token)));
  }
  if (shape.empty()) {
    throw std::invalid_argument("npy shape tuple must contain at least one dimension");
  }
  return shape;
}

std::string ParseDescr(const std::string& header) {
  const std::regex descr_regex("'descr'\\s*:\\s*'([^']+)'");
  std::smatch match;
  if (!std::regex_search(header, match, descr_regex)) {
    throw std::invalid_argument("npy header is missing descr");
  }
  return match[1].str();
}

bool ParseFortranOrder(const std::string& header) {
  const std::regex fortran_regex("'fortran_order'\\s*:\\s*(True|False)");
  std::smatch match;
  if (!std::regex_search(header, match, fortran_regex)) {
    throw std::invalid_argument("npy header is missing fortran_order");
  }
  return match[1].str() == "True";
}

NpyArrayInfo ParseNpyHeader(const std::string& entry_name, std::istream& stream) {
  const std::string magic = ReadBytesAsString(stream, 6U);
  if (magic != std::string("\x93NUMPY", 6U)) {
    throw std::invalid_argument("entry is not a valid .npy payload: " + entry_name);
  }

  const std::uint8_t major_version = static_cast<std::uint8_t>(stream.get());
  const std::uint8_t minor_version = static_cast<std::uint8_t>(stream.get());
  if (!stream) {
    throw std::runtime_error("failed to read npy version");
  }

  std::size_t header_length = 0U;
  if (major_version == 1U) {
    header_length = ReadLe16(stream);
  } else if (major_version == 2U) {
    header_length = ReadLe32(stream);
  } else {
    std::ostringstream error;
    error << "unsupported npy major version " << static_cast<int>(major_version) << "."
          << static_cast<int>(minor_version);
    throw std::invalid_argument(error.str());
  }

  const std::string header = ReadBytesAsString(stream, header_length);

  NpyArrayInfo info;
  info.entry_name = entry_name;
  info.dtype_descr = ParseDescr(header);
  info.fortran_order = ParseFortranOrder(header);
  info.shape = ParseShapeTuple(header);
  return info;
}

std::pair<std::uint64_t, std::uint64_t> ResolveZipEntrySizes(std::uint32_t compressed_size_32,
                                                              std::uint32_t uncompressed_size_32,
                                                              const std::string& extra_field) {
  std::uint64_t compressed_size = compressed_size_32;
  std::uint64_t uncompressed_size = uncompressed_size_32;
  if (compressed_size_32 != 0xFFFFFFFFU && uncompressed_size_32 != 0xFFFFFFFFU) {
    return {compressed_size, uncompressed_size};
  }

  std::size_t cursor = 0U;
  while (cursor + 4U <= extra_field.size()) {
    const std::uint16_t header_id = ReadLe16(extra_field.data() + cursor);
    const std::uint16_t data_size = ReadLe16(extra_field.data() + cursor + 2U);
    cursor += 4U;
    if (cursor + data_size > extra_field.size()) {
      throw std::invalid_argument("zip extra field is truncated");
    }

    if (header_id == kZip64ExtendedInformationExtraField) {
      std::size_t zip64_cursor = cursor;
      if (uncompressed_size_32 == 0xFFFFFFFFU) {
        if (zip64_cursor + 8U > cursor + data_size) {
          throw std::invalid_argument("zip64 extra field is missing uncompressed size");
        }
        uncompressed_size = ReadLe64(extra_field.data() + zip64_cursor);
        zip64_cursor += 8U;
      }
      if (compressed_size_32 == 0xFFFFFFFFU) {
        if (zip64_cursor + 8U > cursor + data_size) {
          throw std::invalid_argument("zip64 extra field is missing compressed size");
        }
        compressed_size = ReadLe64(extra_field.data() + zip64_cursor);
        zip64_cursor += 8U;
      }
      return {compressed_size, uncompressed_size};
    }

    cursor += data_size;
  }

  throw std::invalid_argument("zip64 extra field was required but not found");
}

}  // namespace

StoredNpzArchive StoredNpzArchive::Open(const std::string& npz_path) {
  std::ifstream stream(npz_path, std::ios::binary);
  if (!stream.is_open()) {
    throw std::invalid_argument("failed to open npz file: " + npz_path);
  }

  StoredNpzArchive archive;
  while (true) {
    const std::streampos entry_start = stream.tellg();
    if (entry_start == std::streampos(-1)) {
      break;
    }

    if (stream.peek() == std::char_traits<char>::eof()) {
      break;
    }

    const std::uint32_t signature = ReadLe32(stream);
    if (signature == kZipCentralDirectorySignature ||
        signature == kZipEndOfCentralDirectorySignature) {
      break;
    }
    if (signature != kZipLocalFileHeaderSignature) {
      std::ostringstream error;
      error << "unsupported zip signature 0x" << std::hex << signature << " in " << npz_path;
      throw std::invalid_argument(error.str());
    }

    (void)ReadLe16(stream);  // version needed
    const std::uint16_t general_flags = ReadLe16(stream);
    const std::uint16_t compression_method = ReadLe16(stream);
    (void)ReadLe16(stream);  // mod time
    (void)ReadLe16(stream);  // mod date
    (void)ReadLe32(stream);  // crc32
    const std::uint32_t compressed_size_32 = ReadLe32(stream);
    const std::uint32_t uncompressed_size_32 = ReadLe32(stream);
    const std::uint16_t file_name_length = ReadLe16(stream);
    const std::uint16_t extra_field_length = ReadLe16(stream);

    if (general_flags != 0U) {
      throw std::invalid_argument(
          "npz entry uses unsupported general-purpose flags; expected stored entries without data descriptor");
    }
    if (compression_method != 0U) {
      throw std::invalid_argument(
          "npz entry uses compression; Stage 0 reader currently only supports stored entries");
    }

    const std::string entry_name = ReadBytesAsString(stream, file_name_length);
    const std::string extra_field = ReadBytesAsString(stream, extra_field_length);
    const auto [compressed_size, uncompressed_size] =
        ResolveZipEntrySizes(compressed_size_32, uncompressed_size_32, extra_field);
    if (compressed_size != uncompressed_size) {
      throw std::invalid_argument("stored npz entry has mismatched compressed/uncompressed sizes");
    }

    const std::streampos payload_start = stream.tellg();
    if (payload_start == std::streampos(-1)) {
      throw std::runtime_error("failed to locate npz payload start");
    }
    archive.arrays_.emplace(entry_name, ParseNpyHeader(entry_name, stream));

    stream.clear();
    stream.seekg(payload_start + static_cast<std::streamoff>(compressed_size));
    if (!stream) {
      throw std::runtime_error("failed to seek to next npz entry");
    }
  }

  if (archive.arrays_.empty()) {
    throw std::invalid_argument("npz archive does not contain any readable .npy entries");
  }
  return archive;
}

bool StoredNpzArchive::HasArray(const std::string& entry_name) const {
  return arrays_.find(entry_name) != arrays_.end();
}

const NpyArrayInfo& StoredNpzArchive::GetArrayInfo(const std::string& entry_name) const {
  const auto entry_it = arrays_.find(entry_name);
  if (entry_it == arrays_.end()) {
    throw std::out_of_range("npz archive is missing entry: " + entry_name);
  }
  return entry_it->second;
}

}  // namespace zky_rl_deploy
