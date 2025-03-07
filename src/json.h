/*
 * Copyright (c) 2022-2025, William Wei. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <nlohmann/json.hpp>
#include <Eigen/Dense>

using json = nlohmann::json;

inline json read_json(const std::string& path) {
  std::ifstream f(path);
  return json::parse(f);
}

inline std::vector<json> retrieve_info(const std::string& info_path) {
  std::ifstream file(info_path);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << info_path << std::endl;
    return {};
  }

  std::vector<json> json_objects;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty())
      continue;

    json json_obj;
    auto result = json::parse(line, nullptr, false);
    if (result.is_discarded()) {
      std::cerr << "JSON parsing error in line: " << line << std::endl;
      continue;
    }

    json_objects.push_back(result);
  }

  return json_objects;
}

inline json retrieve_pack_info_by_frame(const std::vector<json>& frames_struct,
                                        int frame_id,
                                        const std::string& key = "lanes") {
  if (frame_id < 1 || frame_id > frames_struct.size()) {
    std::cerr << "Error: frame_id is out of bounds." << std::endl;
    return json();
  }

  const json& frame_data = frames_struct[frame_id - 1];

  if (!frame_data.contains(key)) {
    std::cerr << "Error: Key '" << key << "' not found in frame data."
              << std::endl;
    return json();
  }

  return frame_data[key];
}


inline std::vector<Eigen::MatrixXf> json_to_eigen_matrix(const json& json_data) {
  std::vector<Eigen::MatrixXf> m;
  for (const auto& data : json_data) {
    if (data.empty())
      continue;
    size_t rows = data.size();
    size_t cols = data[0].size();

    Eigen::MatrixXf mat(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
      for (size_t k = 0; k < cols; ++k) {
        mat(i, k) = data[i][k].get<float>();
      }
    }

    m.push_back(mat);
  }
  return m;
}