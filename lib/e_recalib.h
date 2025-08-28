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

#ifdef _WIN32
    #ifdef BUILDING_E_RECALIB
        #define E_RECALIB_API __declspec(dllexport)
    #else
        #define E_RECALIB_API __declspec(dllimport)
    #endif
#else
    #define E_RECALIB_API __attribute__((visibility("default")))
#endif

#include <opencv2/opencv.hpp>
#include <string>

struct RecalibInfo {
  cv::Vec3d angle_degrees;
  cv::Matx33d K;
};

E_RECALIB_API RecalibInfo recalib(const std::string& input_path,
                    const std::string& intrinsic_path, int image_height = 1080,
                    int image_width = 1920,
                    const cv::Size& pattern_size = cv::Size(6, 3),
                    float square_size = 0.025);

E_RECALIB_API cv::Mat adjust(const RecalibInfo& info, const cv::Mat& im);
