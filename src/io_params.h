/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
 *
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "io_image.h"   // for CpuImage
#include "resources.h"  // for UNSET_SIZET

#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <unordered_map>

enum class ClearWhen : uint32_t
{
  eNever  = 0,
  eAlways = 1,
};

struct StorageBufferParameters
{
  // Buffer size in elements = base + perTile * round(screenSize / tileSize)
  int base      = 0;
  int perTile   = 0;
  int tileSizeX = 1;
  int tileSizeY = 1;

  // Computes the byte size of the buffer if the viewport has size `resolution`
  // and buffer elements must have a stride of `elementStride`. (For simplicity,
  // we include padding at the end if size < stride).
  // This will also fix non-positive tile sizes.
  size_t computeBufferSize(VkExtent2D resolution, size_t elementStride);
};

// These are the values that get saved in the .json file.
struct ShaderParameters
{
  std::unordered_map<std::string, std::array<uint32_t, 16>> uniforms;
  std::unordered_map<std::string, StorageBufferParameters>  storageBuffers;
  // This defaults to cornflower blue for the same reason XNA used it:
  // if nothing's being written, then it's still a good indication that
  // frames are happening.
  ClearWhen            clearColorWhen = ClearWhen::eAlways;
  std::array<float, 4> clearColor{100.f / 255.f, 149.f / 255.f, 237.f / 255.f, 1.0};
  ClearWhen            clearDepthStencilWhen = ClearWhen::eAlways;
  float                clearDepth            = 1.0f;
  uint32_t             clearStencil          = 0;
  size_t               texDisplay            = UNSET_SIZET;  // UNSET_SIZET == use texFrame
  float                timeSpeed             = 1.0;
  double               time                  = 0.0;
  bool                 paused                = false;
};

struct ShaderConfigFile
{
  std::string      description;
  ShaderParameters parameters;
  CpuImage       thumbnail;
};

// Extracts a description from Slang code.
std::string_view extractDescription(const std::string_view& code);

ShaderConfigFile deserializeShaderConfig(const std::filesystem::path& configFile);
// Returns true on success.
bool serializeShaderConfig(const std::filesystem::path& outPath, const ShaderConfigFile& config);

struct ExampleShaderCache
{
public:
  ~ExampleShaderCache();
  // TODO: If we make use of the full app, then we can remove the VkCommandPool
  // and VkQueue arguments
  void init(nvvk::StagingUploader& staging,
            nvapp::Application&    dealloc,
            VkCommandPool          nonOwningTransientCommandPool,
            VkQueue                transferQueue,
            VkSampler              uiSampler);
  void deinit();
  // Handles UI; automatically updates within a few seconds when files change.
  // If the user clicked on an example, returns the path; otherwise, returns
  // nothing.
  std::optional<std::filesystem::path> doUI();

private:
  struct Implementation;
  Implementation* m_implementation = nullptr;
};
