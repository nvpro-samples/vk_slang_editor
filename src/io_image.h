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

// Contains functions and caches for reading/writing images.

#include "resources.h"

#include <nvapp/application.hpp>
#include <nvvk/resources.hpp>
#include <nvvk/staging.hpp>

#include <vulkan/vulkan_core.h>

#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

struct CpuImage
{
  VkExtent3D      size{};
  VkFormat        format    = VK_FORMAT_UNDEFINED;
  VkImageViewType dimension = VK_IMAGE_VIEW_TYPE_2D;

  void allocate(uint32_t numMips = 1, uint32_t numLayers = 1, uint32_t numFaces = 1)
  {
    m_subresources.resize(numMips * numLayers * numFaces);
    m_numMips   = numMips;
    m_numLayers = numLayers;
    m_numFaces  = numFaces;
  }
  std::vector<char>& subresource(uint32_t mip = 0, uint32_t layer = 0, uint32_t face = 0)
  {
    return m_subresources[(mip * m_numLayers + layer) * m_numFaces + face];
  }
  const std::vector<char>& subresource(uint32_t mip = 0, uint32_t layer = 0, uint32_t face = 0) const
  {
    return m_subresources[(mip * m_numLayers + layer) * m_numFaces + face];
  }
  uint32_t getNumMips() const { return m_numMips; }
  uint32_t getNumLayers() const { return m_numLayers; }
  uint32_t getNumFaces() const { return m_numFaces; }
  bool     empty() const { return m_subresources.empty(); }

private:
  std::vector<std::vector<char>> m_subresources;
  uint32_t                       m_numMips   = 1;
  uint32_t                       m_numLayers = 1;
  uint32_t                       m_numFaces  = 1;
};

// Disk -> CPU
VkResult decompressImage(CpuImage& result, const std::filesystem::path& path);
// compressed CPU -> decompressed CPU
VkResult decompressImageFromMemory(CpuImage& result, const void* memory, const size_t numBytes);
// CPU -> GPU
VkResult cmdUpload(VkCommandBuffer        cmd,
                   Texture&               texDst,
                   const CpuImage&        texSrc,
                   VkSampler              uiSampler,
                   nvvk::StagingUploader& staging,
                   nvapp::Application&    dealloc);
// GPU -> CPU. Image must be accessible by the host; its layout will not be
// changed.
// This might be over-engineered.
std::optional<std::vector<uint8_t>> compressImageToJpeg(CpuImage image, int jpegQuality);

struct TextureCache
{
  ~TextureCache();
  // Disk -> GPU. Loads a newer version of the file if there is one.
  VkResult cmdUpload(VkCommandBuffer              cmd,
                     Texture&                     texDst,
                     const std::filesystem::path& path,
                     VkSampler                    uiSampler,
                     nvvk::StagingUploader&       staging,
                     nvapp::Application&          dealloc);

private:
  struct Implementation;
  Implementation* m_implementation = nullptr;
};
