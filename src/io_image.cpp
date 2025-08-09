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

#include "io_image.h"
#include "utilities.h"

#include <nvimageformats/nv_dds.h>
#include <nvimageformats/nv_ktx.h>
#include <nvimageformats/texture_formats.h>
#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvvk/check_error.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#include <volk.h>

// An std::streambuf interface that operates on an in-memory array.
// Note: If this works, then nv_dds::MemoryStreamBuffer can be much simplified.
// This can also be removed once nv_ktx gets a readFromMemory function.
class MemoryStreamBuffer : public std::streambuf
{
public:
  MemoryStreamBuffer(char* data, size_t size) { setg(data, data, data + size); }
};

// An std::istream interface for a constant, in-memory array.
class MemoryStream : public std::istream
{
public:
  MemoryStream(const char* data, size_t sizeInBytes)
      : m_buffer(const_cast<char*>(data), sizeInBytes)
      , std::istream(&m_buffer)
  {
  }

private:
  MemoryStreamBuffer m_buffer;
};

VkResult decompressImage(CpuImage& result, const std::filesystem::path& path)
{
  const std::string fileContents = nvutils::loadFile(path);
  return decompressImageFromMemory(result, fileContents.data(), fileContents.size());
}

VkResult decompressImageFromMemory(CpuImage& result, const void* memory, const size_t numBytes)
{
  const stbi_uc* fileBytes = reinterpret_cast<const stbi_uc*>(memory);

  if(numBytes >= std::numeric_limits<int>::max())
  {
    LOGE("File too large!\n");
    return VK_ERROR_OUT_OF_HOST_MEMORY;
  }

  // 4 bytes is enough to identify files using magic numbers; conversely,
  // any images with 3 bytes or less are almost certainly invalid.
  if(numBytes < 4)
  {
    LOGE("File too short! It was only %zu byte(s) long.\n", numBytes);
    return VK_ERROR_FORMAT_NOT_SUPPORTED;
  }

  // Try KTX2 first
  const uint8_t ktxIdentifier[4] = {0xAB, 0x4B, 0x54, 0x58};
  if(memcmp(fileBytes, ktxIdentifier, 4) == 0)
  {
    // We don't currently handle the full KTX2 format; we only load the first
    // face, layer, and mip. Some of this code is set up for when we do handle
    // it, though.
    nv_ktx::KTXImage            image;
    MemoryStream                stream(reinterpret_cast<const char*>(memory), numBytes);
    const nv_ktx::ErrorWithText maybeError = image.readFromStream(stream, {});
    if(maybeError.has_value())
    {
      LOGE("File read with nv_ktx failed: %s\n", maybeError.value().c_str());
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

    // Note: It would be really nice if nv_ktx included this as a helper function.
    const bool isArray = (image.num_layers_possibly_0 > 0);
    switch(image.getImageType())
    {
      case VK_IMAGE_TYPE_1D:
        result.dimension = isArray ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D;
        break;
      case VK_IMAGE_TYPE_2D:
        if(image.num_faces > 1)
        {
          result.dimension = isArray ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE;
        }
        else
        {
          result.dimension = isArray ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
        }
        break;
      case VK_IMAGE_TYPE_3D:
        result.dimension = VK_IMAGE_VIEW_TYPE_3D;
        break;
      default:
        assert(!"Should be unreachable");
        break;
    }
    result.format = image.format;
    result.size = VkExtent3D{std::max(1U, image.mip_0_width), std::max(1U, image.mip_0_height), std::max(1U, image.mip_0_depth)};
    result.allocate(image.num_mips, std::max(1U, image.num_layers_possibly_0), image.num_faces);
    for(uint32_t mip = 0; mip < result.getNumMips(); mip++)
    {
      for(uint32_t layer = 0; layer < result.getNumLayers(); layer++)
      {
        for(uint32_t face = 0; face < result.getNumFaces(); face++)
        {
          result.subresource(mip, layer, face) = std::move(image.subresource(mip, layer, face));
        }
      }
    }
    return VK_SUCCESS;
  }

  // Try DDS
  if(memcmp(fileBytes, "DDS ", 4) == 0)
  {
    // Similarly, we don't currently handle the full DDS format.
    nv_dds::Image               image;
    const nv_dds::ErrorWithText maybeError = image.readFromMemory(reinterpret_cast<const char*>(memory), numBytes, {});
    if(maybeError.has_value())
    {
      LOGE("File read with nv_dds filed: %s\n", maybeError.value().c_str());
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

    if(image.getNumFaces() != 1 && image.getNumFaces() != 6)
    {
      LOGE(TARGET_NAME " does not support DDS files with incomplete cubemaps; this file had %u faces.\n", image.getNumFaces());
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

    const bool isArray = (image.getNumLayers() > 1);
    switch(image.inferResourceDimension())
    {
      case nv_dds::ResourceDimension::eBuffer:
      case nv_dds::ResourceDimension::eTexture1D:
        result.dimension = isArray ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D;
        break;
      case nv_dds::ResourceDimension::eTexture2D:
        if(image.getNumFaces() == 6)
        {
          result.dimension = isArray ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE;
        }
        else
        {
          result.dimension = isArray ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
        }
        break;
      case nv_dds::ResourceDimension::eTexture3D:
        result.dimension = VK_IMAGE_VIEW_TYPE_3D;
        break;
      default:
        break;
    }
    result.format = texture_formats::dxgiToVulkan(image.dxgiFormat);
    result.size   = VkExtent3D{image.mip0Width, image.mip0Height, image.mip0Depth};
    result.allocate(image.getNumMips(), std::max(1U, image.getNumLayers()), image.getNumFaces());
    for(uint32_t mip = 0; mip < result.getNumMips(); mip++)
    {
      for(uint32_t layer = 0; layer < result.getNumLayers(); layer++)
      {
        for(uint32_t face = 0; face < result.getNumFaces(); face++)
        {
          result.subresource(mip, layer, face) = std::move(image.subresource(mip, layer, face).data);
        }
      }
    }
    return VK_SUCCESS;
  }

  // Try stb_image
  const int fileSizeI = static_cast<int>(numBytes);
  if(stbi_is_hdr_from_memory(fileBytes, fileSizeI))
  {
    // Using VK_FORMAT_R32G32B32A32_SFLOAT is pretty wasteful here!
    // But we need at least 8 bits in the exponent for RGBE's E8R8G8B8 format.
    result.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    int    w = 0, h = 0, realComponents = 4;
    float* data = stbi_loadf_from_memory(fileBytes, fileSizeI, &w, &h, &realComponents, 4);
    if(!data)
    {
      LOGE("stbi_loadf_from_memory failed. stbi_failure_reason: %s\n", stbi_failure_reason());
      return VK_ERROR_UNKNOWN;
    }
    result.size = {static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1};
    result.allocate(1, 1, 1);
    const char* dataBytes = reinterpret_cast<const char*>(data);
    result.subresource(0, 0, 0).assign(dataBytes,
                                       dataBytes + static_cast<size_t>(w) * static_cast<size_t>(h) * 4 * sizeof(data[0]));
    stbi_image_free(data);
    return VK_SUCCESS;
  }

  if(stbi_is_16_bit_from_memory(fileBytes, fileSizeI))
  {
    // 16-bit RGBA
    result.format = VK_FORMAT_R16G16B16A16_UNORM;
    int       w = 0, h = 0, realComponents = 4;
    uint16_t* data = stbi_load_16_from_memory(fileBytes, fileSizeI, &w, &h, &realComponents, 4);
    if(!data)
    {
      LOGE("stbi_is_16_bit_from_memory failed. stbi_failure_reason: %s\n", stbi_failure_reason());
      return VK_ERROR_UNKNOWN;
    }
    result.size = {static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1};
    result.allocate(1, 1, 1);
    const char* dataBytes = reinterpret_cast<const char*>(data);
    result.subresource(0, 0, 0).assign(dataBytes,
                                       dataBytes + static_cast<size_t>(w) * static_cast<size_t>(h) * 4 * sizeof(data[0]));
    stbi_image_free(data);
    return VK_SUCCESS;
  }

  // Otherwise:
  {
    // Use regular 8-bit RGBA loader.
    result.format = VK_FORMAT_R8G8B8A8_UNORM;
    int      w = 0, h = 0, realComponents = 4;
    uint8_t* data = stbi_load_from_memory(fileBytes, fileSizeI, &w, &h, &realComponents, 4);
    if(!data)
    {
      LOGE("stbi_load_from_memory failed. stbi_failure_reason: %s\n", stbi_failure_reason());
      return VK_ERROR_UNKNOWN;
    }
    result.size = {static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1};
    result.allocate(1, 1, 1);
    const char* dataBytes = reinterpret_cast<const char*>(data);
    result.subresource(0, 0, 0).assign(dataBytes,
                                       dataBytes + static_cast<size_t>(w) * static_cast<size_t>(h) * 4 * sizeof(data[0]));
    stbi_image_free(data);
    return VK_SUCCESS;
  }
}

VkResult cmdUpload(VkCommandBuffer cmd, Texture& texDst, const CpuImage& texSrc, VkSampler uiSampler, nvvk::StagingUploader& staging, nvapp::Application& dealloc)
{
  texDst.format = texSrc.format;
  NVVK_FAIL_RETURN(texDst.resize(cmd, texSrc.size, texSrc.getNumMips(), texSrc.getNumLayers() * texSrc.getNumFaces(),
                                 uiSampler, *staging.getResourceAllocator(), dealloc));

  nvvk::BarrierContainer barriers;
  texDst.addTransitionTo(barriers.imageBarriers, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, nvvk::INFER_BARRIER_PARAMS, true);
  barriers.cmdPipelineBarrier(cmd, 0);

  for(uint32_t mip = 0; mip < texSrc.getNumMips(); mip++)
  {
    const VkExtent3D size{std::max(1U, texSrc.size.width >> mip),   //
                          std::max(1U, texSrc.size.height >> mip),  //
                          std::max(1U, texSrc.size.depth >> mip)};

    for(uint32_t layer = 0; layer < texSrc.getNumLayers(); layer++)
    {
      for(uint32_t face = 0; face < texSrc.getNumFaces(); face++)
      {
        const uint32_t vkLayer = layer * texSrc.getNumFaces() + face;
        staging.appendImageSub(texDst.image,  //dstImage
                               {0, 0, 0},     // offset
                               size,          // extent
                               VkImageSubresourceLayers{.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                                                        .mipLevel       = mip,
                                                        .baseArrayLayer = vkLayer,
                                                        .layerCount     = 1},        // subresource
                               std::span(texSrc.subresource(mip, layer, face)),  // data
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);            // layout
      }
    }
  }

  return VK_SUCCESS;
}

using CompressedImage = std::vector<uint8_t>;

static void stbiWriteCallback(void* context, void* data, int size)
{
  CompressedImage& image = *reinterpret_cast<CompressedImage*>(context);
  const uint8_t*   bytes = reinterpret_cast<const uint8_t*>(data);
  image.insert(image.end(), bytes, bytes + size);
}

std::optional<std::vector<uint8_t>> compressImageToJpeg(CpuImage image, int jpegQuality)
{
  CompressedImage result;
  const int       width  = static_cast<int>(image.size.width);
  const int       height = static_cast<int>(image.size.height);
  assert(image.format == VK_FORMAT_R8G8B8A8_UNORM);
  const int components = 4;
  if(!stbi_write_jpg_to_func(stbiWriteCallback, &result, width, height, components, image.subresource(0,0,0).data(), jpegQuality))
  {
    return {};
  }
  return std::move(result);
}

//-----------------------------------------------------------------------------
// Texture cache

struct TextureCache::Implementation
{
private:
  struct CacheEntry
  {
    CpuImage                        image;
    std::filesystem::file_time_type lastModified;
  };
  // TODO: Improve this so that we don't need to do CPU->GPU texture uploads
  // every compile. This would probably mean Textures become reference-counted.
  std::unordered_map<std::filesystem::path, CacheEntry, PathHash> m_cache;

public:
  VkResult cmdUpload(VkCommandBuffer              cmd,
                     Texture&                     texDst,
                     const std::filesystem::path& path,
                     VkSampler                    uiSampler,
                     nvvk::StagingUploader&       staging,
                     nvapp::Application&          dealloc)
  {
    // No matter what, we'll need the time the file was modified:
    const std::filesystem::file_time_type diskLastModified = std::filesystem::last_write_time(path);

    // And we'll get a pointer to the cache entry:
    const CacheEntry* cacheEntry = nullptr;

    // Do we need to load this data from disk?
    bool needLoad = true;
    {
      const auto& cacheSearch = m_cache.find(path);
      if(cacheSearch != m_cache.end())
      {
        needLoad = (diskLastModified != cacheSearch->second.lastModified);
        if(!needLoad)
        {
          cacheEntry = &(cacheSearch->second);
        }
        else
        {
          LOGI("Image `%s` changed on disk; reloading.\n", nvutils::utf8FromPath(path).c_str());
        }
      }
    }

    if(needLoad)
    {
      CacheEntry newEntry{.lastModified = diskLastModified};
      NVVK_FAIL_RETURN(decompressImage(newEntry.image, path));
      cacheEntry = &(m_cache[path] = std::move(newEntry));
    }

    // Create the texture and upload its data:
    assert(cacheEntry);
    return ::cmdUpload(cmd, texDst, cacheEntry->image, uiSampler, staging, dealloc);
  }
};

TextureCache::~TextureCache()
{
  delete m_implementation;
}

VkResult TextureCache::cmdUpload(VkCommandBuffer              cmd,
                                 Texture&                     texDst,
                                 const std::filesystem::path& path,
                                 VkSampler                    uiSampler,
                                 nvvk::StagingUploader&       staging,
                                 nvapp::Application&          dealloc)
{
  if(!m_implementation)
  {
    m_implementation = new Implementation();
  }
  return m_implementation->cmdUpload(cmd, texDst, path, uiSampler, staging, dealloc);
}
