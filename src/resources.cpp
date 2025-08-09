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

#include "resources.h"

#include "io_image.h"

#include <nvapp/application.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvvk/barriers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/staging.hpp>

#include <backends/imgui_impl_vulkan.h>
#include <vulkan/vk_enum_string_helper.h>

#include <unordered_map>

static VkImageAspectFlags getAspectMask(VkFormat format)
{
  // TODO: Add other formats
  if(format == VK_FORMAT_D24_UNORM_S8_UINT)
  {
    return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
  }
  return VK_IMAGE_ASPECT_COLOR_BIT;
}

VkResult Texture::resize(VkCommandBuffer          cmd,
                         VkExtent3D               newSize,
                         uint32_t                 numMips,
                         uint32_t                 numVkLayers,
                         VkSampler                uiSampler,
                         nvvk::ResourceAllocator& alloc,
                         nvapp::Application&      dealloc)
{
  if(newSize.width == size.width && newSize.height == size.height && newSize.depth == size.depth
     && numMips == image.mipLevels && numVkLayers == image.arrayLayers)
  {
    return VK_SUCCESS;  // Nothing to do
  }

  dealloc.submitResourceFree([&alloc, tex = *this]() mutable { tex.deinitResources(alloc); });
  size = newSize;

  const VkImageAspectFlags aspect = getAspectMask(format);
#if 0
  const VkImageAspectFlags stencilAspect   = aspect & VK_IMAGE_ASPECT_STENCIL_BIT;
  const VkImageAspectFlags noStencilAspect = aspect & ~VK_IMAGE_ASPECT_STENCIL_BIT;
#endif

  VkImageType imageType{};
  switch(dimension)
  {
    case VK_IMAGE_VIEW_TYPE_1D:
    case VK_IMAGE_VIEW_TYPE_1D_ARRAY:
      imageType = VK_IMAGE_TYPE_1D;
      break;
    case VK_IMAGE_VIEW_TYPE_2D:
    case VK_IMAGE_VIEW_TYPE_CUBE:
    case VK_IMAGE_VIEW_TYPE_2D_ARRAY:
    case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY:
      imageType = VK_IMAGE_TYPE_2D;
      break;
    case VK_IMAGE_VIEW_TYPE_3D:
      imageType = VK_IMAGE_TYPE_3D;
      break;
    default:
      LOGE("Invalid image type 0x%x!\n", static_cast<unsigned>(imageType));
      break;
  }

  const VkImageCreateInfo info{.sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                               .imageType   = imageType,
                               .format      = format,
                               .extent      = newSize,
                               .mipLevels   = numMips,
                               .arrayLayers = numVkLayers,
                               .samples     = VK_SAMPLE_COUNT_1_BIT,
                               .usage       = usage};

  const VkImageViewCreateInfo rgbaViewInfo{.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                           .viewType = dimension,
                                           .subresourceRange = {.aspectMask = aspect, .levelCount = numMips, .layerCount = numVkLayers}};
  NVVK_FAIL_RETURN(alloc.createImage(image, info, rgbaViewInfo));
  nvvk::DebugUtil::getInstance().setObjectName(image.image, name);
  nvvk::DebugUtil::getInstance().setObjectName(image.descriptor.imageView, name);

  const VkImageAspectFlags    noStencilAspect = aspect & ~VK_IMAGE_ASPECT_STENCIL_BIT;
  const VkImageViewCreateInfo rgb1ViewInfo{
      .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image    = image.image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format   = format,
      .components = {.r = VK_COMPONENT_SWIZZLE_IDENTITY, .g = VK_COMPONENT_SWIZZLE_IDENTITY, .b = VK_COMPONENT_SWIZZLE_IDENTITY, .a = VK_COMPONENT_SWIZZLE_ONE},
      .subresourceRange = {.aspectMask = noStencilAspect, .levelCount = numMips, .layerCount = numVkLayers}};
  NVVK_FAIL_RETURN(vkCreateImageView(alloc.getDevice(), &rgb1ViewInfo, nullptr, &uiImageView));
  nvvk::DebugUtil::getInstance().setObjectName(uiImageView, name + ".uiImageView");

  uiDescriptorSet = ImGui_ImplVulkan_AddTexture(uiSampler, uiImageView, guiImageLayout);
  nvvk::DebugUtil::getInstance().setObjectName(uiDescriptorSet, name + ".uiDescriptorSet");
  return VK_SUCCESS;
}

void Texture::deinitResources(nvvk::ResourceAllocator& alloc)
{
  if(uiDescriptorSet)
  {
    ImGui_ImplVulkan_RemoveTexture(uiDescriptorSet);
    uiDescriptorSet = VK_NULL_HANDLE;
    vkDestroyImageView(alloc.getDevice(), uiImageView, nullptr);
    alloc.destroyImage(image);
  }
}

void Texture::addTransitionTo(std::vector<VkImageMemoryBarrier2>& barriers, VkImageLayout newLayout, VkPipelineStageFlags2 newStages, bool mayWrite)
{
  // We can skip this if we're reading and then reading again, and if the
  // stages and layout aren't changing.
  if(image.descriptor.imageLayout == newLayout && currentStages == newStages && !maybeWritten && !mayWrite)
  {
    return;
  }

  VkImageMemoryBarrier2 barrier = nvvk::makeImageMemoryBarrier(nvvk::ImageMemoryBarrierParams{
      .image            = image.image,
      .oldLayout        = image.descriptor.imageLayout,
      .newLayout        = newLayout,
      .subresourceRange = {getAspectMask(format), 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS},
      .srcStageMask     = currentStages,
      .dstStageMask     = newStages});
  barriers.push_back(std::move(barrier));

  image.descriptor.imageLayout = newLayout;
  currentStages                = barrier.dstStageMask;
  maybeWritten                 = mayWrite;
}

template <class ElementT>
VkResult textureFillUninitializedPatternInternal(Texture&                    tex,
                                                 nvvk::StagingUploader&      staging,
                                                 const nvvk::SemaphoreState& semaphoreState,
                                                 ElementT                    checker0,
                                                 ElementT                    checker1,
                                                 ElementT                    text)
{
  // Draw the main swatch to a temporary 8x32 buffer so that filling the whole image is faster.
  // This is the "UNINIT" text; 1s are text and 0s are background; MSBs are smaller x values.
  const std::array<uint32_t, 4>           textLines = {0b00010010100101110100101110111000,  //
                                                       0b00010010110100100110100100010000,  //
                                                       0b00010010101100100101100100010000,  //
                                                       0b00001100100101110100101110010000};
  std::array<std::array<ElementT, 32>, 8> swatch;
  for(size_t y = 0; y < 8; y++)
  {
    for(size_t x = 0; x < 32; x++)
    {
      const bool checker = ((x & 4) != (y & 4));
      ElementT   texel   = (checker ? checker1 : checker0);
      if(2 <= y && y < 6)
      {
        const uint32_t line = textLines[y - 2];
        if((line >> (31 - x)) & 1)
        {
          texel = text;
        }
      }
      swatch[y][x] = texel;
    }
  }

  // Fill the whole array:
  for(uint32_t mip = 0; mip < tex.image.mipLevels; mip++)
  {
    const size_t          width  = std::max(1u, tex.image.extent.width >> mip);
    const size_t          height = std::max(1u, tex.image.extent.height >> mip);
    const size_t          depth  = std::max(1u, tex.image.extent.depth >> mip);
    const uint32_t        layers = tex.image.arrayLayers;
    std::vector<ElementT> data(layers * width * height * depth);
    for(size_t layer = 0; layer < layers; layer++)
    {
      for(size_t z = 0; z < depth; z++)
      {
        for(size_t y = 0; y < height; y++)
        {
          for(size_t x = 0; x < width; x += 32)
          {
            const size_t copyWidth = std::min(size_t(32), width - x);
            memcpy(&data[((layer * depth + z) * height + y) * width + x],  //
                   swatch[y % swatch.size()].data(),                       //
                   sizeof(ElementT) * copyWidth);
          }
        }
      }
    }

    NVVK_FAIL_RETURN(staging.appendImageSub(
        tex.image,         // image
        {0, 0, 0},         // offset
        tex.image.extent,  // extent
        VkImageSubresourceLayers{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = mip, .baseArrayLayer = 0, .layerCount = layers},
        std::span<ElementT>(data),             // data
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,  //newLayout
        semaphoreState));                      // semaphoreState
  }

  return VK_SUCCESS;
}

VkResult Texture::fillUninitializedPattern(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const nvvk::SemaphoreState& sem)
{
  if(!image.image)
  {
    return VK_SUCCESS;
  }

  nvvk::BarrierContainer barriers;
  addTransitionTo(barriers.imageBarriers, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT, true);
  barriers.cmdPipelineBarrier(cmd, 0);

  // Switch off the texture format (currently handling the most common ones).
  // In all cases, the top-left checkerboard color is 75% transparent pink;
  // the second checkerboard color is 0% transparent black; and the text color
  // is 100% opaque white.
  switch(format)
  {
    case VK_FORMAT_R32G32B32A32_SFLOAT:
      return textureFillUninitializedPatternInternal<std::array<float, 4>>(*this, staging, sem, {1.0f, 0.f, 0.5f, 0.75f},
                                                                           {0.f, 0.f, 0.f, 0.f}, {1.0f, 1.0f, 1.0f, 1.0f});
    case VK_FORMAT_R16G16B16A16_SFLOAT:
      return textureFillUninitializedPatternInternal<std::array<uint16_t, 4>>(*this, staging, sem, {0x3c00, 0, 0x3800, 0x3a00},
                                                                              {0, 0, 0, 0}, {0x3c00, 0x3c00, 0x3c00, 0x3c00});
    case VK_FORMAT_R32G32_SFLOAT:
      return textureFillUninitializedPatternInternal<std::array<float, 2>>(*this, staging, sem, {1.0f, 0.f}, {0.f, 0.f},
                                                                           {1.0f, 1.0f});
    case VK_FORMAT_R16G16_SFLOAT:
      return textureFillUninitializedPatternInternal<std::array<uint16_t, 2>>(*this, staging, sem, {0x3c00, 0}, {0, 0},
                                                                              {0x3c00, 0x3c00});
    case VK_FORMAT_R32_SFLOAT:
      return textureFillUninitializedPatternInternal<float>(*this, staging, sem, 0.75f, 0.0f, 1.0f);
    case VK_FORMAT_R16_SFLOAT:
      return textureFillUninitializedPatternInternal<uint16_t>(*this, staging, sem, 0x3a00, 0, 0x3c00);
    case VK_FORMAT_R16G16B16A16_UNORM:
      return textureFillUninitializedPatternInternal<std::array<uint16_t, 4>>(*this, staging, sem, {65535, 0, 32768, 49152},
                                                                              {0, 0, 0, 0}, {65535, 65535, 65535, 65535});
    case VK_FORMAT_R8G8B8A8_UNORM:
      return textureFillUninitializedPatternInternal<std::array<uint8_t, 4>>(*this, staging, sem, {255, 0, 128, 192},
                                                                             {0, 0, 0, 0}, {255, 255, 255, 255});
    case VK_FORMAT_R16G16_UNORM:
      return textureFillUninitializedPatternInternal<std::array<uint16_t, 2>>(*this, staging, sem, {65535, 0}, {0, 0},
                                                                              {65535, 65535});
    case VK_FORMAT_R8G8_UNORM:
      return textureFillUninitializedPatternInternal<std::array<uint8_t, 2>>(*this, staging, sem, {255, 0}, {0, 0}, {255, 255});
    case VK_FORMAT_R16_UNORM:
      return textureFillUninitializedPatternInternal<uint16_t>(*this, staging, sem, 49152, 0, 65535);
    case VK_FORMAT_R8_UNORM:
      return textureFillUninitializedPatternInternal<uint8_t>(*this, staging, sem, 192, 0, 255);
    default:
      LOGD("Texture::fillUninitializedPattern: Not implemented for VkFormat %s.\n", string_VkFormat(format));
      return VK_ERROR_UNKNOWN;
  }
}

void destroyResources(nvvk::ResourceAllocator& alloc, UniformBuffer& ubc)
{
  alloc.destroyBuffer(ubc.buffer);
  ubc = {};
}

void UniformBuffer::addTransitionTo(std::vector<VkBufferMemoryBarrier2>& barriers, VkPipelineStageFlags2 newStages, bool mayWrite)
{
  if(currentStages == newStages && !maybeWritten && !mayWrite)
  {
    return;
  }

  VkBufferMemoryBarrier2 barrier = nvvk::makeBufferMemoryBarrier(
      nvvk::BufferMemoryBarrierParams{.buffer = buffer.buffer, .srcStageMask = currentStages, .dstStageMask = newStages});
  barriers.push_back(std::move(barrier));

  currentStages = newStages;
  maybeWritten  = mayWrite;
}

void destroyResources(nvvk::ResourceAllocator& alloc, StorageBuffer& buf)
{
  alloc.destroyBuffer(buf.buffer);
  buf = {};
}

VkResult StorageBuffer::resize(VkCommandBuffer cmd, size_t newByteSize, nvvk::ResourceAllocator& alloc, nvapp::Application& dealloc)
{
  if(buffer.bufferSize == newByteSize)
  {
    return VK_SUCCESS;  // Nothing to do
  }

  if(buffer.buffer)
  {
    dealloc.submitResourceFree([&alloc, buffer = buffer]() mutable { alloc.destroyBuffer(buffer); });
    buffer = {};
  }

  // TODO: Determine proper usage flags
  NVVK_FAIL_RETURN(alloc.createBuffer(buffer, newByteSize, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT));
  currentStages = {};

  return VK_SUCCESS;
}

void StorageBuffer::addTransitionTo(std::vector<VkBufferMemoryBarrier2>& barriers, VkPipelineStageFlags2 newStages, bool mayWrite)
{
  if(currentStages == newStages && !maybeWritten && !mayWrite)
  {
    return;
  }

  VkBufferMemoryBarrier2 barrier = nvvk::makeBufferMemoryBarrier(
      nvvk::BufferMemoryBarrierParams{.buffer = buffer.buffer, .srcStageMask = currentStages, .dstStageMask = newStages});
  barriers.push_back(std::move(barrier));

  currentStages = newStages;
  maybeWritten  = mayWrite;
}

void Resources::updatePipelineStats(VkDevice device)
{
  if(loadedPipelineStats)
  {
    return;
  }

  for(Pass& pass : passes)
  {
    const VkPipelineInfoKHR pipelineInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR, .pipeline = pass.pipeline};
    uint32_t                numPipelineExecutables = 0;
    NVVK_CHECK(vkGetPipelineExecutablePropertiesKHR(device, &pipelineInfo, &numPipelineExecutables, nullptr));
    pass.pipelineProperties.resize(numPipelineExecutables, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
    pass.pipelineStatistics.resize(numPipelineExecutables);
    pass.pipelineRepresentations.resize(numPipelineExecutables);
    NVVK_CHECK(vkGetPipelineExecutablePropertiesKHR(device, &pipelineInfo, &numPipelineExecutables,
                                                    pass.pipelineProperties.data()));

    for(uint32_t exeIdx = 0; exeIdx < numPipelineExecutables; exeIdx++)
    {
      uint32_t                          numStatistics = 0;
      const VkPipelineExecutableInfoKHR exeInfo{
          .sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR, .pipeline = pass.pipeline, .executableIndex = exeIdx};
      auto& statistics = pass.pipelineStatistics[exeIdx];
      NVVK_CHECK(vkGetPipelineExecutableStatisticsKHR(device, &exeInfo, &numStatistics, nullptr));
      statistics.resize(numStatistics, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
      NVVK_CHECK(vkGetPipelineExecutableStatisticsKHR(device, &exeInfo, &numStatistics, statistics.data()));

      uint32_t numRepresentations = 0;
      auto&    representations    = pass.pipelineRepresentations[exeIdx];
      NVVK_CHECK(vkGetPipelineExecutableInternalRepresentationsKHR(device, &exeInfo, &numRepresentations, nullptr));
      representations.resize(numRepresentations, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR});
      hadAnyPipelineRepresentations |= (numRepresentations > 0);
      NVVK_CHECK(vkGetPipelineExecutableInternalRepresentationsKHR(device, &exeInfo, &numRepresentations,
                                                                   representations.data()));
      // Now allocate space for each of the data blocks.
      for(uint32_t repr = 0; repr < numRepresentations; repr++)
      {
        std::vector<char> data(representations[repr].dataSize);
        representations[repr].pData = data.data();
        pass.pipelineRepresentationData.push_back(std::move(data));
      }
      // And finally fill them:
      NVVK_CHECK(vkGetPipelineExecutableInternalRepresentationsKHR(device, &exeInfo, &numRepresentations,
                                                                   representations.data()));
    }
  }

  loadedPipelineStats = true;
}

void destroyResources(nvvk::ResourceAllocator& alloc, std::shared_ptr<Resources> resources)
{
  if(!resources)
  {
    return;
  }

  VkDevice device = alloc.getDevice();

  for(size_t i = 0; i < resources->passes.size(); i++)
  {
    vkDestroyPipeline(device, resources->passes[i].pipeline, nullptr);
  }

  vkDestroyShaderModule(device, resources->shaderModule, nullptr);

  vkDestroyPipelineLayout(device, resources->pipelineLayout, nullptr);
  for(size_t set = 0; set < resources->descriptorSetLayouts.size(); set++)
  {
    vkDestroyDescriptorSetLayout(device, resources->descriptorSetLayouts[set], nullptr);
  }
  vkDestroyDescriptorPool(device, resources->descriptorPool, nullptr);

  for(size_t i = 0; i < resources->storageBuffers.size(); i++)
  {
    destroyResources(alloc, resources->storageBuffers[i]);
  }

  for(size_t i = 0; i < resources->textures.size(); i++)
  {
    resources->textures[i].deinitResources(alloc);
  }

  for(size_t i = 0; i < resources->uniformBuffers.size(); i++)
  {
    destroyResources(alloc, resources->uniformBuffers[i]);
  }
}
