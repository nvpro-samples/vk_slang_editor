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

// Simple nvvk::GBuffer-like resizable texture class, plus image I/O functions.

#include <nvutils/hash_operations.hpp>
#include <nvvk/resources.hpp>
#include <nvvk/semaphore.hpp>
#include <slang.h>

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace nvapp {
class Application;
}

namespace nvvk {
class ResourceAllocator;
class StagingUploader;
}  // namespace nvvk

// Where the data for a uniform or resource comes from.
enum class Source
{
  eUnknown,      // Unknown uniforms become sliders; unknown textures are screen-sized.
  eResolution,   // in pixels, e.g. (float|uint)(2|3) iResolution (.z reserved for VR). Bonzomatic uses v2Resolution
  eTime,         // Time since playback start, e.g. float iTime. Bonzomatic uses fFrameTime
  eView,         // View matrix
  eViewInverse,  // Inverse of view matrix
  eProj,         // Projection matrix
  eProjInverse,
  eProjView,
  eProjViewInverse,
  eEye,                   // float3 camera position
  eFragCoordToDirection,  // mul(float3(threadIdx,1), this) == ray direction; see code for implementation
  eFrameIndex,            // Number of frames since start, e.g. int iFrame;
  eMouse,                 // .xy = mouse position, in pixels, .z = LMB, .w = RMB
  eTexFrame,              // Screen-sized texture; our default output.
  eTexDepth,              // Also screen-sized.
  eTexFile,               // Texture loaded from a file.

  // TODO: texFFT, texFFTSmoothed, texFFTIntegrated, texDepth, ...
  // TODO: Add Back/Previous versions for the previous frame.
  // Shadertoy also provides: iTimeDelta, iFrameRate, iChannelTime, iChannelResolution, iMouse, iDate
  // TODO: also provide Back/Previous versions for the previous frame.
};

struct UniformWrite
{
  std::string name;
  Source      source      = Source::eUnknown;
  size_t      bufferIndex = 0;  // Which buffer will this be written to?
  size_t      byteOffset  = 0;  // At which byte offset?
  // In case someone requests both float iTime and double iTime, or
  // float3x4 modelToWorld:
  SlangScalarType scalarType = SLANG_SCALAR_TYPE_NONE;
  unsigned        rows       = 1;
  unsigned        cols       = 1;
  uint32_t        byteSize() const;
};

struct Texture
{
  std::string     name;
  Source          source    = Source::eUnknown;
  VkFormat        format    = VK_FORMAT_UNDEFINED;
  VkImageViewType dimension = VK_IMAGE_VIEW_TYPE_2D;
  // All textures at least have these 3 flags because we read and write them,
  // and because we might display them in ImGui.
  VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  VkPipelineStageFlags2 currentStages{};
  bool                  maybeWritten = false;

  // These are created by `resize`:
  nvvk::Image     image;                             // Image, image view, sampler, layout
  VkImageView     uiImageView     = VK_NULL_HANDLE;  // RGB1 for ImGui
  VkDescriptorSet uiDescriptorSet = VK_NULL_HANDLE;  // For ImGui
  VkExtent3D      size{0, 0, 1};

  [[nodiscard]] VkDescriptorSet getImguiID() const { return uiDescriptorSet; }

  // The old texture will be defer-destroyed.
  VkResult resize(VkCommandBuffer          cmd,
                  VkExtent3D               newSize,
                  uint32_t                 numMips,
                  uint32_t                 numVkLayers, // layers * faces
                  VkSampler                uiSampler,
                  nvvk::ResourceAllocator& alloc,
                  nvapp::Application&      dealloc);
  // Immediately destroys the texture.
  void deinitResources(nvvk::ResourceAllocator& alloc);
  // `image.descriptor.imageLayout` and `currentStages` will be updated.
  // newStages is allowed to be nvvk::INFER_BARRIER_PARAMS.
  void addTransitionTo(std::vector<VkImageMemoryBarrier2>& barriers, VkImageLayout newLayout, VkPipelineStageFlags2 newStages, bool mayWrite);
  // Fills all mips of the texture with a repeating pattern that says "UNINIT".
  // Note that a transfer will only be queued in the staging uploader.
  // This will automatically transition the layout to DST_OPTIMAL for you.
  // This is so that it's evident if you're trying to render from a texture that's uninitialized-
  // even if you're visualizing only .r, .g, .b, or .a.
  VkResult fillUninitializedPattern(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const nvvk::SemaphoreState& semaphoreState);
};

// Whichever texture's being displayed by ImGui always needs to have this layout:
constexpr VkImageLayout guiImageLayout = VK_IMAGE_LAYOUT_GENERAL;

struct DescriptorIndex
{
  uint32_t binding = 0;
  uint32_t set     = 0;

  bool operator==(const DescriptorIndex& other) const { return binding == other.binding && set == other.set; }
};

template <>
struct std::hash<DescriptorIndex>
{
  std::size_t operator()(const DescriptorIndex& v) const { return nvutils::hashVal(v.binding, v.set); }
};

constexpr size_t UNSET_SIZET = ~size_t(0);

struct DescriptorWrite
{
  size_t             resourceIndex = UNSET_SIZET;  // Depends on the descriptor type
  DescriptorIndex    index;
  VkDescriptorType   descriptorType = VK_DESCRIPTOR_TYPE_MAX_ENUM;
  VkImageLayout      layout         = VK_IMAGE_LAYOUT_UNDEFINED;
  VkShaderStageFlags stages         = 0;
};

static_assert(CHAR_BIT == 8, "Bytes must be 8 bits long, please!");
struct UniformBuffer
{
  std::string           name;
  std::vector<char>     cpuData;  // Staging area for data so we can write it all at once
  nvvk::Buffer          buffer;
  VkPipelineStageFlags2 currentStages{};
  bool                  maybeWritten = false;

  void addTransitionTo(std::vector<VkBufferMemoryBarrier2>& barriers, VkPipelineStageFlags2 newStages, bool mayWrite);
};

struct StorageBuffer
{
  std::string           name;
  Source                source = Source::eUnknown;
  nvvk::Buffer          buffer;
  size_t                elementStride = 0;  // In bytes; 0 = unknown
  VkPipelineStageFlags2 currentStages{};
  bool                  maybeWritten = false;

  // Makes the buffer have exactly a certain size, creating it if it didn't exist.
  // The previous buffer (if it existed) will be defer-destroyed.
  VkResult resize(VkCommandBuffer cmd, size_t newByteSize, nvvk::ResourceAllocator& alloc, nvapp::Application& dealloc);
  void addTransitionTo(std::vector<VkBufferMemoryBarrier2>& barriers, VkPipelineStageFlags2 newStages, bool mayWrite);
};

// Vertex attributes we know about. These are the members of nvvkgltf::SceneVk::VertexBuffers.
enum class VertexAttribute
{
  eUnknown,    //
  ePosition,   // float3
  eNormal,     // float3
  eTangent,    // float4
  eTexCoord0,  // float2
  eTexCoord1,  // float2
  eColor,      // unorm4x8
};

struct VtxAttribInfo
{
  uint32_t        binding   = 0;
  VertexAttribute attribute = VertexAttribute::eUnknown;
};

struct Pass
{
  VkShaderStageFlags       shaderStages;
  VkPipeline               pipeline;
  std::array<SlangUInt, 3> workgroupSize{1, 1, 1};
  struct UsedResource
  {
    size_t                resourceIndex = 0;                          // We use this resource index...
    VkPipelineStageFlags2 stages        = 0;                          // from these stages
    VkImageLayout         layout        = VK_IMAGE_LAYOUT_UNDEFINED;  // With this layout (for images)
  };
  std::vector<UsedResource>                   usedStorageBuffers;
  std::vector<UsedResource>                   usedTextures;
  std::vector<UsedResource>                   usedUniformBuffers;
  std::unordered_map<unsigned, VtxAttribInfo> vtxAttribInfos;
  // Generated from vtxAttribInfos:
  std::vector<VkVertexInputAttributeDescription2EXT> vertexAttributeDescriptions;
  std::vector<VkVertexInputBindingDescription2EXT>   vertexBindingDescriptions;
  // For the UI:
  std::string                                                             debugName;
  std::vector<VkPipelineExecutablePropertiesKHR>                          pipelineProperties;
  std::vector<std::vector<VkPipelineExecutableStatisticKHR>>              pipelineStatistics;
  std::vector<std::vector<VkPipelineExecutableInternalRepresentationKHR>> pipelineRepresentations;
  std::vector<std::vector<char>> pipelineRepresentationData;  // memory for pipelineRepresentations
};

// Dynamic resources for renderer; these are created based off of the shader.
struct Resources
{
  std::vector<UniformWrite>    uniformUpdates;
  std::vector<DescriptorWrite> descriptorSetUpdates;

  std::vector<StorageBuffer> storageBuffers;

  std::vector<Texture> textures;
  VkSampler            sampler = VK_NULL_HANDLE;

  std::vector<UniformBuffer> uniformBuffers;  // [buffer index]

  std::vector<std::vector<VkDescriptorSet>> descriptorSets;        // [frame cycle][descriptor set]
  std::vector<VkDescriptorSetLayout>        descriptorSetLayouts;  // [set]
  VkPipelineLayout                          pipelineLayout = VK_NULL_HANDLE;
  VkDescriptorPool                          descriptorPool = VK_NULL_HANDLE;

  VkShaderModule shaderModule = VK_NULL_HANDLE;

  std::vector<Pass> passes;

  // Which texture is texFrame, the one we should display by default?
  // If texFrameIndex == UNSET_SIZET, nothing's texFrame.
  size_t texFrameIndex = UNSET_SIZET;
  size_t texDepthIndex = UNSET_SIZET;
  // Does the shader use camera parameters at all?
  bool hasCameraUniform = false;
  // Have we tried to load pipeline statistics and representations yet?
  bool loadedPipelineStats = false;
  // Did we get any pipeline representations from the driver?
  bool hadAnyPipelineRepresentations = false;
  // Tries to load pipeline statistics and representations if not yet queried.
  void updatePipelineStats(VkDevice device);
};

// Consumes and destroys a Resources object.
// This takes a shared_ptr to make submitResourceFree() work.
void destroyResources(nvvk::ResourceAllocator& alloc, std::shared_ptr<Resources> resources);
