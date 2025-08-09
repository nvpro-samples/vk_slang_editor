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

#define VMA_IMPLEMENTATION

#include "sample.h"
#include "utilities.h"

#include <nvgui/window.hpp>
#include <nvutils/alignment.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/formats.hpp>
#include <nvvk/resources.hpp>

#include <glm/detail/type_half.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <bit>
#include <limits>
#include <set>
#include <unordered_map>

constexpr VkShaderStageFlags kAllRasterGraphicsStages = VK_SHADER_STAGE_ALL_GRAPHICS;
constexpr VkShaderStageFlags kAllRasterMeshStages =
    VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT;
constexpr VkShaderStageFlags kAllRasterStages     = kAllRasterGraphicsStages | kAllRasterMeshStages;
constexpr VkShaderStageFlags kAllComputeStages    = VK_SHADER_STAGE_COMPUTE_BIT;
constexpr VkShaderStageFlags kAllRayTracingStages = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR
                                                    | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR
                                                    | VK_SHADER_STAGE_CALLABLE_BIT_KHR;

static const std::vector<VkDynamicState> kDynamicStates{VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT,  //
                                                        VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT,   //
                                                        //
                                                        VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,   //
                                                        VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,  //
                                                        VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,    //
                                                        VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE,
                                                        //
                                                        VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,   //
                                                        VK_DYNAMIC_STATE_STENCIL_OP,            //
                                                        VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,  //
                                                        VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,    //
                                                        VK_DYNAMIC_STATE_STENCIL_REFERENCE,
                                                        //
                                                        VK_DYNAMIC_STATE_CULL_MODE,   //
                                                        VK_DYNAMIC_STATE_FRONT_FACE,  //
                                                        //
                                                        VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE,  //
                                                        VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE,   //
                                                        VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY,         //
                                                        // VK_DYNAMIC_STATE_LINE_WIDTH,
                                                        // VK_DYNAMIC_STATE_LINE_STIPPLE,
                                                        //
                                                        VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT,    //
                                                        VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT,  //
                                                        VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT};

static bool isRasterGraphicsFlags(VkShaderStageFlags stageFlags)
{
  // "It doesn't have any stages other than those in kAllRasterGraphicsStages"
  return (stageFlags & kAllRasterGraphicsStages) == stageFlags;
}

static uint32_t slangScalarTypeBitSize(SlangScalarType stype)
{
  switch(stype)
  {
    case SLANG_SCALAR_TYPE_NONE:
    case SLANG_SCALAR_TYPE_VOID:
    default:
      assert(false);
      return 0;
    case SLANG_SCALAR_TYPE_FLOAT16:
    case SLANG_SCALAR_TYPE_INT16:
    case SLANG_SCALAR_TYPE_UINT16:
      return 16;
    case SLANG_SCALAR_TYPE_BOOL:  // Yes, bools are 32 bits
    case SLANG_SCALAR_TYPE_INT32:
    case SLANG_SCALAR_TYPE_UINT32:
    case SLANG_SCALAR_TYPE_FLOAT32:
      return 32;
    case SLANG_SCALAR_TYPE_INT64:
    case SLANG_SCALAR_TYPE_UINT64:
    case SLANG_SCALAR_TYPE_FLOAT64:
    case SLANG_SCALAR_TYPE_INTPTR:
    case SLANG_SCALAR_TYPE_UINTPTR:
      return 64;
  }
}

uint32_t UniformWrite::byteSize() const
{
  return (slangScalarTypeBitSize(scalarType) * rows * cols + 7) / 8;
}

struct half
{
  uint16_t data = 0;
  half() {}
  explicit half(float f) { data = glm::detail::toFloat16(f); }
  explicit operator float() const { return glm::detail::toFloat32(data); }
  explicit half(bool b) { data = (b ? 0x3c00u : 0); }
  explicit half(int32_t i)
      : half(static_cast<float>(i))
  {
  }
  explicit half(uint32_t i)
      : half(static_cast<float>(i))
  {
  }
  explicit half(int64_t i)
      : half(static_cast<float>(i))
  {
  }
  explicit half(uint64_t i)
      : half(static_cast<float>(i))
  {
  }
  explicit half(double f)
      : half(static_cast<float>(f))
  {
  }
  explicit half(int8_t i)
      : half(static_cast<float>(i))
  {
  }
  explicit half(uint8_t i)
      : half(static_cast<float>(i))
  {
  }
  explicit half(int16_t i)
      : half(static_cast<float>(i))
  {
  }
  explicit half(uint16_t i)
      : half(static_cast<float>(i))
  {
  }
};

// Copies min(dstNumElements, srcNumElements) from src to dst with conversion
// from T to dstType.
template <class To, class From>
static void convertCopyTo(To* dst, const size_t dstNumElements, const From* src, const size_t srcNumElements)
{
  const size_t copyNumElements = std::min(dstNumElements, srcNumElements);
  for(size_t i = 0; i < copyNumElements; i++)
  {
    const To to = static_cast<To>(src[i]);
    memcpy(&dst[i], &to, sizeof(To));
  }
}

// Same as convertCopyTo but the type parameter is a Slang type instead.
template <class From>
static void convertCopyTo(void* dst, SlangScalarType dstType, const size_t dstNumElements, const From* src, const size_t srcNumElements)
{
  switch(dstType)
  {
    case SLANG_SCALAR_TYPE_INT32:
      convertCopyTo(reinterpret_cast<int32_t*>(dst), dstNumElements, src, srcNumElements);
      break;
    case SLANG_SCALAR_TYPE_BOOL:  // Since uniform bools are 32 bits in Slang
    case SLANG_SCALAR_TYPE_UINT32:
      convertCopyTo(reinterpret_cast<uint32_t*>(dst), dstNumElements, src, srcNumElements);
      break;
    case SLANG_SCALAR_TYPE_INT64:
      convertCopyTo(reinterpret_cast<int64_t*>(dst), dstNumElements, src, srcNumElements);
      break;
    case SLANG_SCALAR_TYPE_UINT64:
      convertCopyTo(reinterpret_cast<uint64_t*>(dst), dstNumElements, src, srcNumElements);
      break;
    case SLANG_SCALAR_TYPE_FLOAT16:
      convertCopyTo(reinterpret_cast<half*>(dst), dstNumElements, src, srcNumElements);
      break;
    case SLANG_SCALAR_TYPE_FLOAT32:
      convertCopyTo(reinterpret_cast<float*>(dst), dstNumElements, src, srcNumElements);
      break;
    case SLANG_SCALAR_TYPE_FLOAT64:
      convertCopyTo(reinterpret_cast<double*>(dst), dstNumElements, src, srcNumElements);
      break;
    case SLANG_SCALAR_TYPE_INT8:
      convertCopyTo(reinterpret_cast<int8_t*>(dst), dstNumElements, src, srcNumElements);
      break;
    case SLANG_SCALAR_TYPE_UINT8:
      convertCopyTo(reinterpret_cast<uint8_t*>(dst), dstNumElements, src, srcNumElements);
      break;
    case SLANG_SCALAR_TYPE_INT16:
      convertCopyTo(reinterpret_cast<int16_t*>(dst), dstNumElements, src, srcNumElements);
      break;
    case SLANG_SCALAR_TYPE_UINT16:
      convertCopyTo(reinterpret_cast<uint16_t*>(dst), dstNumElements, src, srcNumElements);
      break;
    default:
      // TODO: Bool vectors
      assert(!"Unimplemented");
  }
}

template <class From>
static void bitCopyTo(void* dst, SlangScalarType dstType, const size_t dstNumElements, const From* src, const size_t srcNumElements)
{
  const size_t dstNumBytes  = ((static_cast<size_t>(slangScalarTypeBitSize(dstType)) + 7) / 8) * dstNumElements;
  const size_t srcNumBytes  = sizeof(From) * srcNumElements;
  const size_t copyNumBytes = std::min(dstNumBytes, srcNumBytes);
  memcpy(dst, src, copyNumBytes);
}


nvvk::GraphicsPipelineState& Sample::getOrCreateGraphicsPipelineState(size_t passIndex)
{
  if(passIndex >= m_graphicsPipelineStates.size())
  {
    nvvk::GraphicsPipelineState newState;
    // Be kind to developers whose models have inconsistent winding
    newState.rasterizationState.cullMode = VK_CULL_MODE_NONE;
    // Default to premultiplied alpha blending:
    newState.colorBlendEnables   = {VK_TRUE};
    newState.colorBlendEquations = {VkColorBlendEquationEXT{.srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
                                                            .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                                                            .colorBlendOp        = VK_BLEND_OP_ADD,
                                                            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
                                                            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                                                            .alphaBlendOp        = VK_BLEND_OP_ADD}};
    m_graphicsPipelineStates.resize(passIndex + 1, std::move(newState));
  }

  nvvk::GraphicsPipelineState& result = m_graphicsPipelineStates[passIndex];

  // If we have a tessellation control shader, then we need to input
  // VK_PRIMITIVE_TOPOLOGY_PATCH_LIST instead of VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  // this is forced even if the state already existed.
  if(m_resources->passes[passIndex].shaderStages & VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT)
  {
    result.inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
  }
  else if(result.inputAssemblyState.topology == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST)
  {
    result.inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  }

  return result;
}


// Handle resources that need to be loaded or resized.
void Sample::syncResources(VkCommandBuffer cmd)
{
  // Note that this should not contain significant work that needs to be done
  // every frame, as it's not included in the onRender() profile!
  NVVK_DBG_SCOPE(cmd);

  const VkExtent2D resolution = m_app->getViewportSize();
  // If you need to upload things, use this semaphoreState.
  nvvk::SemaphoreState semaphoreState = nvvk::SemaphoreState::makeFixed(m_staging.timelineSemaphore, m_staging.timelineValue);

  for(size_t i = 0; i < m_resources->storageBuffers.size(); i++)
  {
    StorageBuffer& storageBuffer = m_resources->storageBuffers[i];
    if(storageBuffer.source != Source::eUnknown)
    {
      continue;
    }

    StorageBufferParameters& params       = m_shaderParams.storageBuffers[storageBuffer.name];
    const VkDeviceSize       requiredSize = params.computeBufferSize(resolution, storageBuffer.elementStride);

    if(requiredSize > m_physicalDeviceProperties.limits.maxStorageBufferRange)
    {
      continue;
    }

    // We sort of arbitrarily choose to avoid re-allocating buffer data if the
    // buffer becomes smaller. Maybe preserving data if someone inputs a smaller
    // buffer size and then changes their mind is nice.
    if(storageBuffer.buffer.bufferSize < requiredSize)
    {
      NVVK_FAIL_REPORT(storageBuffer.resize(cmd, requiredSize, m_alloc, *m_app));
    }
  }

  // Normally this would be in onResize(), but we can consolidate it here.
  for(size_t i = 0; i < m_resources->textures.size(); i++)
  {
    Texture& tex = m_resources->textures[i];
    if(Source::eTexFile != tex.source && (tex.size.width != resolution.width || tex.size.height != resolution.height))
    {
      NVVK_FAIL_REPORT(tex.resize(cmd, {resolution.width, resolution.height, tex.size.depth}, 1, 1,
                                  m_resources->sampler, m_alloc, *m_app));
      if(Source::eTexDepth != tex.source)
      {
        tex.fillUninitializedPattern(cmd, m_staging.uploader, semaphoreState);
      }
    }
  }

  if(!m_staging.uploader.isAppendedEmpty())
  {
    // Upload new textures; don't run anything until they've finished.
    m_staging.uploader.cmdUploadAppended(cmd);
    const VkSemaphoreSubmitInfo semaphoreSignalInfo{
        .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .semaphore = m_staging.timelineSemaphore,
        .value     = m_staging.timelineValue,
        .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
    };
    m_app->addSignalSemaphore(semaphoreSignalInfo);
  }
}

void Sample::onRender(VkCommandBuffer cmd)
{
  m_staging.uploader.releaseStaging();
  m_profilerTimeline->frameAdvance();
  syncResources(cmd);

  NVVK_DBG_SCOPE(cmd);
  auto profilerRangeRender = m_profilerGPU.cmdFrameSection(cmd, __FUNCTION__);

  const double deltaTime = m_frameTimer.getSeconds();
  m_frameTimer.reset();
  if(!(m_shaderParams.paused || m_temporarilyPaused))
  {
    m_shaderParams.time += m_shaderParams.timeSpeed * deltaTime;
  }

  // Exit early if there's nothing to do
  if(m_resources->passes.empty())
  {
    return;
  }

  // Or if any of the passes didn't generate correctly
  for(const Pass& pass : m_resources->passes)
  {
    if(!pass.pipeline)
    {
      return;
    }
  }

  const uint32_t   frameCycle = m_app->getFrameCycleIndex();
  const VkExtent2D resolution = m_app->getViewportSize();

  // Update uniform buffers.
  {
    auto profilerRangeUpdateUniforms = m_profilerGPU.cmdFrameSection(cmd, "Update Uniforms");

    // Update uniform buffers.
    // TODO (nbickford): See how much closer we can get this to memcpy speed
    // without needing JIT
    std::set<size_t> writtenUniformBuffers;
    for(size_t i = 0; i < m_resources->uniformUpdates.size(); i++)
    {
      const UniformWrite& write = m_resources->uniformUpdates[i];

      std::vector<char>& cpuBuffer = m_resources->uniformBuffers[write.bufferIndex].cpuData;
      assert(write.byteOffset + write.byteSize() <= cpuBuffer.size());
      char*        writeTo          = cpuBuffer.data() + write.byteOffset;
      const size_t writeNumElements = static_cast<size_t>(write.rows) * write.cols;

      switch(write.source)
      {
        case Source::eResolution: {
          convertCopyTo(writeTo, write.scalarType, writeNumElements, reinterpret_cast<const uint32_t*>(&resolution), 2);
          break;
        }
        case Source::eTime: {
          convertCopyTo(writeTo, write.scalarType, writeNumElements, &m_shaderParams.time, 1);
          break;
        }
        case Source::eView: {
          const glm::mat4& view = m_cameraControl->getViewMatrix();
          convertCopyTo(writeTo, write.scalarType, writeNumElements, glm::value_ptr(view), 16);
          break;
        }
        case Source::eViewInverse: {
          const glm::mat4 viewInverse = glm::inverse(m_cameraControl->getViewMatrix());
          convertCopyTo(writeTo, write.scalarType, writeNumElements, glm::value_ptr(viewInverse), 16);
          break;
        }
        case Source::eProj: {
          const glm::mat4& proj = m_cameraControl->getPerspectiveMatrix();
          convertCopyTo(writeTo, write.scalarType, writeNumElements, glm::value_ptr(proj), 16);
          break;
        }
        case Source::eProjInverse: {
          const glm::mat4 projInverse = glm::inverse(m_cameraControl->getPerspectiveMatrix());
          convertCopyTo(writeTo, write.scalarType, writeNumElements, glm::value_ptr(projInverse), 16);
          break;
        }
        case Source::eProjView: {
          const glm::mat4 projView = m_cameraControl->getPerspectiveMatrix() * m_cameraControl->getViewMatrix();
          convertCopyTo(writeTo, write.scalarType, writeNumElements, glm::value_ptr(projView), 16);
          break;
        }
        case Source::eProjViewInverse: {
          const glm::mat4 projView        = m_cameraControl->getPerspectiveMatrix() * m_cameraControl->getViewMatrix();
          const glm::mat4 projViewInverse = glm::inverse(projView);
          convertCopyTo(writeTo, write.scalarType, writeNumElements, glm::value_ptr(projViewInverse), 16);
          break;
        }
        case Source::eEye: {
          const glm::vec3 eye = m_cameraControl->getEye();
          convertCopyTo(writeTo, write.scalarType, writeNumElements, glm::value_ptr(eye), 3);
          break;
        }
        case Source::eFragCoordToDirection: {
          const glm::mat4 view = m_cameraControl->getViewMatrix();
          const glm::mat4 proj = m_cameraControl->getPerspectiveMatrix();
          // The following code depends on the projection matrix not being a
          // skew matrix, so double-check that:
          assert(0.0f == proj[0][1] && 0.0f == proj[0][2] && 0.0f == proj[0][3]     //
                 && 0.0f == proj[1][0] && 0.0f == proj[1][2] && 0.0f == proj[1][3]  //
                 && 0.0f == proj[2][0] && 0.0f == proj[2][1]                        //
                 && 0.0f == proj[3][0] && 0.0f == proj[3][1]);
          // Concatenate the following operations:
          // const float2 clipXY = thread * (2 / iResolution) - 1; (fragcoord to clip space)
          // clipXY = mul(float4(clipXY,0,0), inverse(proj)).xy; (undo projection FOV)
          // z = -1; (GLM uses -z for forward)
          glm::mat3x3 mat{};
          mat[0][0] = 2.f / (static_cast<float>(resolution.width) * proj[0][0]);
          mat[1][1] = 2.f / (static_cast<float>(resolution.height) * proj[1][1]);
          mat[2][0] = -1.f / proj[0][0];
          mat[2][1] = -1.f / proj[1][1];
          mat[2][2] = -1.f;
          // Undo the view rotation.
          // This truncates view to just its 3x3 rotational part, then transposes
          // it (since we know that's the same as the transpose):
          glm::mat3 invViewRot = glm::transpose(glm::mat3(view));
          mat                  = invViewRot * mat;
          convertCopyTo(writeTo, write.scalarType, writeNumElements, glm::value_ptr(mat), 9);
          break;
        }
        case Source::eFrameIndex: {
          convertCopyTo(writeTo, write.scalarType, writeNumElements, &m_frame, 1);
          break;
        }
        case Source::eMouse: {
          const ImVec2 pos = ImGui::GetMousePos();
          glm::vec4    result(pos.x, pos.y, 0.0f, 0.0f);
          ImGuiWindow* viewport = ImGui::FindWindowByName("Viewport");
          if(viewport)
          {
            result.x -= viewport->Pos.x;
            result.y -= viewport->Pos.y;

            // Only react to mouse clicks if we're inside the viewport
            if(nvgui::isWindowHovered(viewport))
            {
              result.z = ImGui::IsMouseDown(ImGuiMouseButton_Left) ? 1.0f : 0.0f;
              result.w = ImGui::IsMouseDown(ImGuiMouseButton_Right) ? 1.0f : 0.0f;
            }
          }
          convertCopyTo(writeTo, write.scalarType, writeNumElements, glm::value_ptr(result), 4);
          break;
        }
        case Source::eUnknown: {
          // Does it exist in our uniforms?
          const auto& it = m_shaderParams.uniforms.find(write.name);
          if(it != m_shaderParams.uniforms.end())
          {
            // Write its bytes:
            bitCopyTo(writeTo, write.scalarType, writeNumElements, it->second.data(), it->second.size());
          }
          else
          {
            // Write zeros:
            std::array<uint32_t, 16> zeros{};
            bitCopyTo(writeTo, write.scalarType, writeNumElements, zeros.data(), zeros.size());
          }
          break;
        }
        default:
          LOGW("Unhandled source: %s\n", write.name.c_str());
          continue;
      }

      writtenUniformBuffers.emplace(write.bufferIndex);
    }

    // Transition changed buffers so they can be written
    nvvk::BarrierContainer barriers;
    for(size_t bufferIdx : writtenUniformBuffers)
    {
      UniformBuffer& buf = m_resources->uniformBuffers[bufferIdx];
      buf.addTransitionTo(barriers.bufferBarriers, VK_PIPELINE_STAGE_2_TRANSFER_BIT, true);
    }
    barriers.cmdPipelineBarrier(cmd, 0);

    // And record updates into the command buffer:
    for(size_t bufferIdx : writtenUniformBuffers)
    {
      UniformBuffer& ubo = m_resources->uniformBuffers[bufferIdx];
      vkCmdUpdateBuffer(cmd, ubo.buffer.buffer, 0, ubo.cpuData.size(), ubo.cpuData.data());
    }
  }

  // Queue descriptor updates
  const auto& frameDescriptorSets = m_resources->descriptorSets[frameCycle];
  {
    auto profilerRangeUpdateDescriptors = m_profilerGPU.cmdFrameSection(cmd, "Update Descriptors");
    // TODO: Texture ping-pong support (at the moment updating textures
    // constantly is redundant)
    {
      std::vector<VkWriteDescriptorSet> updates(m_resources->descriptorSetUpdates.size());
      // FIXME: This over-allocates for now; ideally we'd count the exact number
      std::vector<VkDescriptorBufferInfo> bufferInfos(m_resources->descriptorSetUpdates.size());
      std::vector<VkDescriptorImageInfo>  imageInfos(m_resources->descriptorSetUpdates.size());
      for(size_t i = 0; i < m_resources->descriptorSetUpdates.size(); i++)
      {
        const DescriptorWrite& update = m_resources->descriptorSetUpdates[i];
        VkWriteDescriptorSet   write{.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                     .dstSet          = frameDescriptorSets[update.index.set],
                                     .dstBinding      = update.index.binding,
                                     .dstArrayElement = 0,
                                     .descriptorCount = 1,
                                     .descriptorType  = update.descriptorType};
        switch(update.descriptorType)
        {
          case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
          case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
          case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            imageInfos[i] =
                VkDescriptorImageInfo{.sampler = m_resources->sampler,
                                      .imageView = m_resources->textures[update.resourceIndex].image.descriptor.imageView,
                                      .imageLayout = update.layout};
            write.pImageInfo = &imageInfos[i];
            break;
          case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            bufferInfos[i] = VkDescriptorBufferInfo{.buffer = m_resources->uniformBuffers[update.resourceIndex].buffer.buffer,
                                                    .offset = 0,
                                                    .range  = VK_WHOLE_SIZE};
            write.pBufferInfo = &bufferInfos[i];
            break;
          case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            bufferInfos[i] = VkDescriptorBufferInfo{.buffer = m_resources->storageBuffers[update.resourceIndex].buffer.buffer,
                                                    .offset = 0,
                                                    .range  = VK_WHOLE_SIZE};
            write.pBufferInfo = &bufferInfos[i];
            break;
          default:
            assert(!"Unhandled descriptor type!");
            continue;
        }
        updates[i] = write;
      }
      vkUpdateDescriptorSets(m_ctx.getDevice(), static_cast<uint32_t>(updates.size()), updates.data(), 0, nullptr);
    }
  }

  // Clear texFrame (even if it's not being visualized) and the main depth buffer.
  if(UNSET_SIZET != m_resources->texFrameIndex && m_shaderParams.clearColorWhen == ClearWhen::eAlways)
  {
    auto                   profilerRangeClear = m_profilerGPU.cmdFrameSection(cmd, "ClearColor");
    Texture&               tex                = m_resources->textures[m_resources->texFrameIndex];
    nvvk::BarrierContainer barriers;
    tex.addTransitionTo(barriers.imageBarriers, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, nvvk::INFER_BARRIER_PARAMS, true);
    barriers.cmdPipelineBarrier(cmd, 0);

    VkClearColorValue color{};
    memcpy(color.float32, m_shaderParams.clearColor.data(), sizeof(float) * m_shaderParams.clearColor.size());
    const VkImageSubresourceRange range{.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                                        .baseMipLevel   = 0,
                                        .levelCount     = VK_REMAINING_MIP_LEVELS,
                                        .baseArrayLayer = 0,
                                        .layerCount     = VK_REMAINING_ARRAY_LAYERS};
    vkCmdClearColorImage(cmd, m_resources->textures[m_resources->texFrameIndex].image.image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);
  }

  if(UNSET_SIZET != m_resources->texDepthIndex && m_shaderParams.clearDepthStencilWhen == ClearWhen::eAlways)
  {
    auto                   profilerRangeClear = m_profilerGPU.cmdFrameSection(cmd, "ClearDepth");
    Texture&               tex                = m_resources->textures[m_resources->texDepthIndex];
    nvvk::BarrierContainer barriers;
    tex.addTransitionTo(barriers.imageBarriers, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, nvvk::INFER_BARRIER_PARAMS, true);
    barriers.cmdPipelineBarrier(cmd, 0);

    const VkClearDepthStencilValue dsValue{.depth = m_shaderParams.clearDepth, .stencil = m_shaderParams.clearStencil};
    const VkImageSubresourceRange  range{.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                                         .baseMipLevel   = 0,
                                         .levelCount     = VK_REMAINING_MIP_LEVELS,
                                         .baseArrayLayer = 0,
                                         .layerCount     = VK_REMAINING_ARRAY_LAYERS};
    vkCmdClearDepthStencilImage(cmd, m_resources->textures[m_resources->texDepthIndex].image.image,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &dsValue, 1, &range);
  }

  // For each pass in the pipeline:
  for(size_t passIndex = 0; passIndex < m_resources->passes.size(); passIndex++)
  {
    const Pass& pass              = m_resources->passes[passIndex];
    auto        profilerRangePass = m_profilerGPU.cmdFrameSection(cmd, "Pass 0: " + pass.debugName);
    // Memory barriers as needed
    nvvk::BarrierContainer barriers;
    for(size_t i = 0; i < pass.usedTextures.size(); i++)
    {
      const Pass::UsedResource& use = pass.usedTextures[i];
      Texture&                  tex = m_resources->textures[use.resourceIndex];
      tex.addTransitionTo(barriers.imageBarriers, use.layout, use.stages, true /* TODO: Check if written */);
    }
    for(size_t i = 0; i < pass.usedUniformBuffers.size(); i++)
    {
      const Pass::UsedResource& use = pass.usedUniformBuffers[i];
      UniformBuffer&            buf = m_resources->uniformBuffers[use.resourceIndex];
      buf.addTransitionTo(barriers.bufferBarriers, use.stages, false);
    }
    for(size_t i = 0; i < pass.usedStorageBuffers.size(); i++)
    {
      const Pass::UsedResource& use = pass.usedStorageBuffers[i];
      StorageBuffer&            buf = m_resources->storageBuffers[use.resourceIndex];
      buf.addTransitionTo(barriers.bufferBarriers, use.stages, true /* TODO: Check if written */);
    }

    barriers.cmdPipelineBarrier(cmd, 0);

    // Bind and dispatch pipeline
    if((pass.shaderStages & ~kAllComputeStages) == 0)  // Compute shader
    {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass.pipeline);
      if(frameDescriptorSets.size() > 0)
      {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_resources->pipelineLayout, 0,
                                static_cast<uint32_t>(frameDescriptorSets.size()), frameDescriptorSets.data(), 0, nullptr);
      }

      // Dispatch (always screen-sized for now)
      std::array<uint32_t, 3> numWorkgroups{};
      numWorkgroups[0] = static_cast<uint32_t>((resolution.width + pass.workgroupSize[0] - 1) / pass.workgroupSize[0]);
      numWorkgroups[1] = static_cast<uint32_t>((resolution.height + pass.workgroupSize[1] - 1) / pass.workgroupSize[1]);
      numWorkgroups[2] = 1;
      vkCmdDispatch(cmd, numWorkgroups[0], numWorkgroups[1], numWorkgroups[2]);
    }
    else if((pass.shaderStages & ~kAllRasterGraphicsStages) == 0)  // Raster pipeline with vertex shader
    {
      const Texture& texFrame = m_resources->textures[m_resources->texFrameIndex];
      const Texture& texDepth = m_resources->textures[m_resources->texDepthIndex];

      const VkRenderingAttachmentInfo colorAttachment{
          .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
          .imageView   = texFrame.image.descriptor.imageView,
          .imageLayout = texFrame.image.descriptor.imageLayout,
          .loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD,
          .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
      };

      const VkRenderingAttachmentInfo depthAttachment{.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                                                      .imageView   = texDepth.image.descriptor.imageView,
                                                      .imageLayout = texDepth.image.descriptor.imageLayout,
                                                      .loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD,
                                                      .storeOp     = VK_ATTACHMENT_STORE_OP_STORE};

      const VkRenderingAttachmentInfo stencilAttachment = depthAttachment;

      const VkRenderingInfo beginRenderingInfo{.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
                                               .renderArea           = {.extent = resolution},
                                               .layerCount           = 1,
                                               .colorAttachmentCount = 1,
                                               .pColorAttachments    = &colorAttachment,
                                               .pDepthAttachment     = &depthAttachment,
                                               .pStencilAttachment   = &stencilAttachment};
      vkCmdBeginRendering(cmd, &beginRenderingInfo);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pass.pipeline);
      if(frameDescriptorSets.size() > 0)
      {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_resources->pipelineLayout, 0,
                                static_cast<uint32_t>(frameDescriptorSets.size()), frameDescriptorSets.data(), 0, nullptr);
      }
      // Dynamic state
      const nvvk::GraphicsPipelineState& dynamicState = getOrCreateGraphicsPipelineState(passIndex);
      dynamicState.cmdApplyDynamicStates(cmd, kDynamicStates);
      dynamicState.cmdSetViewportAndScissor(cmd, resolution);
      // TODO: Write object transforms to the vertex buffer
      // Or even better, do this indirect
      for(const nvvkgltf::RenderNode& node : m_currentScene.getRenderNodes())
      {
        vkCmdBindIndexBuffer(cmd, m_currentSceneVk.indices()[node.renderPrimID].buffer, 0, VK_INDEX_TYPE_UINT32);

        if(!pass.vtxAttribInfos.empty())
        {
          const auto&               vertexBuffersNvvk = m_currentSceneVk.vertexBuffers()[node.renderPrimID];
          std::vector<VkBuffer>     vertexBuffers(pass.vtxAttribInfos.size());
          std::vector<VkDeviceSize> vertexBufferOffsets(pass.vtxAttribInfos.size(), 0);
          for(const auto& [location, info] : pass.vtxAttribInfos)
          {
            switch(info.attribute)
            {
              case VertexAttribute::ePosition:
                vertexBuffers[info.binding] = vertexBuffersNvvk.position.buffer;
                break;
              case VertexAttribute::eNormal:
                vertexBuffers[info.binding] = vertexBuffersNvvk.normal.buffer;
                break;
              case VertexAttribute::eTangent:
                vertexBuffers[info.binding] = vertexBuffersNvvk.tangent.buffer;
                break;
              case VertexAttribute::eTexCoord0:
                vertexBuffers[info.binding] = vertexBuffersNvvk.texCoord0.buffer;
                break;
              case VertexAttribute::eTexCoord1:
                vertexBuffers[info.binding] = vertexBuffersNvvk.texCoord1.buffer;
                break;
              case VertexAttribute::eColor:
                vertexBuffers[info.binding] = vertexBuffersNvvk.color.buffer;
                break;
            }
          }
          vkCmdBindVertexBuffers(cmd, 0, static_cast<uint32_t>(vertexBuffers.size()), vertexBuffers.data(),
                                 vertexBufferOffsets.data());
        }

        vkCmdDrawIndexed(cmd, m_currentScene.getRenderPrimitives()[node.renderPrimID].indexCount, 1, 0, 0, 0);
      }

      vkCmdEndRendering(cmd);
    }
    else if((pass.shaderStages & ~kAllRasterMeshStages) == 0)
    {
      LOGE("TODO: Support pipelines with mesh shaders.\n");
    }
    else if((pass.shaderStages & ~kAllRayTracingStages) == 0)
    {
      LOGE("TODO: Support pipelines with ray tracing shaders.\n");
    }
    else
    {
      LOGE("Unknown or undefined pass mask %zu; this should never happen.\n", static_cast<size_t>(pass.shaderStages));
    }
  }

  // Transition the texture to visualize to guiImageLayout
  Texture* displayTexture = getDisplayTexture();
  if(displayTexture)
  {
    nvvk::BarrierContainer barriers;
    displayTexture->addTransitionTo(barriers.imageBarriers, guiImageLayout, nvvk::INFER_BARRIER_PARAMS, false);
    barriers.cmdPipelineBarrier(cmd, 0);
  }

  // Next frame
  m_frame++;
  m_staging.timelineValue++;
}

static VkShaderStageFlagBits vkStageFromSlangStage(SlangStage stage)
{
  switch(stage)
  {
    case SlangStage::SLANG_STAGE_VERTEX:
      return VK_SHADER_STAGE_VERTEX_BIT;
    case SlangStage::SLANG_STAGE_HULL:
      return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
    case SlangStage::SLANG_STAGE_DOMAIN:
      return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    case SlangStage::SLANG_STAGE_GEOMETRY:
      return VK_SHADER_STAGE_GEOMETRY_BIT;
    case SlangStage::SLANG_STAGE_FRAGMENT:
      return VK_SHADER_STAGE_FRAGMENT_BIT;
    case SlangStage::SLANG_STAGE_COMPUTE:
      return VK_SHADER_STAGE_COMPUTE_BIT;
    case SlangStage::SLANG_STAGE_RAY_GENERATION:
      return VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    case SlangStage::SLANG_STAGE_INTERSECTION:
      return VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
    case SlangStage::SLANG_STAGE_ANY_HIT:
      return VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    case SlangStage::SLANG_STAGE_CLOSEST_HIT:
      return VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    case SlangStage::SLANG_STAGE_MISS:
      return VK_SHADER_STAGE_MISS_BIT_KHR;
    case SlangStage::SLANG_STAGE_CALLABLE:
      return VK_SHADER_STAGE_CALLABLE_BIT_KHR;
    case SlangStage::SLANG_STAGE_MESH:
      return VK_SHADER_STAGE_MESH_BIT_EXT;
    case SlangStage::SLANG_STAGE_AMPLIFICATION:
      return VK_SHADER_STAGE_TASK_BIT_EXT;
    default:
      LOGE("Unimplemented or unknown stage %u!\n", static_cast<unsigned>(stage));
      return static_cast<VkShaderStageFlagBits>(0);
  }
}

static SlangStage slangStageFromVkStage(VkShaderStageFlagBits stage)
{
  switch(stage)
  {
    case VK_SHADER_STAGE_VERTEX_BIT:
      return SlangStage::SLANG_STAGE_VERTEX;
    case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT:
      return SlangStage::SLANG_STAGE_HULL;
    case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT:
      return SlangStage::SLANG_STAGE_DOMAIN;
    case VK_SHADER_STAGE_GEOMETRY_BIT:
      return SlangStage::SLANG_STAGE_GEOMETRY;
    case VK_SHADER_STAGE_FRAGMENT_BIT:
      return SlangStage::SLANG_STAGE_FRAGMENT;
    case VK_SHADER_STAGE_COMPUTE_BIT:
      return SlangStage::SLANG_STAGE_COMPUTE;
    case VK_SHADER_STAGE_RAYGEN_BIT_KHR:
      return SlangStage::SLANG_STAGE_RAY_GENERATION;
    case VK_SHADER_STAGE_INTERSECTION_BIT_KHR:
      return SlangStage::SLANG_STAGE_INTERSECTION;
    case VK_SHADER_STAGE_ANY_HIT_BIT_KHR:
      return SlangStage::SLANG_STAGE_ANY_HIT;
    case VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR:
      return SlangStage::SLANG_STAGE_CLOSEST_HIT;
    case VK_SHADER_STAGE_MISS_BIT_KHR:
      return SlangStage::SLANG_STAGE_MISS;
    case VK_SHADER_STAGE_CALLABLE_BIT_KHR:
      return SlangStage::SLANG_STAGE_CALLABLE;
    case VK_SHADER_STAGE_MESH_BIT_EXT:
      return SlangStage::SLANG_STAGE_MESH;
    case VK_SHADER_STAGE_TASK_BIT_EXT:
      return SlangStage::SLANG_STAGE_AMPLIFICATION;
    default:
      LOGE("Unimplemented or unknown stage %u!\n", static_cast<unsigned>(stage));
      return SLANG_STAGE_NONE;
  }
}

// Note: This uses the implementation from gui_reflection.cpp;
// maybe this should be moved to a common file.
const char* slangStageToString(SlangStage stage);

// This is for resources accessed by shaders; don't use this for places where
// the pipeline accesses resources outside of shaders.
static VkPipelineStageFlagBits2 getPipelineStage2FromShaderStage(VkShaderStageFlagBits stage)
{
  switch(stage)
  {
    case VK_SHADER_STAGE_VERTEX_BIT:
      return VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT:
      return VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT;
    case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT:
      return VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT;
    case VK_SHADER_STAGE_GEOMETRY_BIT:
      return VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT;
    case VK_SHADER_STAGE_FRAGMENT_BIT:
      return VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    case VK_SHADER_STAGE_COMPUTE_BIT:
      return VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    case VK_SHADER_STAGE_RAYGEN_BIT_KHR:
    case VK_SHADER_STAGE_ANY_HIT_BIT_KHR:
    case VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR:
    case VK_SHADER_STAGE_MISS_BIT_KHR:
    case VK_SHADER_STAGE_INTERSECTION_BIT_KHR:
    case VK_SHADER_STAGE_CALLABLE_BIT_KHR:
      return VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    case VK_SHADER_STAGE_TASK_BIT_EXT:
      return VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT;
    case VK_SHADER_STAGE_MESH_BIT_EXT:
      return VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
    default:
      LOGE("Got 0 or an unhandled value for VkShaderStageFlagBits (%u)!\n", static_cast<unsigned>(stage));
      return 0;
  }
}

static VkDescriptorType vkDescriptorTypeFromSlangBinding(slang::BindingType bindingType)
{
  switch(bindingType)
  {
    case slang::BindingType::Sampler:
      return VK_DESCRIPTOR_TYPE_SAMPLER;
    case slang::BindingType::CombinedTextureSampler:
      return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    case slang::BindingType::Texture:
      return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    case slang::BindingType::MutableTexture:
      return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    case slang::BindingType::TypedBuffer:
      return VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    case slang::BindingType::MutableTypedBuffer:
      return VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
    case slang::BindingType::RawBuffer:
    case slang::BindingType::MutableRawBuffer:
      return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case slang::BindingType::InputRenderTarget:
      return VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    case slang::BindingType::InlineUniformData:
      return VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT;
    case slang::BindingType::RayTracingAccelerationStructure:
      return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    case slang::BindingType::ConstantBuffer:
      return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    default:
      LOGE("Unsupported binding type %u!\n", static_cast<unsigned>(bindingType));
      return VK_DESCRIPTOR_TYPE_MAX_ENUM;
  }
}

static VkFormat vkFormatFromSlangFormat(SlangImageFormat imageFormat)
{
  switch(imageFormat)
  {
      // See https://docs.vulkan.org/spec/latest/appendices/spirvenv.html
      // Slang lists them in SPIR-V order; we need to turn them into Vulkan order
    case SLANG_IMAGE_FORMAT_unknown:
      return VK_FORMAT_UNDEFINED;
    case SLANG_IMAGE_FORMAT_rgba32f:
      return VK_FORMAT_R32G32B32A32_SFLOAT;
    case SLANG_IMAGE_FORMAT_rgba16f:
      return VK_FORMAT_R16G16B16A16_SFLOAT;
    case SLANG_IMAGE_FORMAT_rg32f:
      return VK_FORMAT_R32G32_SFLOAT;
    case SLANG_IMAGE_FORMAT_rg16f:
      return VK_FORMAT_R16G16_SFLOAT;
    case SLANG_IMAGE_FORMAT_r11f_g11f_b10f:
      return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case SLANG_IMAGE_FORMAT_r32f:
      return VK_FORMAT_R32_SFLOAT;
    case SLANG_IMAGE_FORMAT_r16f:
      return VK_FORMAT_R16_SFLOAT;
    case SLANG_IMAGE_FORMAT_rgba16:
      return VK_FORMAT_R16G16B16A16_UNORM;
    case SLANG_IMAGE_FORMAT_rgb10_a2:
      return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case SLANG_IMAGE_FORMAT_rgba8:
      return VK_FORMAT_R8G8B8A8_UNORM;
    case SLANG_IMAGE_FORMAT_rg16:
      return VK_FORMAT_R16G16_UNORM;
    case SLANG_IMAGE_FORMAT_rg8:
      return VK_FORMAT_R8G8_UNORM;
    case SLANG_IMAGE_FORMAT_r16:
      return VK_FORMAT_R16_UNORM;
    case SLANG_IMAGE_FORMAT_r8:
      return VK_FORMAT_R8_UNORM;
    case SLANG_IMAGE_FORMAT_rgba16_snorm:
      return VK_FORMAT_R16G16B16A16_SNORM;
    case SLANG_IMAGE_FORMAT_rgba8_snorm:
      return VK_FORMAT_R8G8B8A8_SNORM;
    case SLANG_IMAGE_FORMAT_rg8_snorm:
      return VK_FORMAT_R8G8_SNORM;
    case SLANG_IMAGE_FORMAT_r16_snorm:
      return VK_FORMAT_R16_SNORM;
    case SLANG_IMAGE_FORMAT_r8_snorm:
      return VK_FORMAT_R8_SNORM;
    case SLANG_IMAGE_FORMAT_rgba32i:
      return VK_FORMAT_R32G32B32A32_SINT;
    case SLANG_IMAGE_FORMAT_rgba16i:
      return VK_FORMAT_R16G16B16A16_SINT;
    case SLANG_IMAGE_FORMAT_rgba8i:
      return VK_FORMAT_R8G8B8A8_SINT;
    case SLANG_IMAGE_FORMAT_rg32i:
      return VK_FORMAT_R32G32_SINT;
    case SLANG_IMAGE_FORMAT_rg16i:
      return VK_FORMAT_R16G16_SINT;
    case SLANG_IMAGE_FORMAT_rg8i:
      return VK_FORMAT_R8G8_SINT;
    case SLANG_IMAGE_FORMAT_r32i:
      return VK_FORMAT_R32_SINT;
    case SLANG_IMAGE_FORMAT_r16i:
      return VK_FORMAT_R16_SINT;
    case SLANG_IMAGE_FORMAT_r8i:
      return VK_FORMAT_R8_SINT;
    case SLANG_IMAGE_FORMAT_rgba32ui:
      return VK_FORMAT_R32G32B32A32_UINT;
    case SLANG_IMAGE_FORMAT_rgba16ui:
      return VK_FORMAT_R16G16B16A16_UINT;
    case SLANG_IMAGE_FORMAT_rgb10_a2ui:
      return VK_FORMAT_A2B10G10R10_UINT_PACK32;
    case SLANG_IMAGE_FORMAT_rgba8ui:
      return VK_FORMAT_R8G8B8A8_UINT;
    case SLANG_IMAGE_FORMAT_rg32ui:
      return VK_FORMAT_R32G32_UINT;
    case SLANG_IMAGE_FORMAT_rg16ui:
      return VK_FORMAT_R16G16_UINT;
    case SLANG_IMAGE_FORMAT_rg8ui:
      return VK_FORMAT_R8G8_UINT;
    case SLANG_IMAGE_FORMAT_r32ui:
      return VK_FORMAT_R32_UINT;
    case SLANG_IMAGE_FORMAT_r16ui:
      return VK_FORMAT_R16_UINT;
    case SLANG_IMAGE_FORMAT_r8ui:
      return VK_FORMAT_R8_UINT;
    case SLANG_IMAGE_FORMAT_r64ui:
      return VK_FORMAT_R64_UINT;
    case SLANG_IMAGE_FORMAT_r64i:
      return VK_FORMAT_R64_SINT;
    case SLANG_IMAGE_FORMAT_bgra8:
      return VK_FORMAT_B8G8R8A8_UNORM;
    default:
      LOGE("Unknown image type %u!\n", static_cast<unsigned>(imageFormat));
      return VK_FORMAT_UNDEFINED;
  }
}

static std::string_view removeUniformPrefix(std::string_view name)
{
  if(startsWithI(name, "i") || startsWithI(name, "f"))  // iFrame, fFrame -> Frame
  {
    name = name.substr(1);
  }
  else if(startsWithI(name, "v2"))  // v2Resolution -> Resolution
  {
    name = name.substr(2);
  }

  return name;
}

static std::string textureVariableToId(std::string_view name)
{
  if(startsWithI(name, "tex"))
  {
    name = name.substr(3);
  }
  else if(startsWithI(name, "texture"))
  {
    name = name.substr(7);
  }
  std::string id = std::string(name);
  for(char& c : id)
  {
    c = myToLower(c);
  }
  return id;
}

// Recursively iterate over all shader parameters.
// There's a lot of functions in the Slang API to do this, so it can be quite
// overwhelming.
// The best way to do this is based on
// https://github.com/shader-slang/slang-rhi/blob/83d6c0967c6e6e16d9a05dbe94d374527b914c89/src/vulkan/vk-shader-object-layout.h :
// * Track offsets for the descriptor set, descriptor binding, push constant range, and buffer byte offset
//   * Use getBindingSpace/getOffset for the set and binding; getOffset for the rest.
// * Use getBindingRangeType to get the descriptor type
// TODO: Check if the sub-object API is more elegant than the Kind enum
// TODO: This actually needs to be redone. Basically,
// https://github.com/shader-slang/slang-rhi/blob/main/src/vulkan/vk-shader-object-layout.cpp#L62 is the only way to do it.
// There's some real subtleties when containers use offsets that aren't applied to their elements.
struct TraversalShaderOffset
{
  DescriptorIndex desc{};
  uint32_t        pushConstantRange = 0;
  size_t          byte              = 0;
};
struct TraversalOutputs
{
  std::unordered_map<DescriptorIndex, size_t> descIdxToResource;
  std::vector<Diagnostic>                     diagnostics;
  std::unordered_map<std::string, size_t>     nameToStorageBuffer;
  std::unordered_map<std::string, size_t>     nameToTexture;

  Resources* resources = nullptr;
};
static void traverseVariableLayout(slang::VariableLayoutReflection* varLayoutRefl, TraversalOutputs& outputs, TraversalShaderOffset offset)
{
  assert(varLayoutRefl);
  slang::TypeLayoutReflection* typeLayoutRefl = varLayoutRefl->getTypeLayout();
  slang::TypeReflection*       typeRefl       = typeLayoutRefl->getType();
  const char*                  name           = varLayoutRefl->getName();
  if(name == nullptr)  // This can happen for the global uniform buffer
  {
    name = "<unnamed>";
  }

  // Apply offsets
  offset.desc.set += static_cast<uint32_t>(varLayoutRefl->getBindingSpace(SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT));
  offset.desc.binding += static_cast<uint32_t>(varLayoutRefl->getOffset(SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT));
  offset.pushConstantRange += static_cast<uint32_t>(varLayoutRefl->getOffset(SLANG_PARAMETER_CATEGORY_PUSH_CONSTANT_BUFFER));
  offset.byte += varLayoutRefl->getOffset(SLANG_PARAMETER_CATEGORY_UNIFORM);

  const slang::TypeReflection::Kind kind = typeLayoutRefl->getKind();
  if(kind == slang::TypeReflection::Kind::Struct)
  {
    // Nothing to allocate for a struct -- we just need to iterate
    // over each of its fields and process those.
    unsigned int fieldCount = typeLayoutRefl->getFieldCount();
    for(unsigned int fieldIdx = 0; fieldIdx < fieldCount; fieldIdx++)
    {
      slang::VariableLayoutReflection* fieldRefl = typeLayoutRefl->getFieldByIndex(fieldIdx);
      traverseVariableLayout(fieldRefl, outputs, offset);
    }
    return;
  }
  else if(kind == slang::TypeReflection::Kind::Scalar || kind == slang::TypeReflection::Kind::Vector
          || kind == slang::TypeReflection::Kind::Matrix)
  {
    UniformWrite write{.name       = name,
                       .byteOffset = offset.byte,
                       .scalarType = static_cast<SlangScalarType>(typeRefl->getScalarType()),
                       .rows       = std::max(1U, typeRefl->getRowCount()),
                       .cols       = std::max(1U, typeRefl->getColumnCount())};
    // Which uniform buffer should we write to?
    const auto& it = outputs.descIdxToResource.find(offset.desc);
    if(it == outputs.descIdxToResource.end())
    {
      LOGE("Couldn't find a uniform buffer!\n");
      assert(false);
    }
    write.bufferIndex = it->second;
    outputs.resources->uniformUpdates.push_back(std::move(write));
    return;
  }

  // Otherwise, we have a resource of some sort.
  DescriptorWrite desc = {.index = offset.desc};
  // Assume it generates exactly one binding range. What descriptor type is it?
  // TODO: Handle AppendStructuredBuffer, which generates 2 binding ranges
  const SlangUInt bindingRangeCount = typeLayoutRefl->getBindingRangeCount();
  assert(bindingRangeCount == 1);

  desc.descriptorType = vkDescriptorTypeFromSlangBinding(typeLayoutRefl->getBindingRangeType(0));
  if(desc.descriptorType == VK_DESCRIPTOR_TYPE_MAX_ENUM)
  {
    outputs.diagnostics.push_back({.text = std::string("Unknown descriptor type for ") + name});
    return;
  }

  if(kind == slang::TypeReflection::Kind::ConstantBuffer)
  {
    desc.resourceIndex                    = outputs.resources->uniformBuffers.size();
    outputs.descIdxToResource[desc.index] = desc.resourceIndex;
    UniformBuffer newUniformBuffer{.name = name, .currentStages = VK_PIPELINE_STAGE_2_TRANSFER_BIT};
    outputs.resources->uniformBuffers.push_back(std::move(newUniformBuffer));
    outputs.resources->descriptorSetUpdates.push_back(std::move(desc));
    // We also want to iterate over the fields of the constant buffer:
    traverseVariableLayout(typeLayoutRefl->getElementVarLayout(), outputs, offset);
    return;
  }
  else if(kind == slang::TypeReflection::Kind::Resource)
  {
    SlangResourceShape shape = typeRefl->getResourceShape();
    // Ignore the COMBINED flag for now since this is Vulkan
    shape = static_cast<SlangResourceShape>(shape & ~SLANG_TEXTURE_COMBINED_FLAG);
    if(shape == SLANG_TEXTURE_1D || shape == SLANG_TEXTURE_2D || shape == SLANG_TEXTURE_3D || shape == SLANG_TEXTURE_CUBE)
    {
      // Have we already added this texture under another name?
      const auto& it = outputs.nameToTexture.find(name);
      if(it == outputs.nameToTexture.end())
      {
        // It's a new texture
        desc.resourceIndex          = outputs.resources->textures.size();
        outputs.nameToTexture[name] = desc.resourceIndex;
        Texture tex{.name = name, .format = vkFormatFromSlangFormat(varLayoutRefl->getImageFormat())};
        outputs.resources->textures.push_back(std::move(tex));
      }
      else
      {
        // It's an old one
        desc.resourceIndex = it->second;
      }

      desc.layout = VK_IMAGE_LAYOUT_GENERAL;  // TODO: VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
    }
    else if(shape == SLANG_STRUCTURED_BUFFER || shape == SLANG_BYTE_ADDRESS_BUFFER)
    {
      // Have we already added this buffer under another name?
      const auto& it = outputs.nameToStorageBuffer.find(name);
      if(it == outputs.nameToStorageBuffer.end())
      {
        // It's a new SSBO
        desc.resourceIndex                = outputs.resources->storageBuffers.size();
        outputs.nameToStorageBuffer[name] = desc.resourceIndex;
        StorageBuffer buf{.name = name};
        if(shape == SLANG_STRUCTURED_BUFFER)
        {
          // The tricky thing here is getting the stride between elements.
          // StructuredBuffer is an array-like type, and usually contains
          // more than one element. So we want to use getElementTypeLayout,
          // not getElementVarLayout -- getElementVarLayout won't give us the
          // stride info!
          // Which sort of makes sense; we're asking "what's the stride between
          // elements of T?" instead of "what's the offset of `iResolution`?".
          slang::TypeLayoutReflection* elementTypeLayoutRefl = typeLayoutRefl->getElementTypeLayout();
          if(!elementTypeLayoutRefl)
          {
            std::string text = std::string(
                                   "Could not get element type layout reflection for the "
                                   "StructuredBuffer with name `")
                               + name + "`; defaulting to a stride of " + std::to_string(buf.elementStride) + "bytes.";
            outputs.diagnostics.push_back({.text = std::move(text), .level = Diagnostic::Level::eWarning});
          }
          else
          {
            const size_t stride = elementTypeLayoutRefl->getStride();
            if(buf.elementStride != 0 && buf.elementStride != stride)
            {
              std::string text =
                  std::string("The StructuredBuffer with name `") + name + "` was defined with multiple types; the first one had stride "
                  + std::to_string(buf.elementStride) + ", and the second one had stride " + std::to_string(stride)
                  + ". vk_slang_editor will bind both to the same range of memory, so the same data might be read as different types. Are you sure you want to do that?";
              outputs.diagnostics.push_back({.text = std::move(text), .level = Diagnostic::Level::eWarning});
            }
            else
            {
              buf.elementStride = stride;
            }
          }
        }

        outputs.resources->storageBuffers.push_back(std::move(buf));
      }
      else
      {
        // It's an old one
        desc.resourceIndex = it->second;
      }
    }
    else
    {
      outputs.diagnostics.push_back({.text = TARGET_NAME "does not implement Slang resource shape "
                                             + std::to_string(static_cast<uint32_t>(shape)) + "!"});
    }
    outputs.resources->descriptorSetUpdates.push_back(std::move(desc));
    return;
  }

  outputs.diagnostics.push_back({.text = "Unimplemented type kind " + std::to_string(static_cast<uint32_t>(kind)) + "!"});
}

// Recursively iterate over a struct to find vertex attributes.
// Since we only have to handle structs and variables, this is easier than
// the more complex traversal function above. But we still have to handle
// cases where e.g. one of the attributes was optimized out.
static void traverseForVertexAttributes(std::unordered_map<unsigned, VtxAttribInfo>& attributes,
                                        slang::VariableLayoutReflection*             varLayoutRefl,
                                        unsigned                                     offset = 0)
{
  assert(varLayoutRefl);
  slang::TypeLayoutReflection* typeLayoutRefl = varLayoutRefl->getTypeLayout();
  slang::VariableReflection*   varRefl        = varLayoutRefl->getVariable();
  const char*                  name           = varRefl->getName();

  // Make sure this has the varyingInput parameter category; otherwise,
  // this is something like a Texture2D passed into inputs.
  bool           hadVaryingInput = false;
  const unsigned numCategories   = typeLayoutRefl->getCategoryCount();
  for(unsigned i = 0; i < numCategories; i++)
  {
    const slang::ParameterCategory category = typeLayoutRefl->getCategoryByIndex(i);
    if(category == slang::ParameterCategory::VaryingInput)
    {
      hadVaryingInput = true;
    }
  }
  if(!hadVaryingInput)
  {
    return;
  }

  offset += static_cast<unsigned>(varLayoutRefl->getOffset(SLANG_PARAMETER_CATEGORY_VARYING_INPUT));

  const slang::TypeReflection::Kind kind = typeLayoutRefl->getKind();
  if(kind == slang::TypeReflection::Kind::Struct)
  {
    // Iterate over sub-fields
    unsigned int fieldCount = typeLayoutRefl->getFieldCount();
    for(unsigned int fieldIdx = 0; fieldIdx < fieldCount; fieldIdx++)
    {
      slang::VariableLayoutReflection* fieldRefl = typeLayoutRefl->getFieldByIndex(fieldIdx);
      traverseForVertexAttributes(attributes, fieldRefl, offset);
    }
    return;
  }
  else if(kind == slang::TypeReflection::Kind::Vector)
  {
    // Awesome, we have an input!
    // Do we recognize it?
    VertexAttribute attrib = VertexAttribute::eUnknown;
    // Since semantics are standardized, we prefer matching against those.
    // See https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-semantics#vertex-shader-semantics
    // Otherwise (or if there's no semantic), we use the variable-matching logic.
    const char* semanticName = varLayoutRefl->getSemanticName();
    if(semanticName)
    {
      if(strcmp(semanticName, "POSITION") == 0 || strcmp(semanticName, "POSITION0") == 0)
      {
        attrib = VertexAttribute::ePosition;
      }
      else if(strcmp(semanticName, "NORMAL") == 0 || strcmp(semanticName, "NORMAL0") == 0)
      {
        attrib = VertexAttribute::eNormal;
      }
      else if(strcmp(semanticName, "TANGENT") == 0 || strcmp(semanticName, "TANGENT0") == 0)
      {
        attrib = VertexAttribute::eTangent;
      }
      else if(strcmp(semanticName, "TEXCOORD") == 0 || strcmp(semanticName, "TEXCOORD0") == 0)
      {
        attrib = VertexAttribute::eTexCoord0;
      }
      else if(strcmp(semanticName, "TEXCOORD1") == 0)
      {
        attrib = VertexAttribute::eTexCoord1;
      }
      else if(strcmp(semanticName, "COLOR") == 0 || strcmp(semanticName, "COLOR0") == 0)
      {
        attrib = VertexAttribute::eColor;
      }
    }

    if(attrib == VertexAttribute::eUnknown)
    {
      if(strieq(name, "position"))
      {
        attrib = VertexAttribute::ePosition;
      }
      else if(strieq(name, "normal"))
      {
        attrib = VertexAttribute::eNormal;
      }
      else if(strieq(name, "tangent"))
      {
        attrib = VertexAttribute::eTangent;
      }
      else if(strieqList(name, {"texcoord", "texcoord0"}))
      {
        attrib = VertexAttribute::eTexCoord0;
      }
      else if(strieq(name, "texcoord1"))
      {
        attrib = VertexAttribute::eTexCoord1;
      }
      else if(strieq(name, "color"))
      {
        attrib = VertexAttribute::eColor;
      }
    }

    // Note: It would probably be nice to output diagnostics if the vector
    // size isn't what we expect.
    // And also unknown attributes should be diagnostics instead of log messages.
    if(attrib != VertexAttribute::eUnknown)
    {
      assert(!attributes.contains(offset));  // If this happens it's a programmer error
      attributes[offset].attribute = attrib;
    }
  }
  else
  {
    LOGW("Varying input that we don't know how to handle: %s\n", name);
  }
}


// Using SPIR-V and reflection info, builds Vulkan resources.
// On success, returns VK_SUCCESS; on failure, the returned Resources should
// be destroyed.
static VkResult buildResourcesFromReflection(Resources&                    resources,
                                             std::vector<Diagnostic>&      diagnostics,
                                             const nvslang::SlangCompiler& compiler,
                                             nvvk::ResourceAllocator&      alloc,
                                             nvvk::StagingUploader&        staging,
                                             nvvk::SamplerPool&            samplerPool,
                                             TextureCache&                 textureCache,
                                             nvapp::Application&           app)
{
  // Probably the easiest way to motivate the structure of the code is by
  // looking at what we want the main render loop to be:
  // ```
  // 1. For each shader parameter that's used by at least one pass:
  //    - Write its value into a buffer / queue a push for the descriptor for this frame
  // 2. Upload all changed buffers + push descriptor sets
  // 3. Insert a GPU timeline semaphore to wait for buffer uploads to complete
  // 3a. Clear texFrame and the main depth buffer.
  // 4. Bind descriptor sets
  // 5. For each pass in the technique:
  //    - Memory barriers as needed
  //    - Bind pipeline
  //    - Bind vertex buffers/index buffers/etc.
  //    - Draw/dispatch
  // 6. All G-buffers to GENERAL layout so that they can potentially be visualized
  // ```
  //
  // For (1), we need two lists: one of "get value X and write it into buffer Y at offset Z",
  // and one of "write a descriptor for this frame's resource X and write it into descriptor set Y at offset Z".
  // (TODO: Detect buffers that don't change over time and allocate them as needed).
  //
  // (2) and (3) require lists of which buffers/descriptor sets changed; we can build that during (1).
  //
  // (4) is easy: it's a single vkBindDescriptorSets2 call.
  //
  // (5) means:
  // - Each resource needs to track its layout and what it was just written by
  //   (although that last one is optional since we can use NVVK's helpers to
  //   infer the parts that matter from the layout)
  // - Each pass needs to list the pipeline, how it uses each resource[, and which mesh it uses].
  //
  // So we need to have structures like this:
  // enum class KnownUniform { eTime, eFrame, eResolution, ... }
  // struct KnownUniformWrite { KnownUniform knownUniform; size_t bufferIndex; size_t offset; }
  // struct ConstantBuffer { nvvk::Buffer buffer; VkStageFlags usages; }
  // struct TextureWrite { size_t textureIndex; size_t descriptorSet; size_t descriptorIndex; }
  // struct Texture{ nvvk::GBuffer gBuffer; VkImageLayout layout; VkStageFlags lastWrittenBy; }
  // struct Pass{ size_t pipelineIndex; std::vector<struct{size_t textureIndex, VkStageFlags}> usedTextures; }
  //
  // When we're allocating textures, we also need to know their order, and
  // infer requirements on them (should it be loaded from a file? is it texFrame,
  // which is the one we show by default?). That means we need a mapping from
  // variable name -> texture index, and that the texture struct also needs to
  // store its requirements. Additionally, if we have that mapping, we can
  // check to see if a texture's requirements have changed -- if they haven't,
  // then we can re-use it (and avoid re-allocating/clearing it).
  //
  // That means our structure here will look like this:
  // - Iterate over entrypoints; organize them into Passes. Create a map of
  // [entrypoint index] -> [pass index].
  // - Recursively iterate over all shader parameters
  //   - Figure out which entrypoints they're used by
  //   - Construct KnownConstantWrites, TextureWrites, Texture requirements,
  //     and fill in Pass.usedTextures info
  //   - This also gives us descriptor set layout info
  // - Load/construct/name textures and buffers
  // - Construct descriptor sets
  // - Construct pipelines

  slang::IComponentType* program = compiler.getSlangProgram();
  assert(program);
  slang::ProgramLayout* reflection = program->getLayout();
  assert(reflection);

  // Recursively iterate over all shader parameters.
  TraversalOutputs outputs;
  outputs.resources = &resources;
  traverseVariableLayout(reflection->getGlobalParamsVarLayout(), outputs, {});
#if 0  // TODO: Traverse over entrypoint-specific params; see https://docs.shader-slang.org/en/latest/external/slang/docs/user-guide/09-reflection.html#programs-and-scopes
  const unsigned parameterCount = reflection->getParameterCount();
  for(unsigned i = 0; i < parameterCount; i++)
  {
    slang::VariableLayoutReflection* varLayoutRefl = reflection->getParameterByIndex(i);
    traverseVariableLayout(varLayoutRefl, outputs, {});
  }
#endif

  if(!outputs.diagnostics.empty())
  {
    for(size_t i = 0; i < outputs.diagnostics.size(); i++)
    {
      diagnostics.push_back(std::move(outputs.diagnostics[i]));
    }
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  // Iterate over entrypoints; organize them into Passes.
  //
  // In vk_slang_editor, we take the approach where the user doesn't specify
  // which passes use which entrypoints; we just try to infer what they meant
  // by the order in which they specified them.
  //
  // For instance,if they specified a vertex shader, a fragment shader, a
  // compute shader, and then another fragment shader, they probably wanted
  // 3 passes, using the vertex shader twice: [vs 0, fs 0], [cs 0], [vs 0, fs 1].
  //
  // Similarly, if they specified a fragment shader, a mesh shader,
  // and then a vertex shader, they probably listed the fragment shader first
  // because it was most important, and wanted two rasterization passes:
  // one using the mesh shader + fs, and then one using the vertex shader + fs.
  //
  // The algorithm for this I have in mind is sort of like playing a falling-
  // block game: as entrypoints come in, we check them against the previous
  // entrypoints to see if they fit into a pass. If they're blocked by a pass
  // or hit the ground, then we create a new one.
  // At the end, we search backwards and then forwards to fill in any missing
  // shaders -- so that vs0, fs0, fs1, vs1, fs2 gives us
  // [vs0, fs0], [vs0, fs1], [vs1, fs2].
  const SlangUInt entrypointCount = reflection->getEntryPointCount();
  if(entrypointCount == 0)
  {
    // TODO: "Try adding a simple shader from the Add menu."
    diagnostics.push_back({.text = "No entrypoints found!"});
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  // First pass of falling block algorithm; gets data and also sets up
  // m_resources->passes[*].passType.
  std::unordered_map<SlangUInt, std::vector<size_t>> entrypointIndexToPasses;
  std::unordered_map<size_t, std::vector<SlangUInt>> passIndexToEntrypoints;

  std::vector<std::string>           entrypointNames(entrypointCount);
  std::vector<VkShaderStageFlagBits> entrypointStages(entrypointCount);
  std::vector<slang::IMetadata*>     entrypointMetadata(entrypointCount);
  for(SlangUInt entrypointIdx = 0; entrypointIdx < entrypointCount; entrypointIdx++)
  {
    slang::EntryPointReflection* entrypoint     = reflection->getEntryPointByIndex(entrypointIdx);
    const char*                  entrypointName = entrypoint->getName();
    entrypointNames[entrypointIdx]              = entrypointName;
    const SlangResult slangResult = program->getEntryPointMetadata(entrypointIdx, 0, &entrypointMetadata[entrypointIdx]);
    if(SLANG_FAILED(slangResult))
    {
      diagnostics.push_back(
          {.text = std::string("Could not get entrypoint metadata for entrypoint ") + std::to_string(entrypointIdx) + "!"});
      return VK_ERROR_INITIALIZATION_FAILED;
    }

    const SlangStage slangStage = entrypoint->getStage();
    // The pass we match against must not have a shader for this stage already:
    const VkShaderStageFlagBits vkStage = vkStageFromSlangStage(slangStage);
    // And must not contain any shaders outside of this set:
    VkShaderStageFlags vkMatchStages{};
    switch(slangStage)
    {
      case SlangStage::SLANG_STAGE_VERTEX:
      case SlangStage::SLANG_STAGE_HULL:
      case SlangStage::SLANG_STAGE_DOMAIN:
      case SlangStage::SLANG_STAGE_GEOMETRY:
        vkMatchStages = kAllRasterGraphicsStages;
        break;
      case SlangStage::SLANG_STAGE_FRAGMENT:
        vkMatchStages = kAllRasterStages;
        break;
      case SlangStage::SLANG_STAGE_COMPUTE:
        vkMatchStages = kAllComputeStages;
        break;
      case SlangStage::SLANG_STAGE_RAY_GENERATION:
      case SlangStage::SLANG_STAGE_INTERSECTION:
      case SlangStage::SLANG_STAGE_ANY_HIT:
      case SlangStage::SLANG_STAGE_CLOSEST_HIT:
      case SlangStage::SLANG_STAGE_MISS:
      case SlangStage::SLANG_STAGE_CALLABLE:
        vkMatchStages = kAllRayTracingStages;
        break;
      case SlangStage::SLANG_STAGE_MESH:
      case SlangStage::SLANG_STAGE_AMPLIFICATION:
        vkMatchStages = kAllRasterMeshStages;
        break;
      default:
        LOGE("Unimplemented or unknown stage %u!\n", static_cast<unsigned>(slangStage));
        break;
    }

    const size_t numPassesBefore = resources.passes.size();
    size_t       passIndex       = numPassesBefore;
    for(int64_t searchIndex = static_cast<int64_t>(numPassesBefore) - 1; searchIndex >= 0; searchIndex--)
    {
      auto& pass = resources.passes[searchIndex];
      if((pass.shaderStages & vkStage) == 0 && (pass.shaderStages & ~vkMatchStages) == 0)
      {
        passIndex = static_cast<size_t>(searchIndex);
        pass.debugName += std::string(" + ") + entrypointName;
        break;
      }
    }

    if(passIndex == numPassesBefore)
    {
      // Create a new pass
      resources.passes.push_back({.debugName = entrypointName});
    }

    resources.passes[passIndex].shaderStages |= static_cast<VkShaderStageFlags>(vkStage);
    entrypointStages[entrypointIdx] = vkStage;
    entrypointIndexToPasses[entrypointIdx].push_back(passIndex);
    passIndexToEntrypoints[passIndex].push_back(entrypointIdx);
  }

  // Now search back and forth to find missing stages.
  {
    bool failed = false;
    for(size_t passIdx = 0; passIdx < resources.passes.size(); passIdx++)
    {
      auto& pass = resources.passes[passIdx];
      // Determine which stages we need to find.
      assert(pass.shaderStages != 0);
      VkShaderStageFlags requiredStages = 0;
      // Is this a traditional rasterization pipeline?
      if(isRasterGraphicsFlags(pass.shaderStages))
      {
        // Require a vertex and fragment shader
        requiredStages |= VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
      }
      if(pass.shaderStages & VK_SHADER_STAGE_TASK_BIT_EXT)
      {
        // Must have all 3 shaders
        requiredStages |= kAllRasterMeshStages;
      }
      if(pass.shaderStages & VK_SHADER_STAGE_MESH_BIT_EXT)
      {
        // Must have a fragment shader; don't need a task shader
        requiredStages |= VK_SHADER_STAGE_FRAGMENT_BIT;
      }
      // If we have a TCS, we must also have a TES, and vice versa
      if(pass.shaderStages & (VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT))
      {
        requiredStages |= VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
      }
      if(pass.shaderStages & kAllRayTracingStages)
      {
        // Must have a ray gen shader.
        requiredStages |= VK_SHADER_STAGE_RAYGEN_BIT_KHR;
      }

      // Clear stages we've already found
      requiredStages &= ~pass.shaderStages;

      // If the pass with the given index has shaders we need, add them to
      // the current pass.
      auto doPass = [&](size_t otherIdx) {
        const auto& otherPass = resources.passes[otherIdx];
        // Rasterization pipelines are the only ones where we have to disambiguate
        // and check that they're the same type before adding shaders
        // we need from it.
        if(isRasterGraphicsFlags(pass.shaderStages) == isRasterGraphicsFlags(otherPass.shaderStages))
        {
          for(SlangUInt otherEntrypointIdx : passIndexToEntrypoints[otherIdx])
          {
            const VkShaderStageFlagBits otherEntrypointStage = entrypointStages[otherEntrypointIdx];
            if(requiredStages & otherEntrypointStage)
            {
              requiredStages &= ~otherEntrypointStage;
              pass.shaderStages |= otherEntrypointStage;
              pass.debugName += " + " + entrypointNames[otherEntrypointIdx];
              passIndexToEntrypoints[passIdx].push_back(otherEntrypointIdx);
              entrypointIndexToPasses[otherEntrypointIdx].push_back(passIdx);
            }
          }
        }
      };

      for(int64_t searchIdx = static_cast<int64_t>(passIdx) - 1; searchIdx >= 0 && (requiredStages != 0); searchIdx--)
      {
        doPass(static_cast<size_t>(searchIdx));
      }

      for(size_t searchIdx = passIdx + 1; searchIdx < resources.passes.size() && (requiredStages != 0); searchIdx++)
      {
        doPass(searchIdx);
      }

      // If we didn't find all the passes we need, emit a diagnostic.
      if(requiredStages != 0)
      {
        std::stringstream text;
        text << "Pass " << passIdx << " (" << pass.debugName << ") needs the following shader stages to be a complete pipeline: ";
        bool        printedFirst = false;
        size_t      numPrinted   = 0;
        const char* lastPrintedStage;
        for(VkShaderStageFlagBits bit = VK_SHADER_STAGE_VERTEX_BIT; bit <= VK_SHADER_STAGE_MESH_BIT_EXT;
            bit                       = static_cast<VkShaderStageFlagBits>(bit << 1))
        {
          if(requiredStages & bit)
          {
            if(numPrinted > 0)
            {
              text << ", ";
            }
            lastPrintedStage = slangStageToString(slangStageFromVkStage(bit));
            text << lastPrintedStage;
            numPrinted++;
          }
        }

        assert(numPrinted != 0);
        if(numPrinted == 1)
        {
          text << ". Try adding shaders with those shader stages.";
        }
        else
        {
          text << ". Try adding a shader with that shader stage, like this:\n"
                  "[shader(\""
               << lastPrintedStage
               << "\")]\n"
                  "{\n"
                  "\t// your code here"
                  "}";
        }
        diagnostics.push_back({.text = text.str()});
        failed = true;
      }
    }

    if(failed)
    {
      std::stringstream text;
      text << "Here's a list of all the computed passes, in case it helps:\n";
      for(size_t passIdx = 0; passIdx < resources.passes.size(); passIdx++)
      {
        text << passIdx << ": " << resources.passes[passIdx].debugName << "\n";
      }
      diagnostics.push_back({.text = text.str(), .level = Diagnostic::Level::eInfo});
      return VK_ERROR_INITIALIZATION_FAILED;
    }
  }

  // Get additional pass information.
  // For passes with compute shaders, get their workgroup sizes.
  // For passes with vertex shaders, figure out what vertex attributes they need.
  for(SlangUInt passIdx = 0; passIdx < resources.passes.size(); passIdx++)
  {
    auto& pass = resources.passes[passIdx];
    for(SlangUInt entrypointIdx : passIndexToEntrypoints[passIdx])
    {
      const VkShaderStageFlagBits stage = entrypointStages[entrypointIdx];
      if(stage == VK_SHADER_STAGE_COMPUTE_BIT)
      {
        slang::EntryPointReflection* entrypoint = reflection->getEntryPointByIndex(entrypointIdx);
        entrypoint->getComputeThreadGroupSize(3, pass.workgroupSize.data());
        for(size_t i = 0; i < 3; i++)
        {
          pass.workgroupSize[i] = std::max(SlangUInt(1), pass.workgroupSize[i]);  // Correctness check
        }
      }
      else if(stage == VK_SHADER_STAGE_VERTEX_BIT)
      {
        // Look through all of the parameters passed to this function;
        // traverse through structs, and find all the fields marked
        // as varying inputs.
        slang::EntryPointReflection* entrypoint = reflection->getEntryPointByIndex(entrypointIdx);
        slang::IMetadata*            metadata   = entrypointMetadata[entrypointIdx];

        const unsigned int numParameters = entrypoint->getParameterCount();
        for(unsigned int parameterIdx = 0; parameterIdx < numParameters; parameterIdx++)
        {
          traverseForVertexAttributes(pass.vtxAttribInfos, entrypoint->getParameterByIndex(parameterIdx));
        }

        // Remove attributes that weren't actually used. Hopefully this accounts
        // for passes like spirv-opt.
        for(auto& [key, value] : pass.vtxAttribInfos)
        {
          bool        isUsed = true;
          SlangResult result =
              metadata->isParameterLocationUsed(SlangParameterCategory::SLANG_PARAMETER_CATEGORY_VARYING_INPUT, 0, key, isUsed);
          assert(SLANG_SUCCEEDED(result));
          if(!isUsed)
          {
            pass.vtxAttribInfos.erase(key);
          }
        }
      }
    }

    // Assign vertex attributes unique bindings:
    {
      uint32_t nextBinding = 0;
      for(auto& [key, value] : pass.vtxAttribInfos)
      {
        value.binding = nextBinding++;
      }
    }
  }

  // We have a DescriptorWrite for each descriptor in our sets.
  // Figure out which entrypoints use each one; use that to populate
  // Pass::usedTextures, Pass::usedUniformBuffers, Pass::usedStorageBuffers,
  // and the stage mask of descriptorSetUpdates.
  for(DescriptorWrite& descriptorWrite : resources.descriptorSetUpdates)
  {
    std::set<size_t>      passIndices;
    VkShaderStageFlags    shaderStages   = 0;
    VkPipelineStageFlags2 pipelineStages = 0;
    for(SlangUInt entrypointIdx = 0; entrypointIdx < entrypointCount; entrypointIdx++)
    {
      bool              isUsed = false;
      const SlangResult result = entrypointMetadata[entrypointIdx]->isParameterLocationUsed(
          SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT, descriptorWrite.index.set, descriptorWrite.index.binding, isUsed);
      assert(SLANG_SUCCEEDED(result));
      if(isUsed)
      {
        const auto& passes = entrypointIndexToPasses[entrypointIdx];
        passIndices.insert(passes.begin(), passes.end());

        const VkShaderStageFlagBits entrypointStage = entrypointStages[entrypointIdx];
        shaderStages |= entrypointStage;
        pipelineStages |= getPipelineStage2FromShaderStage(entrypointStage);
      }
    }

    descriptorWrite.stages = shaderStages;

    // TODO: Figure out the right layouts for various things; right
    // now we use VK_IMAGE_LAYOUT_GENERAL for everything.
    Pass::UsedResource use{.resourceIndex = descriptorWrite.resourceIndex, .stages = pipelineStages, .layout = VK_IMAGE_LAYOUT_GENERAL};
    switch(descriptorWrite.descriptorType)
    {
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        for(size_t passIdx : passIndices)
        {
          resources.passes[passIdx].usedTextures.push_back(use);
        }
        break;
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        for(size_t passIdx : passIndices)
        {
          resources.passes[passIdx].usedUniformBuffers.push_back(use);
        }
        break;
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        for(size_t passIdx : passIndices)
        {
          resources.passes[passIdx].usedStorageBuffers.push_back(use);
        }
        break;
      default:
        assert(false);
        break;
    }

    // For textures, we set usage flags depending on which descriptors they're
    // written to:
    switch(descriptorWrite.descriptorType)
    {
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        resources.textures[descriptorWrite.resourceIndex].usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
        break;
      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        resources.textures[descriptorWrite.resourceIndex].usage |= VK_IMAGE_USAGE_STORAGE_BIT;
        break;
      default:
        break;
    }
  }

  // Iterate over uniforms; see if we recognize them.
  for(UniformWrite& write : resources.uniformUpdates)
  {
    const std::string_view searchName = removeUniformPrefix(write.name);
    if(strieq(searchName, "frame"))
    {
      write.source = Source::eFrameIndex;
    }
    else if(strieqList(searchName, {"resolution", "screenSize"}))
    {
      write.source = Source::eResolution;
    }
    else if(strieqList(searchName, {"time", "FrameTime"}))
    {
      write.source = Source::eTime;
    }
    else if(strieq(searchName, "view"))
    {
      write.source               = Source::eView;
      resources.hasCameraUniform = true;
    }
    else if(strieqList(searchName, {"viewInverse", "inverseView"}))
    {
      write.source               = Source::eViewInverse;
      resources.hasCameraUniform = true;
    }
    else if(strieqList(searchName, {"proj", "projection"}))
    {
      write.source               = Source::eProj;
      resources.hasCameraUniform = true;
    }
    else if(strieqList(searchName, {"projInverse", "inverseProj", "projectionInverse", "inverseProjection"}))
    {
      write.source               = Source::eProjInverse;
      resources.hasCameraUniform = true;
    }
    else if(strieqList(searchName, {"viewProj", "viewProjection", "projView", "projectionView", "worldToClip"}))
    {
      write.source               = Source::eProjView;
      resources.hasCameraUniform = true;
    }
    else if(strieqList(searchName, {"viewProjInverse", "inverseViewProj", "viewProjectionInverse",
                                    "inverseViewProjection", "projViewInverse", "viewProjInverse",
                                    "projectionViewInverse", "inverseProjectionView", "ClipToWorld"}))
    {
      write.source               = Source::eProjViewInverse;
      resources.hasCameraUniform = true;
    }
    else if(strieq(searchName, "eye"))
    {
      write.source               = Source::eEye;
      resources.hasCameraUniform = true;
    }
    else if(strieq(searchName, "fragCoordToDirection"))
    {
      write.source               = Source::eFragCoordToDirection;
      resources.hasCameraUniform = true;
    }
    else if(strieq(searchName, "mouse"))
    {
      write.source = Source::eMouse;
    }
  }

  // Iterate over textures; try to recognize them.
  struct TextureData
  {
    std::vector<char> data;
    VkFormat          format = VK_FORMAT_UNDEFINED;
  };
  std::vector<TextureData> loadedTextureData(resources.textures.size());
  for(size_t texIdx = 0; texIdx < resources.textures.size(); texIdx++)
  {
    Texture&          tex        = resources.textures[texIdx];
    const std::string searchName = textureVariableToId(tex.name);
    if(strieq(searchName, "frame"))
    {
      tex.source              = Source::eTexFrame;
      resources.texFrameIndex = texIdx;
    }
    else if(strieq(searchName, "depth"))
    {
      tex.source              = Source::eTexDepth;
      resources.texDepthIndex = texIdx;
    }
  }

  // Iterate over storage buffers; TODO: try to recognize them.
  // At the moment we only assign storage buffers that have a stride of 0
  // (e.g. ByteAddressBuffers) a stride of 1.
  for(size_t bufIdx = 0; bufIdx < resources.storageBuffers.size(); bufIdx++)
  {
    StorageBuffer& buf = resources.storageBuffers[bufIdx];
    if(buf.elementStride == 0)
    {
      buf.elementStride = 1;
    }
  }

  // If there's rasterization pipelines, then they write to texFrame and
  // texDepth by default. If these textures don't exist, we create them.
  for(Pass& pass : resources.passes)
  {
    if(pass.shaderStages & kAllRasterStages)
    {
      // Create texFrame if we haven't
      if(resources.texFrameIndex == UNSET_SIZET)
      {
        resources.texFrameIndex = resources.textures.size();
        resources.textures.push_back({.source = Source::eTexFrame});
      }

      if(resources.texDepthIndex == UNSET_SIZET)
      {
        resources.texDepthIndex = resources.textures.size();
        resources.textures.push_back({.source = Source::eTexDepth});
      }

      // And mark that this pass writes to them:
      pass.usedTextures.push_back({.resourceIndex = resources.texFrameIndex,
                                   .stages        = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                   .layout        = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
      pass.usedTextures.push_back({.resourceIndex = resources.texDepthIndex,
                                   .stages = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                                   .layout = VK_IMAGE_LAYOUT_GENERAL});
    }
  }

  // The remaining part is to create the Vulkan objects.
  VkDevice       device         = alloc.getDevice();
  const uint32_t frameCycleSize = app.getFrameCycleSize();
  // First, set up the uniform buffers. We loop over the updates to figure out
  // how large each one is.
  {
    std::vector<size_t> uniformSizes(resources.uniformBuffers.size(), 0);
    for(const UniformWrite& write : resources.uniformUpdates)
    {
      uniformSizes[write.bufferIndex] = std::max(uniformSizes[write.bufferIndex], write.byteOffset + write.byteSize());
    }
    // Now create them:
    for(size_t i = 0; i < resources.uniformBuffers.size(); i++)
    {
      UniformBuffer& ubo = resources.uniformBuffers[i];
      // We have to align up to a multiple of 4 bytes here because of the
      // restriction in vkCmdUpdateBuffer:
      const size_t size = nvutils::align_up(uniformSizes[i], 4);
      ubo.cpuData.resize(size);
      NVVK_FAIL_RETURN(alloc.createBuffer(ubo.buffer, size, VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT));
    }
  }

  // Next, textures.
  // We run and wait on a temp command buffer so that they get initialized.
  // TODO: requiredFormat handling
  {
    // Find all image files
    const std::filesystem::path                            exeDir = nvutils::getExecutablePath().parent_path();
    const std::array<std::filesystem::path, 5>             searchPaths{exeDir, TARGET_NAME "_files/media",
                                                           exeDir / TARGET_EXE_TO_SOURCE_DIRECTORY / "media",
                                                           exeDir / TARGET_EXE_TO_DOWNLOAD_DIRECTORY};
    std::unordered_map<std::string, std::filesystem::path> idToFile;
    for(const std::filesystem::path& searchDir : searchPaths)
    {
      // recursive_directory_iterator throws if the directory doesn't exist,
      // so catch that situation and avoid it:
      if(!std::filesystem::exists(searchDir))
      {
        continue;
      }
      if(!std::filesystem::is_directory(searchDir))
      {
        continue;
      }
      for(const std::filesystem::directory_entry& entry : std::filesystem::recursive_directory_iterator(searchDir))
      {
        if(!entry.is_regular_file())
          continue;
        const std::filesystem::path& path = entry.path();
        const std::filesystem::path  stem = path.stem();
        const std::string            id   = std::string(textureVariableToId(nvutils::utf8FromPath(stem)));
        if(idToFile.count(id) == 0)
        {
          idToFile[id] = path;
        }
      }
    }

    // Load image data
    // TODO: Create samplers dynamically, like in Shadertoy
    NVVK_FAIL_RETURN(samplerPool.acquireSampler(resources.sampler, VkSamplerCreateInfo{.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                                                                       .magFilter = VK_FILTER_LINEAR,
                                                                                       .minFilter = VK_FILTER_LINEAR,
                                                                                       .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                                                                       .anisotropyEnable = true,
                                                                                       .maxAnisotropy    = 4.0f,
                                                                                       .maxLod = VK_LOD_CLAMP_NONE}));
    VkCommandBuffer cmd = app.createTempCmdBuffer();
    for(size_t i = 0; i < resources.textures.size(); i++)
    {
      Texture& tex = resources.textures[i];
      if(tex.source != Source::eUnknown)  // Already know where this texture comes from?
        continue;
      const std::string searchName = std::string(textureVariableToId(tex.name));
      const auto&       it         = idToFile.find(searchName);
      if(it == idToFile.end())  // Have a matching image file?
        continue;
      const std::filesystem::path& path = it->second;
      LOGI("%s -> %s\n", tex.name.c_str(), nvutils::utf8FromPath(path).c_str());
      if(VK_SUCCESS == textureCache.cmdUpload(cmd, tex, path, resources.sampler, staging, app))
      {
        tex.source = Source::eTexFile;
      }
    }

    // Create other textures
    for(size_t i = 0; i < resources.textures.size(); i++)
    {
      Texture& tex = resources.textures[i];
      if(tex.source == Source::eTexFile)
      {
        // Already uploaded; nothing to do
      }
      else if(tex.source == Source::eTexDepth)
      {
        // TODO: Support different formats; we'll need to split to depth and
        // stencil formats.
        tex.format = VK_FORMAT_D24_UNORM_S8_UINT;
        tex.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
      }
      else
      {
        if(tex.format == VK_FORMAT_UNDEFINED)
        {
          tex.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        }
        tex.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
      }
    }

    staging.cmdUploadAppended(cmd);
    app.submitAndWaitTempCmdBuffer(cmd);
    staging.releaseStaging();
  }

  // Set up the descriptor sets and the pipeline layout
  {
    // We do this the linear search way instead of an unordered_map for now
    std::vector<VkDescriptorPoolSize>                      poolSizes;
    std::vector<std::vector<VkDescriptorSetLayoutBinding>> bindings;  // [set][index]
    for(const DescriptorWrite& write : resources.descriptorSetUpdates)
    {
      bool foundPoolSize = false;
      for(VkDescriptorPoolSize& poolSize : poolSizes)
      {
        if(poolSize.type == write.descriptorType)
        {
          poolSize.descriptorCount += frameCycleSize;
          foundPoolSize = true;
          break;
        }
      }
      if(!foundPoolSize)
      {
        poolSizes.emplace_back(write.descriptorType, frameCycleSize);
      }

      const VkDescriptorSetLayoutBinding binding{.binding         = write.index.binding,
                                                 .descriptorType  = write.descriptorType,
                                                 .descriptorCount = 1,
                                                 .stageFlags      = write.stages};

      if(bindings.size() <= write.index.set)
      {
        bindings.resize(static_cast<size_t>(write.index.set) + 1);
      }
      bindings[write.index.set].push_back(binding);
    }

    const VkDescriptorPoolCreateInfo poolInfo{.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                              .maxSets       = static_cast<uint32_t>(frameCycleSize * bindings.size()),
                                              .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
                                              .pPoolSizes    = poolSizes.data()};
    if(poolInfo.maxSets > 0)
    {
      NVVK_FAIL_RETURN(vkCreateDescriptorPool(device, &poolInfo, nullptr, &resources.descriptorPool));
    }

    resources.descriptorSetLayouts.resize(bindings.size());
    for(size_t set = 0; set < bindings.size(); set++)
    {
      const VkDescriptorSetLayoutCreateInfo setLayoutInfo{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                                          .bindingCount = static_cast<uint32_t>(bindings[set].size()),
                                                          .pBindings    = bindings[set].data()};
      NVVK_FAIL_RETURN(vkCreateDescriptorSetLayout(device, &setLayoutInfo, nullptr, &resources.descriptorSetLayouts[set]));
    }

    // Allocate descriptor sets
    const VkDescriptorSetAllocateInfo setAllocInfo{.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                   .descriptorPool     = resources.descriptorPool,
                                                   .descriptorSetCount = static_cast<uint32_t>(bindings.size()),
                                                   .pSetLayouts        = resources.descriptorSetLayouts.data()};
    resources.descriptorSets.resize(frameCycleSize);
    for(size_t cycle = 0; cycle < frameCycleSize; cycle++)
    {
      resources.descriptorSets[cycle].resize(bindings.size());
      if(bindings.size() > 0)
      {
        NVVK_FAIL_RETURN(vkAllocateDescriptorSets(device, &setAllocInfo, resources.descriptorSets[cycle].data()));
      }
    }

    // And pipeline layout
    const VkPipelineLayoutCreateInfo pipelineLayoutInfo{
        .sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = static_cast<uint32_t>(resources.descriptorSetLayouts.size()),
        .pSetLayouts    = resources.descriptorSetLayouts.data(),
        // TODO: Push constants
    };
    NVVK_FAIL_RETURN(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &resources.pipelineLayout));
  }

  // Create the shader module (since we'll use it multiple times)
  const VkShaderModuleCreateInfo moduleInfo{.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                            .codeSize = compiler.getSpirvSize(),
                                            .pCode    = compiler.getSpirv()};
  NVVK_FAIL_RETURN(vkCreateShaderModule(device, &moduleInfo, nullptr, &resources.shaderModule));

  // TODO: Defer capturing pipeline info until the statistics window's opened --
  // it turns out turning this on takes a lot of time in Vulkan!
  constexpr bool kCaptureStatistics = true;

  // Finally, create the pipelines!
  for(size_t passIdx = 0; passIdx < resources.passes.size(); passIdx++)
  {
    Pass&                      pass              = resources.passes[passIdx];
    const std::vector<size_t>& entrypointIndices = passIndexToEntrypoints[passIdx];
    assert(entrypointIndices.size() >= 1);
    std::vector<VkPipelineShaderStageCreateInfo> shaders(entrypointIndices.size());
    for(size_t shader = 0; shader < entrypointIndices.size(); shader++)
    {
      const size_t                    entrypointIndex = entrypointIndices[shader];
      VkPipelineShaderStageCreateInfo shaderInfo{.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                 .stage  = entrypointStages[entrypointIndex],
                                                 .module = resources.shaderModule,
                                                 .pName  = entrypointNames[entrypointIndex].c_str()};
      shaders[shader] = shaderInfo;
    }

    const VkPipelineCreateFlags pipelineFlags =
        VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR | VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
    const VkPipelineCreateFlags2 pipelineFlags2 =
        VK_PIPELINE_CREATE_2_CAPTURE_STATISTICS_BIT_KHR | VK_PIPELINE_CREATE_2_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;

    // Switch based on the pipeline type:
    if((pass.shaderStages & ~kAllComputeStages) == 0)  // Compute
    {
      const VkComputePipelineCreateInfo createInfo{.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                                   .flags  = pipelineFlags,
                                                   .stage  = shaders[0],
                                                   .layout = resources.pipelineLayout};
      NVVK_FAIL_RETURN(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &createInfo, nullptr, &pass.pipeline));
    }
    else if((pass.shaderStages & ~kAllRasterGraphicsStages) == 0)  // Rasterization pipeline with vertex shaders
    {
      const auto& texFrame = resources.textures[resources.texFrameIndex];
      const auto& texDepth = resources.textures[resources.texDepthIndex];

      nvvk::GraphicsPipelineCreator graphicsPipelineCreator;
      graphicsPipelineCreator.flags2 = pipelineFlags2;
      // We create pretty much everything with dynamic state so that we don't
      // have to go through all this again to reconfigure it
      graphicsPipelineCreator.dynamicStateValues                     = kDynamicStates;
      graphicsPipelineCreator.colorFormats                           = {texFrame.format};
      graphicsPipelineCreator.renderingState.depthAttachmentFormat   = texDepth.format;
      graphicsPipelineCreator.renderingState.stencilAttachmentFormat = texDepth.format;

      graphicsPipelineCreator.pipelineInfo.pStages    = shaders.data();
      graphicsPipelineCreator.pipelineInfo.stageCount = static_cast<uint32_t>(shaders.size());
      graphicsPipelineCreator.pipelineInfo.layout     = resources.pipelineLayout;

      // Now, we essentially only need this for vertex binding info.
      for(const auto& [location, info] : pass.vtxAttribInfos)
      {
        VkFormat format = VK_FORMAT_R32G32B32_SFLOAT;
        uint32_t stride = 3 * sizeof(float);
        switch(info.attribute)
        {
          case VertexAttribute::ePosition:
          case VertexAttribute::eNormal:
            format = VK_FORMAT_R32G32B32_SFLOAT;
            stride = 3 * sizeof(float);
            break;
          case VertexAttribute::eTangent:
            format = VK_FORMAT_R32G32B32A32_SFLOAT;
            stride = 4 * sizeof(float);
            break;
          case VertexAttribute::eTexCoord0:
          case VertexAttribute::eTexCoord1:
            format = VK_FORMAT_R32G32_SFLOAT;
            stride = 2 * sizeof(float);
            break;
          case VertexAttribute::eColor:
            format = VK_FORMAT_R8G8B8A8_UNORM;
            stride = 4 * sizeof(uint8_t);
            break;
          case VertexAttribute::eUnknown:
            assert(!"We shouldn't have unknown vertex attributes at this point!");
            break;
        }
        pass.vertexAttributeDescriptions.push_back(VkVertexInputAttributeDescription2EXT{.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                                                                                         .location = location,
                                                                                         .binding  = info.binding,
                                                                                         .format   = format,
                                                                                         .offset   = 0});
        pass.vertexBindingDescriptions.push_back(VkVertexInputBindingDescription2EXT{.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
                                                                                     .binding = info.binding,
                                                                                     .stride  = stride,
                                                                                     .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
                                                                                     .divisor = 1});
      }
      nvvk::GraphicsPipelineState info;
      info.vertexAttributes                     = pass.vertexAttributeDescriptions;
      info.vertexBindings                       = pass.vertexBindingDescriptions;
      info.tessellationState.patchControlPoints = 3;
      NVVK_FAIL_RETURN(graphicsPipelineCreator.createGraphicsPipeline(device, VK_NULL_HANDLE, info, &pass.pipeline));
    }
    else
    {
      LOGE("TODO: Handle mesh and ray tracing pipelines\n");
      return VK_ERROR_FEATURE_NOT_PRESENT;
    }
  }

  return VK_SUCCESS;
}


bool Sample::updateFromSlangCode(bool autosave)
{
  // Clear reflection outputs; don't clear the pipeline until it's validated
  // and we know we're ready to go
  m_reflectionJson.setNull();
  m_editors[+Editor::eTargetDisassembly].ClearText();
  m_editors[+Editor::eBinaryDisassembly].ClearText();
  m_diagnostics.clear();

  // Backup
  if(autosave)
  {
    saveShaderAndConfig(nvutils::getExecutablePath().parent_path() / "autosave.slang", false);
  }

  // Set up search paths
  {
    m_compiler.clearSearchPaths();
    if(!m_currentShaderFile.empty())
    {
      m_compiler.addSearchPaths({m_currentShaderFile.parent_path()});  // Relative to current file
    }
    const std::filesystem::path exeDir = nvutils::getExecutablePath().parent_path();
    m_compiler.addSearchPaths({
        exeDir / TARGET_EXE_TO_SOURCE_DIRECTORY "/examples/Modules",  // Build
        exeDir / TARGET_NAME "_files/examples/Modules",               // Install
        exeDir / TARGET_EXE_TO_NVSHADERS_DIRECTORY "/nvshaders",      // Build
        exeDir / "nvshaders"                                          //  Install
    });
  }

  // Compile
  // A quick note on the try/catch in this block: Slang throws a
  // Slang::InternalError object in case of a fatal internal error.
  // According to the Slang team, this should be treated as if Slang crashed,
  // and it indicates a bug in Slang. See
  // https://github.com/shader-slang/slang/issues/7989 . So, apps where all
  // shaders are controlled by developers shouldn't need this try/catch, and
  // it's not documented in the Slang API.
  // Pragmatically, it is currently possible for humans to crash Slang (it has
  // not been fuzz-tested), and in this case it looks as if vk_slang_editor
  // crashed. So, we'd like to print a better message here and tell users what
  // to do if they run into this. Therefore, although we can't extract the
  // message inside the InternalError like slangc can (because InternalError
  // does not inherit from std::exception or any public classes), we still
  // catch it.
  const std::string slangCode = m_editors[+Editor::eCode].GetText();
  bool              compileOk = true;

  try
  {
    compileOk = m_compiler.loadFromSourceString("editor", slangCode);
  }
  catch(...)
  {
    std::string crashFilename = "slang-crash-" + pathSafeTimeString() + ".slang";
    addDiagnostic({.text = "You've found a way to crash the Slang compiler! Your shader will be saved to " + crashFilename
                           + ". Please create an issue at https://github.com/shader-slang/slang/issues, attach this "
                             "file, and mention that it can be reproduced using " TARGET_NAME ". Thank you!"});
    saveShaderAndConfig(crashFilename, false);
    LOGOK("Shader saved to %s.\n", crashFilename.c_str());
    return false;
  }

  for(Diagnostic& diagnostic : parseDiagnostics(m_compiler.getLastDiagnosticMessage()))
  {
    addDiagnostic(std::move(diagnostic));
  }

  if(!compileOk)
  {
    // Regular incorrect shader, not a Slang crash
    LOGW("Compilation failed.\n");
    return false;
  }

  // Optional: Run spirv-val on the SPIR-V to catch some errors Slang
  // doesn't catch before this shader reaches the driver.
  {
    // Match the settings to those that the Vulkan Validation Layers use:
    spv_validator_options spvOptions = spvValidatorOptionsCreate();
    spvValidatorOptionsSetRelaxStoreStruct(spvOptions, true);
    spvValidatorOptionsSetRelaxBlockLayout(spvOptions, true);
    spvValidatorOptionsSetUniformBufferStandardLayout(spvOptions, true);
    spvValidatorOptionsSetScalarBlockLayout(spvOptions, true);
    spvValidatorOptionsSetWorkgroupScalarBlockLayout(spvOptions, true);
    spvValidatorOptionsSetAllowLocalSizeId(spvOptions, true);
    spvValidatorOptionsSetAllowOffsetTextureOperand(spvOptions, m_ctx.hasExtensionEnabled(VK_KHR_MAINTENANCE_8_EXTENSION_NAME));
    spvValidatorOptionsSetAllowVulkan32BitBitwise(spvOptions, true);
    // The errors aren't much better with friendly names; VVL turns this off
    // for speed.
    spvValidatorOptionsSetFriendlyNames(spvOptions, false);

    spv_const_binary_t binary{.code = m_compiler.getSpirv(), .wordCount = m_compiler.getSpirvSize() / 4};

    spv_diagnostic     spvDiagnostic = nullptr;
    const spv_result_t spvResult = spvValidateWithOptions(m_spirvTools.CContext(), spvOptions, &binary, &spvDiagnostic);
    spvValidatorOptionsDestroy(spvOptions);

    if(spvResult != SPV_SUCCESS)
    {
      if(spvDiagnostic)
      {
        addDiagnostic(Diagnostic{.text = std::string("spirv-val rejected this SPIR-V: ") + spvDiagnostic->error});
        spvDiagnosticDestroy(spvDiagnostic);
      }
      else
      {
        addDiagnostic(Diagnostic{.text = "spirv-val failed, but also didn't produce a diagnostic message!"});
        assert(false);
      }
      LOGW("Compilation failed.\n");
      return false;
    }
  }

  LOGI("Compiled successfully.\n");

  // Get reflection info
  slang::IComponentType* program = m_compiler.getSlangProgram();
  assert(program);
  slang::ProgramLayout* reflection = program->getLayout();
  assert(reflection);
  SlangResult result = reflection->toJson(m_reflectionJson.writeRef());
  if(SLANG_FAILED(result))
  {
    addDiagnostic({.text = "Could not convert reflection info to JSON", .errorCode = static_cast<long long>(result)});
  }

  // Build resources
  {
    std::unique_ptr<Resources> newResources = std::make_unique<Resources>();
    std::vector<Diagnostic>    buildDiagnostics;
    const VkResult buildResult = buildResourcesFromReflection(*newResources, buildDiagnostics, m_compiler, m_alloc,
                                                              m_staging.uploader, m_samplerPool, m_textureCache, *m_app);
    for(Diagnostic& diagnostic : buildDiagnostics)
    {
      addDiagnostic(std::move(diagnostic));
    }

    if(VK_SUCCESS != buildResult)
    {
      destroyResources(m_alloc, std::move(newResources));
      return false;
    }

    // New resources were built successfully!
    // Destroy the old resources; we use a delayed destructor here in case there
    // are frames in flight still using them.
    {
      std::shared_ptr<Resources> oldResources = std::move(m_resources);
      // FIXME: When this code is called from File > Open, we're in the UI;
      // the app lifecycle then immediately frees the old resources.
      m_app->submitResourceFree([alloc = &m_alloc, oldResources]() {  //
        destroyResources(*alloc, std::move(oldResources));            //
      });
    }

    m_resources = std::move(newResources);
  }

  // Finally, reset m_frame to 0 for shaders that assume 0 == uninitialized data:
  m_frame = 0;

  return true;
}
