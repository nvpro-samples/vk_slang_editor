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

#include "io_params.h"
#include "utilities.h"

#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvvk/commands.hpp>
#include <nvvk/debug_util.hpp>

#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include <tinygltf/json.hpp>
#include <volk.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <map>
#include <unordered_set>

//-----------------------------------------------------------------------------
// Base85 encoder/decoder

static uint8_t decodeBase85Char(const char c)
{
  return static_cast<uint8_t>(c - '!');
}

static char encodeBase85Char(uint8_t zeroTo84)
{
  return static_cast<char>(zeroTo84) + '!';
}

static std::vector<uint8_t> decodeBase85(const std::span<const char>& text)
{
  std::vector<uint8_t> result;
  size_t               i = 0;
  while(i < text.size())
  {
    const char c0 = text[i];  // MSB
    if(c0 == 'z')
    {
      // See https://en.wikipedia.org/wiki/Ascii85#Encoding
      result.insert(result.end(), {0, 0, 0, 0});
      i += 1;
    }
    else
    {
      const char     c1   = text[i + 1];
      const char     c2   = text[i + 2];
      const char     c3   = text[i + 3];
      const char     c4   = text[i + 4];
      const uint32_t word = decodeBase85Char(c4)                     //
                            + 85 * decodeBase85Char(c3)              //
                            + (85 * 85) * decodeBase85Char(c2)       //
                            + (85 * 85 * 85) * decodeBase85Char(c1)  //
                            + (85 * 85 * 85 * 85) * decodeBase85Char(c0);
      result.insert(result.end(), {static_cast<uint8_t>((word >> 24) & 0xFF),  //
                                   static_cast<uint8_t>((word >> 16) & 0xFF),  //
                                   static_cast<uint8_t>((word >> 8) & 0xFF),   //
                                   static_cast<uint8_t>((word >> 0) & 0xFF)});
      i += 5;
    }
  }

  return result;
}

static std::string encodeBase85(const std::span<const uint8_t>& bytes)
{
  std::string result;
  auto        encodeBase85Word = [&](uint32_t word) {
    if(word == 0)
    {
      result.push_back('z');
    }
    else
    {
      std::array<char, 5> chars{};
      for(int i = 0; i < 5; i++)
      {
        chars[i] = encodeBase85Char(static_cast<uint8_t>(word % 85));
        word /= 85;
      }
      result.insert(result.end(), {chars[4], chars[3], chars[2], chars[1], chars[0]});
    }
  };

  for(size_t i = 0; i + 3 < bytes.size(); i += 4)
  {
    encodeBase85Word((bytes[i] << 24) | (bytes[i + 1] << 16) | (bytes[i + 2] << 8) | (bytes[i + 3] << 0));
  }

  uint32_t     tail  = 0;
  const size_t start = bytes.size() - (bytes.size() % 4);
  for(size_t i = start; i < bytes.size(); i++)
  {
    tail = tail | (bytes[i] << (24 - 8 * (i - start)));
  }
  encodeBase85Word(tail);

  return result;
}

//-----------------------------------------------------------------------------
// JSON utilities

// If `key` doesn't exist, does nothing.
// Otherwise, sets `value` to obj[key].
// There might be a better way to do this; I haven't read the full nlohmann::json docs.
// The main challenge is that we don't immediately want to bail if a single
// field isn't found.
template <class T>
static void jsonGet(const nlohmann::json& obj, const std::string& key, T& result)
{
  assert(obj.is_object());

  const auto& it = obj.find(key);
  if(it == obj.end())
  {
    LOGD("Could not find key `%s`.\n", key.c_str());
    return;
  }

  const nlohmann::json& value = *it;
  if(value.is_null())
  {
    return;
  }

  try
  {
    result = value.template get<T>();
  }
  catch(const std::exception& e)
  {
    LOGW("Found `%s`, but couldn't parse it: %s\n", key.c_str(), e.what());
  }
}

// Implementaiton of JSON serialization/deserialization for StorageBufferParameters;
// See https://github.com/nlohmann/json?tab=readme-ov-file#basic-usage
void to_json(nlohmann::json& j, const StorageBufferParameters& p)
{
  j = nlohmann::json{{"base", p.base}, {"perTile", p.perTile}, {"tileSizeX", p.tileSizeX}, {"tileSizeY", p.tileSizeY}};
}

void from_json(const nlohmann::json& j, StorageBufferParameters& p)
{
  jsonGet(j, "base", p.base);
  jsonGet(j, "perTile", p.perTile);
  jsonGet(j, "tileSizeX", p.tileSizeX);
  jsonGet(j, "tileSizeY", p.tileSizeY);
}

//-----------------------------------------------------------------------------
// Main implementations

VkDeviceSize StorageBufferParameters::computeBufferSize(VkExtent2D resolution, size_t elementStride)
{
  // NOTE: It would be better to do fully checked math here (integer overflow
  // can still occur and will be UB in this case!) instead of naively using
  // int64_ts.
  tileSizeX                  = std::max(1, tileSizeX);
  tileSizeY                  = std::max(1, tileSizeY);
  const int64_t  numTilesX   = (static_cast<int64_t>(resolution.width) + tileSizeX - 1) / tileSizeX;
  const int64_t  numTilesY   = (static_cast<int64_t>(resolution.height) + tileSizeY - 1) / tileSizeY;
  const uint64_t numElements = static_cast<uint64_t>(std::max(int64_t(1),  //
                                                              base + perTile * numTilesX * numTilesY));
  return numElements * elementStride;
}

std::string_view extractDescription(const std::string_view& code)
{
  // For the description, take the first line, if it's a comment.
  std::string_view firstLine(code);
  size_t           marker = firstLine.find_first_of("\r\n");
  firstLine               = firstLine.substr(0, marker);  // This works even if marker is npos
  if(firstLine.size() >= 2 && firstLine[0] == '/' && (firstLine[1] == '/' || firstLine[1] == '*'))
  {
    // Skip past comments
    marker = firstLine.find_first_not_of(" \t/*");
    if(marker != firstLine.npos)
    {
      firstLine = firstLine.substr(marker);
      return firstLine;
    }
  }
  return {};
}

ShaderConfigFile deserializeShaderConfig(const std::filesystem::path& configFile)
{
  const std::string config = nvutils::loadFile(configFile);
  if(config.empty())
  {
    return {};
  }

  ShaderConfigFile  result;
  ShaderParameters& params = result.parameters;

  try
  {
    const nlohmann::json json = nlohmann::json::parse(config);
    jsonGet(json, "clearColorWhen", params.clearColorWhen);
    jsonGet(json, "clearColor", params.clearColor);
    jsonGet(json, "clearDepthStencilWhen", params.clearDepthStencilWhen);
    jsonGet(json, "clearDepth", params.clearDepth);
    jsonGet(json, "clearStencil", params.clearStencil);
    jsonGet(json, "texDisplay", params.texDisplay);
    jsonGet(json, "timeSpeed", params.timeSpeed);
    jsonGet(json, "time", params.time);
    jsonGet(json, "paused", params.paused);
    jsonGet(json, "parameters", params.uniforms);
    jsonGet(json, "storageBuffers", params.storageBuffers);

    jsonGet(json, "description", result.description);

    std::string base85Thumbnail;
    jsonGet(json, "preview", base85Thumbnail);
    if(!base85Thumbnail.empty())
    {
      const std::vector<uint8_t> thumbnailData = decodeBase85(base85Thumbnail);
      CpuImage                 thumbnail;
      if(VK_SUCCESS == decompressImageFromMemory(thumbnail, thumbnailData.data(), thumbnailData.size()))
      {
        result.thumbnail = std::move(thumbnail);
      }
    }
  }
  catch(const std::exception& e)
  {
    LOGW("Caught exception while decoding JSON: %s\n", e.what());
  }

  return result;
}

bool serializeShaderConfig(const std::filesystem::path& outPath, const ShaderConfigFile& config)
{
  nlohmann::json json;
  json["description"] = config.description;

  // Compress the thumbnail image to a medium-quality JPEG
  if(!config.thumbnail.empty())
  {
    const auto& compressed = compressImageToJpeg(config.thumbnail, 80);
    if(compressed.has_value())
    {
      // Then encode as Base85:
      const std::string base85 = encodeBase85(compressed.value());
      json["preview"]          = base85;
    }
  }

  const ShaderParameters& params = config.parameters;
  json["clearColorWhen"]         = params.clearColorWhen;
  json["clearColor"]             = params.clearColor;
  json["clearDepthStencilWhen"]  = params.clearDepthStencilWhen;
  json["clearDepth"]             = params.clearDepth;
  json["clearStencil"]           = params.clearStencil;
  json["texDisplay"]             = params.texDisplay;
  json["timeSpeed"]              = params.timeSpeed;
  json["time"]                   = params.time;
  json["paused"]                 = params.paused;
  json["parameters"]             = params.uniforms;
  json["storageBuffers"]         = params.storageBuffers;

  // TODO: Serialize camera

  const std::string text = json.dump(4);
  try
  {
    std::ofstream file(outPath);
    file.write(text.c_str(), text.size());
  }
  catch(const std::exception& e)
  {
    LOGW("File write to %s failed: %s\n", nvutils::utf8FromPath(outPath).c_str(), e.what());
    return false;
  }
  return true;
}

//-----------------------------------------------------------------------------
// ExampleShaderCache

using file_clock = std::chrono::file_clock;

struct ExampleShaderFile
{
  // We only load file contents once we need to generate the tooltip:
  ShaderConfigFile previewContents;
  std::string      displayName;
  // The modification time of the file the last time we had to generate
  // previewContents.
  // Note that we use min instead of zero-initializing it!
  // On Linux, file_clock's epoch is January 1st, 2174 UTC.
  file_clock::time_point lastUpdated = file_clock::time_point::min();
  // The last time this file's modification time was checked.
  // This is so that if the file is hovered, we're not constantly stat'ing it.
  file_clock::time_point lastChecked = file_clock::time_point::min();
};

struct ExampleShaderFolder
{
  // These use std::map so that they show up alphabetically:
  std::map<std::filesystem::path, ExampleShaderFolder> folders;
  std::map<std::filesystem::path, ExampleShaderFile>   configFiles;
  std::string                                          displayName;
  // The last time this folder was scanned.
  file_clock::time_point lastScanned = file_clock::time_point::min();
};

// If we don't already have a value, get a new one.
template <class T>
static void updateOptional(std::optional<T>& var, std::optional<T>&& maybeNewValue)
{
  if(!var.has_value())
  {
    var = std::move(maybeNewValue);
  }
}

static std::string displayNameFromPath(const std::filesystem::path& path)
{
  return nvutils::utf8FromPath(path.stem().replace_extension(""));
}

struct ExampleShaderCache::Implementation
{
private:
  std::optional<std::filesystem::path> m_rootPath;  // Path to vk_slang_editor/examples; empty if init() couldn't find it
  ExampleShaderFolder m_root;

  // This is probably overkill; we only allocate a single preview image and
  // change it when necessary, so we only use 1 descriptor set.
  nvvk::StagingUploader& m_staging;
  nvapp::Application&    m_dealloc;
  VkCommandPool          m_transientCommandPool = VK_NULL_HANDLE;  // Non-owning
  VkQueue                m_transferQueue        = VK_NULL_HANDLE;  // Non-owning
  VkSampler              m_uiSampler            = VK_NULL_HANDLE;  // Non-owning
  Texture                m_previewImage;
  std::filesystem::path  m_previewImageWasFor;

private:
  bool shouldRescan(const file_clock::time_point now, const file_clock::time_point lastUpdated)
  {
    return now >= lastUpdated + std::chrono::duration<double>(5);
  }

  std::optional<std::filesystem::path> doFolderUI(const std::filesystem::path& folderPath,
                                                  ExampleShaderFolder&         folder,
                                                  const file_clock::time_point now)
  {
    // Do we need to update the folder contents?
    if(shouldRescan(now, folder.lastScanned))
    {
      folder.lastScanned = now;

      // We'll hold sets of what folders and files are actually here so that
      // we avoid destroying and recreating things when they haven't changed
      // but have just been re-scanned.
      std::unordered_set<std::filesystem::path, PathHash> diskFolders;
      std::unordered_set<std::filesystem::path, PathHash> diskConfigFiles;
      // For each item in the folder on disk...
      for(const std::filesystem::directory_entry& dirEntry : std::filesystem::directory_iterator(folderPath))
      {
        const std::filesystem::path& path = dirEntry.path();
        if(dirEntry.is_directory())
        {
          diskFolders.insert(path);
        }
        else if(dirEntry.is_regular_file() && nvutils::extensionMatches(path, ".json"))
        {
          diskConfigFiles.insert(path);
        }
      }

      // Remove folders and files that no longer exist
      std::erase_if(folder.folders, [&](const auto& kvp) { return diskFolders.find(kvp.first) == diskFolders.end(); });
      std::erase_if(folder.configFiles,
                    [&](const auto& kvp) { return diskConfigFiles.find(kvp.first) == diskConfigFiles.end(); });

      // Add new folders and files
      for(const std::filesystem::path& diskFolder : diskFolders)
      {
        if(folder.folders.find(diskFolder) == folder.folders.end())
        {
          folder.folders[diskFolder] = ExampleShaderFolder{.displayName = displayNameFromPath(diskFolder)};
        }
      }
      for(const std::filesystem::path& diskConfigFile : diskConfigFiles)
      {
        if(folder.configFiles.find(diskConfigFile) == folder.configFiles.end())
        {
          folder.configFiles[diskConfigFile] = ExampleShaderFile{.displayName = displayNameFromPath(diskConfigFile)};
        }
      }
    }

    if(folder.folders.empty() && folder.configFiles.empty())
    {
      ImGui::MenuItem("<no folders or files>", nullptr, nullptr, false);
    }

    // Note that we don't return with an optional early here; this is so that
    // when the user clicks on a menu item, the rest of the menu doesn't
    // disappear for a frame.
    std::optional<std::filesystem::path> result;

    // List folders first:
    for(auto& subfolder : folder.folders)
    {
      // NOTE: We could cache these names as well to avoid allocations here
      if(ImGui::BeginMenu(subfolder.second.displayName.c_str()))
      {
        updateOptional(result, doFolderUI(subfolder.first, subfolder.second, now));
        ImGui::EndMenu();
      }
    }

    // Then list files
    for(auto& file : folder.configFiles)
    {
      if(ImGui::MenuItem(file.second.displayName.c_str()))
      {
        updateOptional(result, std::optional(file.first));
      }

      // Hover preview text
      if(ImGui::IsItemHovered())
      {
        ImGui::SetNextWindowSize(ImVec2(std::max(240.f, static_cast<float>(m_previewImage.size.width)), 0));
        if(ImGui::BeginTooltip())
        {
          // We need to reload the config file if it's been updated.
          // Only check if it's been updated every once in a while:
          bool needFileReload = false;
          if(shouldRescan(now, file.second.lastChecked))
          {
            file.second.lastChecked = now;

            const file_clock::time_point trueModificationTime = std::filesystem::last_write_time(file.first);
            if(file.second.lastUpdated != trueModificationTime)
            {
              needFileReload          = true;
              file.second.lastUpdated = trueModificationTime;
            }
          }

          if(needFileReload)
          {
            file.second.previewContents = deserializeShaderConfig(file.first);
          }

          if(!file.second.previewContents.thumbnail.empty())
          {
            // We have an image!

            // Do we need to re-allocate the preview image?
            if(needFileReload || (file.first != m_previewImageWasFor))
            {
              m_previewImageWasFor = file.first;

              // FIXME: When we display the image using ImGui::Image, we'll need
              // to hang on to it until the GPU finishes presenting this frame.
              // Meanwhile, if the CPU reaches here early, we might replace
              // the underlying image too early.
              // I think this means that Texture upload functions need to do
              // delayed destruction of their previous contents, and that
              // the menu UI has to happen after the previous frame completes.
              // For now, we work around it by forcing a WFI here.
              const nvvk::ResourceAllocator& alloc = *m_staging.getResourceAllocator();
              NVVK_CHECK(vkDeviceWaitIdle(alloc.getDevice()));

              VkCommandBuffer cmd;
              NVVK_CHECK(nvvk::beginSingleTimeCommands(cmd, alloc.getDevice(), m_transientCommandPool));
              nvvk::DebugUtil::getInstance().setObjectName(cmd, "CPU->GPU preview image upload");

#ifndef NDEBUG
              m_previewImage.name = nvutils::utf8FromPath(m_previewImageWasFor);
#endif
              cmdUpload(cmd, m_previewImage, file.second.previewContents.thumbnail, m_uiSampler, m_staging, m_dealloc);
              m_staging.cmdUploadAppended(cmd);

              nvvk::BarrierContainer barriers;
              m_previewImage.addTransitionTo(barriers.imageBarriers, VK_IMAGE_LAYOUT_GENERAL, nvvk::INFER_BARRIER_PARAMS, true);
              barriers.cmdPipelineBarrier(cmd, 0);

              NVVK_CHECK(nvvk::endSingleTimeCommands(cmd, alloc.getDevice(), m_transientCommandPool, m_transferQueue));
              m_staging.releaseStaging();
            }

            ImGui::SetCursorPos(ImVec2(0, 0));  // We don't want any frame padding at all
            ImGui::Image((ImTextureID)m_previewImage.getImguiID(), ImVec2(static_cast<float>(m_previewImage.size.width),
                                                                          static_cast<float>(m_previewImage.size.height)));
          }

          ImGui::TextWrapped("%s", file.second.previewContents.description.c_str());

          ImGui::EndTooltip();
        }
      }
    }

    return result;
  }

public:
  Implementation(nvvk::StagingUploader& staging,
                 nvapp::Application&    dealloc,
                 VkCommandPool          nonOwningTransientCommandPool,
                 VkQueue                transferQueue,
                 VkSampler              uiSampler)
      : m_staging(staging)
      , m_dealloc(dealloc)
      , m_transientCommandPool(nonOwningTransientCommandPool)
      , m_transferQueue(transferQueue)
      , m_uiSampler(uiSampler)
  {
    // Try to find our examples/ folder:
    const std::vector<std::filesystem::path> searchPaths = {
        ".",                                                                          // Current directory
        nvutils::getExecutablePath().parent_path() / TARGET_EXE_TO_SOURCE_DIRECTORY,  // Source config
        TARGET_NAME "_files"                                                          // Install config
    };
    const std::filesystem::path path = nvutils::findFile("examples", searchPaths, false);
    if(!path.empty())
    {
      LOGI("Found example directory: %s\n", nvutils::utf8FromPath(path).c_str());
      m_rootPath = std::filesystem::absolute(path);
    }
  }

  ~Implementation() { m_previewImage.deinitResources(*m_staging.getResourceAllocator()); }

  std::optional<std::filesystem::path> doUI()
  {
    std::optional<std::filesystem::path> result;
    if(ImGui::BeginMenu("Examples"))
    {
      if(m_rootPath.has_value())
      {
        result = doFolderUI(m_rootPath.value(), m_root, file_clock::now());
      }
      else
      {
        ImGui::MenuItem("Could not find examples/ folder; please see console window for details.", nullptr, nullptr, false);
      }
      ImGui::EndMenu();
    }
    return result;
  }
};

void ExampleShaderCache::init(nvvk::StagingUploader& staging,
                              nvapp::Application&    dealloc,
                              VkCommandPool          nonOwningTransientCommandPool,
                              VkQueue                transferQueue,
                              VkSampler              uiSampler)
{
  m_implementation = new Implementation(staging, dealloc, nonOwningTransientCommandPool, transferQueue, uiSampler);
}

void ExampleShaderCache::deinit()
{
  if(m_implementation)
  {
    delete m_implementation;
    m_implementation = nullptr;
  }
}

std::optional<std::filesystem::path> ExampleShaderCache::doUI()
{
  return m_implementation->doUI();
}

ExampleShaderCache::~ExampleShaderCache()
{
  assert(!m_implementation && "Forgot to deinit()!");
}
