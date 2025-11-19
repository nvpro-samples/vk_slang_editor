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

// An interactive/live code editor for Slang shaders.
//
// The renderer is mostly powered by Slang shader reflection: the main
// .slang file describes what passes there are, and the renderer configures
// itself around that. For parameters that would be unwieldy to specify in the
// shader, we store a .json file with data.
//
// Because this is a desktop app, we can show much lower-level (especially
// performance-related) info than a standard live coding app.

#include "io_image.h"
#include "io_params.h"
#include "language.h"
#include "resources.h"

#include <nvapp/application.hpp>
#include <nvapp/elem_camera.hpp>
#include <nvapp/elem_profiler.hpp>
#include <nvslang/slang.hpp>
#include <nvutils/camera_manipulator.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/profiler.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/context.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/staging.hpp>
#include <nvvkgltf/scene_vk.hpp>

#include <ImGuiColorTextEdit/TextEditor.h>
#include <spirv-tools/libspirv.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct Diagnostic
{
  std::string text;
  int         line      = 0;  // 0 == don't display a line number
  long long   errorCode = 0;  // 0 == don't display an error code
  enum class Level
  {
    eInfo,
    eWarning,
    eError
  } level = Level::eError;

  std::string displayText;
  void        updateDisplayText();
};

std::vector<Diagnostic> parseDiagnostics(const std::string& diagnostics);

enum class Editor : uint32_t
{
  eCode = 0,
  eTargetDisassembly,  // We use a TextEditor for this because it has syntax highlighting
  eBinaryDisassembly,  // And here because it has Ctrl-F
  eCount,
};
inline constexpr uint32_t operator+(Editor v)  // So we can use + on Editor
{
  return static_cast<uint32_t>(v);
}

TextEditor::Palette getNiasDarkPalette();
TextEditor::Palette getNiasLightPalette();

struct Sample : public nvapp::IAppElement
{
public:
  Sample() {}

  // Entry point for the sample. Most errors we might run into are Vulkan types,
  // so we return that for now.
  VkResult run();
  void     deinit();  // Destroys all resources including the app
  void     onAttach(nvapp::Application* app) override {}
  void     onDetach() override {};
  void     onUIRender() override;
  void     onUIMenu() override;
  void     onFileDrop(const std::filesystem::path& filename) override;
  void     onRender(VkCommandBuffer cmd) override;

  // Saves the current shader to a .slang file and project settings to a .json file.
  // The extension will be added if not present.
  void saveShaderAndConfig(const std::filesystem::path& filename, bool saveThumbnail);

  // GLFW close callback.
  void closeCallback();

private:  // Member variables
  // The Sample sets up its own App and Context, so set them up here.
  // This does mean we have a cyclic reference during execution. During normal
  // teardown, the Sample has the App remove its reference to the sample first.
  nvvk::Context                       m_ctx;
  std::shared_ptr<nvapp::Application> m_app;
  // System info
  VkPhysicalDeviceProperties m_physicalDeviceProperties{};
  bool                       m_rayTracingSupported = false;
  // Performance profiler
  nvutils::ProfilerManager                m_profiler;
  nvutils::ProfilerTimeline*              m_profilerTimeline = nullptr;
  nvvk::ProfilerGpuTimer                  m_profilerGPU;
  std::shared_ptr<nvapp::ElementProfiler> m_profilerGUI;
  // Other GUI state
  std::shared_ptr<nvutils::CameraManipulator> m_cameraControl;
  std::shared_ptr<nvapp::ElementCamera>       m_cameraGUI;
  std::filesystem::path                       m_currentShaderFile = {};
  nvvkgltf::Scene                             m_currentScene;
  nvvkgltf::SceneVk                           m_currentSceneVk;
  ExampleShaderCache                          m_exampleShaderCache;
  uint64_t                                    m_frame = 0;
  // FIXME: This will record the time that the CPU recorded the frame, not the
  // time when the frame is expected to render.
  nvutils::PerformanceTimer m_frameTimer;
  // These are outside of Resources so that they persist between shader compiles.
  std::vector<nvvk::GraphicsPipelineState> m_graphicsPipelineStates;
  nvvk::GraphicsPipelineState&             getOrCreateGraphicsPipelineState(size_t passIndex);
  LanguageServer                           m_languageServer;
  bool                                     m_temporarilyPaused = false;

  // State used for the "Are you sure you want to close/exit?" modal
  struct ModalState
  {
    std::filesystem::path fileToLoad;
    std::string           originalCode;
    enum class Action
    {
      eExit,
      eLoadOrNew
    } action                 = {};
    bool confirmationNeeded  = true;
    bool pleaseOpenExitModal = false;
  } m_modal;

  // Non-dynamic resources for renderer
  nvvk::ResourceAllocator m_alloc;
  nvvk::SamplerPool       m_samplerPool;
  struct Staging  // For asynchronous CPU -> GPU uploads; see usage_StagingUploader
  {
    nvvk::StagingUploader uploader;
    VkSemaphore           timelineSemaphore = VK_NULL_HANDLE;
    uint64_t              timelineValue     = 1;  // TODO: Merge this with m_frame
  } m_staging;
  TextureCache           m_textureCache;
  nvslang::SlangCompiler m_compiler{true};
  spvtools::Context      m_spirvTools = spvtools::Context(SPV_ENV_VULKAN_1_4);  // For disassembly

  // Editor settings
  struct EditorSettings
  {
    float uiScale = 1.0f;  // Doesn't include DPI
    // Vsync also counts but is not listed here
    struct ShowPanes
    {
      bool about             = false;
      bool binaryDisassembly = false;
      bool camera            = true;
      bool code              = true;
      bool diagnostics       = true;
      bool editorSettings    = false;
      bool pipelineStats     = true;
      bool quickReference    = false;
      bool reflection        = false;
      bool parameters        = false;
      bool targetDisassembly = false;
      bool transport         = true;
    } showPanes;
    struct Text
    {
      float lineSpacing          = 1.0;
      bool  autoIndent           = true;
      bool  showWhitespaces      = false;
      bool  showLineNumbers      = true;
      bool  showScrollbarMiniMap = true;
      bool  showMatchingBrackets = true;
      bool  completePairedGlyphs = false;
      bool  middleMousePan       = false;
    } text;
    TextEditor::Palette textPalette = getNiasDarkPalette();
    // TODO: Serialize theme colors
    bool                   reflectionShowJson = true;
    nvgui::SettingsHandler iniIO;
  } m_editorSettings;

  // Shader parameters (this is most of what gets stored in the .json file)
  ShaderParameters m_shaderParams;

  // Text editor widgets
  std::array<TextEditorEx, +Editor::eCount> m_editors{};

  // Cached shader info
  Slang::ComPtr<slang::IBlob> m_reflectionJson;
  std::vector<Diagnostic>     m_diagnostics;
  // Dynamic resources for renderer; these are created based off of the shader.
  std::unique_ptr<Resources> m_resources;

private:  // Private member functions, listed in alphabetical order.
  void addDiagnostic(Diagnostic&& diagnostic)
  {
    diagnostic.updateDisplayText();
    LOGW("%s\n", diagnostic.displayText.c_str());
    m_diagnostics.push_back(std::move(diagnostic));
  }
  // nullptr == there are no textures to display.
  Texture* getDisplayTexture();
  CpuImage getThumbnailImage();
  // UI panes
  void guiPaneAbout();
  void guiPaneBinaryDisassembly();
  void guiPaneCode();
  void guiPaneDiagnostics();
  void guiPaneEditorSettings();
  void guiPanePipelineStats();
  void guiPaneQuickReference();
  void guiPaneReflection();
  void guiPaneParameters();
  void guiPaneTargetDisassembly();
  void guiPaneTransport();

  bool hasCodeChanged() { return m_modal.originalCode != m_editors[+Editor::eCode].GetText(); }
  // Multiplies m_editorSettings.uiScale by a factor; scales font size appropriately.
  void scaleUI(const float factor)
  {
    ImGui::GetIO().FontGlobalScale *= factor;
    m_editorSettings.uiScale *= factor;
  }
  // Handles resources that need to be loaded (or resized but outside of onResize()).
  void syncResources(VkCommandBuffer cmd);
  // Makes all text editors use the settings from EditorSettings::Text.
  // Inexpensive, so can be done every frame.
  void updateTextEditorSettings();
  void updateTextEditorPalettes();
  // Loads Slang code into the text editor and calls updateFromSlangCode(false).
  // diskPath can be empty to indicate this is an in-memory buffer.
  bool openSlangCode(const std::string& text, const std::filesystem::path& diskPath);
  bool openSlangCodeAndConfig(const std::filesystem::path& slangOrJson);
  // Runs the Slang compiler; on success, builds/rebuilds the renderer based on the shader and returns true.
  bool updateFromSlangCode(bool autosave = false);
  // Updates diagnostic markers in the text editor.
  void updateDiagnosticMarkers();
};
