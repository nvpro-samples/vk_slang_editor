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

#include "sample.h"

#include "gui_reflection.h"
#include "language.h"
#include "utilities.h"

#include <nvapp/elem_default_title.hpp>
#include <nvgui/axis.hpp>
#include <nvgui/camera.hpp>
#include <nvgui/file_dialog.hpp>
#include <nvgui/fonts.hpp>
#include <nvgui/property_editor.hpp>
#include <nvgui/style.hpp>
#include <nvgui/tooltip.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/helpers.hpp>
#include <nvvk/validation_settings.hpp>

#include <GLFW/glfw3.h>
#include <imgui/imgui_internal.h>
#include <roboto/roboto_regular.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image_resize2.h>
#define TINYGLTF_IMPLEMENTATION
#include <tinygltf/tiny_gltf.h>
#include <volk/volk.h>

#include <fstream>
#include <regex>

#ifndef _WIN32
#include <signal.h>
#endif

// Names for the UI panes.
constexpr const char* kUIPaneViewport          = "Viewport";
constexpr const char* kUIPaneAbout             = "About";
constexpr const char* kUIPaneBinaryDisassembly = "Binary Disassembly";
constexpr const char* kUIPaneCamera            = "Camera";
constexpr const char* kUIPaneCode              = "Code";
constexpr const char* kUIPaneDiagnostics       = "Diagnostics";
constexpr const char* kUIPaneEditorSettings    = "Editor Settings";
constexpr const char* kUIPanePerf              = "Performance";
constexpr const char* kUIPanePipelineStats     = "Pipeline Statistics";
constexpr const char* kUIPaneQuickReference    = "Quick Reference";
constexpr const char* kUIPaneReflection        = "Reflection";
constexpr const char* kUIPaneParameters        = "Shader Parameters";
// TODO: Would like to find a better name for this -- it's more like
// 'disassembly for the code format that the target takes' rather than
// shader bytecode disassembly. I.e. if people count instructions here they'll
// probably get the wrong answer with regards to performance.
constexpr const char* kUIPaneTargetDisassembly = "Target Disassembly";
// Like digital audio workstation transport controls
constexpr const char* kUIPaneTransport = "Transport";
// Modal dialogs
constexpr const char* kUIModalShaderLoad = "Confirm Shader Load";
constexpr const char* kUIModalExit       = "Confirm Exit";

// This is a global variable so that the handler for uncaught exceptions can
// access it if needed.
static std::shared_ptr<Sample> s_sample;

constexpr const char* kDefaultComputeShader =
    R"(RWTexture2D<float4> texFrame; // Output texture
uniform float iTime; // In seconds
uniform float2 iResolution; // Screen size

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 thread: SV_DispatchThreadID)
{
	float2 uv = float2(thread) / iResolution;
	float4 color = float4(uv, 0.5 + 0.5 * sin(iTime), 1.0);
	texFrame[thread] = color;
}
)";

void Diagnostic::updateDisplayText()
{
  if(level == Level::eInfo)
  {
    displayText = "Info";
  }
  else if(level == Level::eWarning)
  {
    displayText = "Warning";
  }
  else
  {
    displayText = "Error";
  }

  if(line != 0)
  {
    displayText = displayText + ", line " + std::to_string(line);
  }
  displayText = displayText + ": " + text;
  if(errorCode != 0)
  {
    displayText = displayText + "\n(code " + std::to_string(errorCode) + ")";
  }
}

#define FOR_EACH_EDITOR_COLOR(FUNC)                                                                                    \
  FUNC(text)                                                                                                           \
  FUNC(keyword)                                                                                                        \
  FUNC(declaration)                                                                                                    \
  FUNC(number)                                                                                                         \
  FUNC(string)                                                                                                         \
  FUNC(punctuation)                                                                                                    \
  FUNC(preprocessor)                                                                                                   \
  FUNC(identifier)                                                                                                     \
  FUNC(knownIdentifier)                                                                                                \
  FUNC(comment)                                                                                                        \
  FUNC(background)                                                                                                     \
  FUNC(cursor)                                                                                                         \
  FUNC(selection)                                                                                                      \
  FUNC(whitespace)                                                                                                     \
  FUNC(matchingBracketBackground)                                                                                      \
  FUNC(matchingBracketActive)                                                                                          \
  FUNC(matchingBracketLevel1)                                                                                          \
  FUNC(matchingBracketLevel2)                                                                                          \
  FUNC(matchingBracketLevel3)                                                                                          \
  FUNC(matchingBracketError)                                                                                           \
  FUNC(lineNumber)                                                                                                     \
  FUNC(currentLineNumber)

TextEditor::Palette getNiasDarkPalette()
{
  TextEditor::Palette palette                                  = TextEditor::GetDarkPalette();
  palette[static_cast<size_t>(TextEditor::Color::text)]        = IM_COL32(0xEF, 0xEF, 0xEF, 0xFF);
  palette[static_cast<size_t>(TextEditor::Color::declaration)] = IM_COL32(0x2F, 0xA0, 0x7D, 0xFF);
  palette[static_cast<size_t>(TextEditor::Color::string)]      = IM_COL32(0xC6, 0x75, 0x53, 0xFF);
  palette[static_cast<size_t>(TextEditor::Color::punctuation)] = IM_COL32(0xFF, 0xFF, 0xFF, 0xFF);
  palette[static_cast<size_t>(TextEditor::Color::identifier)]  = IM_COL32(0xB5, 0xE5, 0xFF, 0xFF);
  palette[static_cast<size_t>(TextEditor::Color::comment)]     = IM_COL32(0x54, 0x87, 0x48, 0xFF);
  palette[static_cast<size_t>(TextEditor::Color::background)]  = IM_COL32(0x10, 0x10, 0x10, 0xFF);
  return palette;
}

TextEditor::Palette getNiasLightPalette()
{
  TextEditor::Palette palette                                  = TextEditor::GetLightPalette();
  palette[static_cast<size_t>(TextEditor::Color::text)]        = IM_COL32(0x00, 0x00, 0x00, 0xFF);
  palette[static_cast<size_t>(TextEditor::Color::declaration)] = IM_COL32(0x36, 0x30, 0xD2, 0xFF);
  palette[static_cast<size_t>(TextEditor::Color::string)]      = IM_COL32(0xB6, 0x34, 0x34, 0xFF);
  palette[static_cast<size_t>(TextEditor::Color::identifier)]  = IM_COL32(0x00, 0x00, 0x00, 0xFF);
  palette[static_cast<size_t>(TextEditor::Color::comment)]     = IM_COL32(0x1C, 0x73, 0x00, 0xFF);
  palette[static_cast<size_t>(TextEditor::Color::background)]  = IM_COL32(0xFE, 0xFE, 0xFE, 0xFF);
  return palette;
}


Texture* Sample::getDisplayTexture()
{
  if(UNSET_SIZET != m_shaderParams.texDisplay)
  {
    if(m_shaderParams.texDisplay < m_resources->textures.size())
    {
      return &(m_resources->textures[m_shaderParams.texDisplay]);
    }
    else
    {
      // Pipeline changed and index now out of bounds; fall back to texFrame
      // and forget which texture was selected
      m_shaderParams.texDisplay = UNSET_SIZET;
    }
  }

  if(UNSET_SIZET != m_resources->texFrameIndex)
  {
    return &(m_resources->textures[m_resources->texFrameIndex]);
  }
  return nullptr;
}

static void dockSetup(ImGuiID viewportID)
{
  // Layout
  ImGuiID codeID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Left, 1.f / 3.f, nullptr, nullptr);
  ImGui::DockBuilderDockWindow(kUIPaneCode, codeID);

  // We intentionally squish this against the bottom of the window to present
  // a clean UI, but to say to the user, "there's something here you can move
  // up to see if you want".
  const ImGuiID perfID = ImGui::DockBuilderSplitNode(codeID, ImGuiDir_Down, 0.12f, nullptr, &codeID);
  // FIXME: How can we control the tab bar order so that Perf appears
  // before Pipeline Stats?
  ImGui::DockBuilderDockWindow(kUIPaneCamera, perfID);
  ImGui::DockBuilderDockWindow(kUIPanePerf, perfID);
  ImGui::DockBuilderDockWindow(kUIPanePipelineStats, perfID);

  const ImGuiID transportID = ImGui::DockBuilderSplitNode(codeID, ImGuiDir_Down, 0.07f, nullptr, &codeID);
  ImGui::DockBuilderDockWindow(kUIPaneTransport, transportID);

  const ImGuiID diagID = ImGui::DockBuilderSplitNode(codeID, ImGuiDir_Down, 0.2f, nullptr, nullptr);
  ImGui::DockBuilderDockWindow(kUIPaneDiagnostics, diagID);
}

static void windowCloseCallback(GLFWwindow* window)
{
  s_sample->closeCallback();
}

void Sample::closeCallback()
{
  m_modal.action             = ModalState::Action::eExit;
  m_modal.confirmationNeeded = m_modal.confirmationNeeded && hasCodeChanged();  // If no code change, no check needed
  if(!m_modal.confirmationNeeded)
  {
    return;
  }

  m_modal.pleaseOpenExitModal = true;
  glfwSetWindowShouldClose(m_app->getWindowHandle(), GLFW_FALSE);
}

VkResult Sample::run()
{
  // Vulkan extensions
  VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pipelineExeFeatures{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
  VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR csDerivativeFeatures{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR};
  VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
  VkPhysicalDeviceExtendedDynamicState3FeaturesEXT dynamic3Features{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT};
  VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR barycentricFeatures{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
  VkPhysicalDeviceRobustness2FeaturesEXT robustness2Features{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT};
  VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  nvvk::ContextInitInfo vkSetup{
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions =
          {
              //
              {.extensionName = VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME, .feature = &pipelineExeFeatures},  //
              {.extensionName = VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME, .feature = &csDerivativeFeatures, .required = false},  //
              {.extensionName = VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, .feature = &atomicFloatFeatures, .required = false},  //
              // {.extensionName = VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME},                              //
              {.extensionName = VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME},                                  //
              {.extensionName = VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME, .feature = &dynamic3Features},  //
              {.extensionName = VK_KHR_MAINTENANCE_5_EXTENSION_NAME},                                           //
              // Turned off until more drivers support it:
              // {.extensionName = VK_KHR_MAINTENANCE_8_EXTENSION_NAME, .required = false},                           //
              {.extensionName = VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME, .feature = &barycentricFeatures},  //
              {.extensionName = VK_EXT_ROBUSTNESS_2_EXTENSION_NAME, .feature = &robustness2Features},                 //
              // FIXME: we mainly need this because nvvk::SceneVk uses VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR.
              {.extensionName = VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME},                        //
              {.extensionName = VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, .feature = &asFeatures},  //
          },
      .apiVersion = VK_API_VERSION_1_3,  // We go one minor version newer so this works with RenderDoc as of 2025-07-08
  };
  // Enable Debug Printf
  nvvk::ValidationSettings validationInfo{};
  validationInfo.setPreset(nvvk::ValidationSettings::LayerPresets::eDebugPrintf);
  validationInfo.printf_to_stdout = true;
  vkSetup.instanceCreateInfoExt   = validationInfo.buildPNextChain();
  // TODO: Headless mode
  if(true)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }

  NVVK_FAIL_RETURN(m_ctx.init(vkSetup));
  vkGetPhysicalDeviceProperties(m_ctx.getPhysicalDevice(), &m_physicalDeviceProperties);

  // Memory allocator
  m_alloc.init(VmaAllocatorCreateInfo{
      .physicalDevice = m_ctx.getPhysicalDevice(), .device = m_ctx.getDevice(), .instance = m_ctx.getInstance()});

  // Window + main loop setup
  nvapp::ApplicationCreateInfo appInfo{
      .name           = TARGET_NAME,
      .instance       = m_ctx.getInstance(),
      .device         = m_ctx.getDevice(),
      .physicalDevice = m_ctx.getPhysicalDevice(),
      .queues         = m_ctx.getQueueInfos(),
      .windowSize     = {1760, 990},
      .vSync          = true,
      // Set up the dock positions for the menus
      .dockSetup = dockSetup,
  };
  m_app = std::make_shared<nvapp::Application>();
  m_languageServer.init(m_editors[+Editor::eCode]);
  // Editor settings handler. We have to do this here because Application()
  // initializes ImGui and init() loads INI settings.
  {
    nvgui::SettingsHandler& iniIO = m_editorSettings.iniIO;
    EditorSettings::Text&   t     = m_editorSettings.text;
    iniIO.setHandlerName(TARGET_NAME);
    iniIO.setSetting("uiScale", &m_editorSettings.uiScale);

    iniIO.setSetting("editor.lineSpacing", &t.lineSpacing);
    iniIO.setSetting("editor.autoIndent", &t.autoIndent);
    iniIO.setSetting("editor.showWhitespaces", &t.showWhitespaces);
    iniIO.setSetting("editor.showLineNumbers", &t.showLineNumbers);
    iniIO.setSetting("editor.showScrollbarMiniMap", &t.showScrollbarMiniMap);
    iniIO.setSetting("editor.showMatchingBrackets", &t.showMatchingBrackets);
    iniIO.setSetting("editor.completePairedGlyphs", &t.completePairedGlyphs);
    iniIO.setSetting("editor.middleMousePan", &t.middleMousePan);

#define EDITOR_COLOR(NAME)                                                                                             \
  iniIO.setSetting("editor.palette." #NAME, &m_editorSettings.textPalette[static_cast<size_t>(TextEditor::Color::NAME)]);
    FOR_EACH_EDITOR_COLOR(EDITOR_COLOR);
#undef EDITOR_COLOR

    for(size_t imguiColorIdx = 0; imguiColorIdx < ImGuiCol_COUNT; imguiColorIdx++)
    {
      // I feel like this could be better
      const std::string base = std::string("ui.palette.") + std::to_string(imguiColorIdx);
      ImVec4&           pCol = ImGui::GetStyle().Colors[imguiColorIdx];
      iniIO.setSetting(base + ".x", &pCol.x);
      iniIO.setSetting(base + ".y", &pCol.y);
      iniIO.setSetting(base + ".z", &pCol.z);
      iniIO.setSetting(base + ".w", &pCol.w);
    }

    LanguageServer::Settings& pSettings = m_languageServer.settings();
    iniIO.setSetting("languageServer.enabled", &pSettings.enabled);
    iniIO.setSetting("languageServer.restartInterval", &pSettings.restartInterval);

    iniIO.setSetting("showPanes.about", &m_editorSettings.showPanes.about);
    iniIO.setSetting("showPanes.binaryDisassembly", &m_editorSettings.showPanes.binaryDisassembly);
    iniIO.setSetting("showPanes.camera", &m_editorSettings.showPanes.camera);
    iniIO.setSetting("showPanes.code", &m_editorSettings.showPanes.code);
    iniIO.setSetting("showPanes.diagnostics", &m_editorSettings.showPanes.diagnostics);
    iniIO.setSetting("showPanes.editorSettings", &m_editorSettings.showPanes.editorSettings);
    iniIO.setSetting("showPanes.pipelineStats", &m_editorSettings.showPanes.pipelineStats);
    iniIO.setSetting("showPanes.quickReference", &m_editorSettings.showPanes.quickReference);
    iniIO.setSetting("showPanes.reflection", &m_editorSettings.showPanes.reflection);
    iniIO.setSetting("showPanes.targetDisassembly", &m_editorSettings.showPanes.targetDisassembly);
    iniIO.setSetting("showPanes.transport", &m_editorSettings.showPanes.transport);

    iniIO.setSetting("reflectionShowJson", &m_editorSettings.reflectionShowJson);
    iniIO.addImGuiHandler();
  }
  //
  m_app->init(appInfo);

  // Attach ourselves as an element to the GUI.
  m_app->addElement(s_sample);

  // Add an element that automatically updates the title with the current
  // size and FPS.
  m_app->addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>());

  // Add a basic camera control.
  m_cameraControl = std::make_shared<nvutils::CameraManipulator>();
  m_cameraGUI     = std::make_shared<nvapp::ElementCamera>();
  m_cameraGUI->setCameraManipulator(m_cameraControl);
  m_app->addElement(m_cameraGUI);

  // CPU/GPU profiler
  m_profilerTimeline = m_profiler.createTimeline({.name = "Primary"});
  m_profilerGPU.init(m_profilerTimeline, m_ctx.getDevice(), m_ctx.getPhysicalDevice(), m_app->getQueue(0).familyIndex, true);
  nvapp::ElementProfiler::ViewSettings perfViewSettings{.name = kUIPanePerf};
  m_profilerGUI = std::make_shared<nvapp::ElementProfiler>(
      &m_profiler, std::make_shared<nvapp::ElementProfiler::ViewSettings>(std::move(perfViewSettings)));
  m_app->addElement(m_profilerGUI);

  // Callback for close button
  glfwSetWindowCloseCallback(m_app->getWindowHandle(), windowCloseCallback);

  // Sampler pool
  m_samplerPool.init(m_ctx.getDevice());

  // Staging buffer uploader
  m_staging.uploader.init(&m_alloc);
  const VkSemaphoreTypeCreateInfo timelineSemaphoreInfo{.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                                        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
                                                        .initialValue  = 0};
  const VkSemaphoreCreateInfo semaphoreInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &timelineSemaphoreInfo};
  vkCreateSemaphore(m_ctx.getDevice(), &semaphoreInfo, nullptr, &m_staging.timelineSemaphore);

  // GUI
  {
    const float uiScale = m_editorSettings.uiScale;
    scaleUI(uiScale);
    m_editorSettings.uiScale = uiScale;
  }

  // "Examples" menu button
  {
    VkSampler linearSampler;
    NVVK_CHECK(m_samplerPool.acquireSampler(linearSampler, VkSamplerCreateInfo{.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                                                               .magFilter = VK_FILTER_LINEAR,
                                                                               .minFilter = VK_FILTER_LINEAR}));
    m_exampleShaderCache.init(m_staging.uploader, *m_app, m_app->getCommandPool(), m_app->getQueue(0).queue, linearSampler);
  }

  // Setup text editor
  updateTextEditorSettings();
  updateTextEditorPalettes();
  m_editors[+Editor::eCode].SetLanguage(getSlangLanguageInfo());
  m_editors[+Editor::eTargetDisassembly].SetReadOnlyEnabled(true);
  m_editors[+Editor::eBinaryDisassembly].SetReadOnlyEnabled(true);
  m_languageServer.init(m_editors[+Editor::eCode]);

  // Default compiler settings
  m_compiler.defaultTarget();
  m_compiler.defaultOptions();
  m_compiler.addOption({slang::CompilerOptionName::DebugInformation, {slang::CompilerOptionValueKind::Int, 1}});
  m_compiler.addOption({slang::CompilerOptionName::Optimization, {slang::CompilerOptionValueKind::Int, 0}});

  // Compile initial shader
  const bool initialCompileOk = openSlangCode(kDefaultComputeShader, "");
  if(!initialCompileOk)
  {
    LOGE("Failed to create a renderer using the initial Slang shader! This indicates a problem with the sample.\n");
#ifndef NDEBUG  // In debug mode we'd like to see what's going on
    return VK_ERROR_INVALID_SHADER_NV;
#endif
  }

  // Load shaderball;
  // this will be moved else soon so we can handle changing the scene on
  // the fly.
  m_currentSceneVk.init(&m_alloc, &m_samplerPool);
  {
    const std::filesystem::path              exeDir          = nvutils::getExecutablePath().parent_path();
    const std::vector<std::filesystem::path> meshSearchPaths = {
        exeDir,                                             // Next to .exe
        exeDir / TARGET_EXE_TO_SOURCE_DIRECTORY / "media",  // Build media path
        exeDir / TARGET_NAME "_files/media",                // Install media path
        exeDir / TARGET_EXE_TO_DOWNLOAD_DIRECTORY,          // Build downloaded resources
        exeDir / "resources"                                // Install downloaded resources
    };
    const std::filesystem::path shaderballPath = nvutils::findFile("shaderball.glb", meshSearchPaths);
    if(shaderballPath.empty())
    {
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    onFileDrop(shaderballPath);
  }

  // Main loop
  m_app->run();

  return VK_SUCCESS;
}

void Sample::deinit()
{
  NVVK_CHECK(vkDeviceWaitIdle(m_ctx.getDevice()));
  m_currentSceneVk.deinit();
  destroyResources(m_alloc, std::move(m_resources));

  vkDestroySemaphore(m_ctx.getDevice(), m_staging.timelineSemaphore, nullptr);
  m_exampleShaderCache.deinit();
  m_staging.uploader.deinit();

  m_samplerPool.deinit();
  m_profilerGPU.deinit();
  m_profiler.destroyTimeline(m_profilerTimeline);

  m_languageServer.deinit();
  m_app->deinit();
  m_alloc.deinit();
  m_ctx.deinit();
}

void Sample::onUIRender()
{
  // This code for the main viewport tells ImGui to composite our color image
  // to the screen once the window class calls ImGui::Render().
  {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
    if(ImGui::Begin(kUIPaneViewport))
    {
      const Texture* displayTex = getDisplayTexture();
      if(displayTex)
      {
        const ImTextureID id = reinterpret_cast<ImTextureID>(displayTex->getImguiID());
        if(id)
        {
          // Display the G-Buffer image
          ImGui::Image(id, ImGui::GetContentRegionAvail());
        }
      }

      if(m_resources->hasCameraUniform)
      {
        // Add a coordinate gizmo
        nvgui::Axis(m_cameraControl->getViewMatrix());
      }
    }
    ImGui::End();
    ImGui::PopStyleVar();
  }

  // About pane
  if(m_editorSettings.showPanes.about)
  {
    ImGui::SetNextWindowSize(ImVec2(400, 0.0f), ImGuiCond_FirstUseEver);
    if(ImGui::Begin(kUIPaneAbout, &m_editorSettings.showPanes.about))
    {
      guiPaneAbout();
    }
    ImGui::End();
  }

  // We only show the Binary Disassembly pane if there are any disassemblies
  // that can be shown. For SPIR-V, see the Target Disassembly pane.
  if(m_resources->hadAnyPipelineRepresentations && m_editorSettings.showPanes.binaryDisassembly)
  {
    ImGui::SetNextWindowSize(ImVec2(400, 650), ImGuiCond_FirstUseEver);
    if(ImGui::Begin(kUIPaneBinaryDisassembly, &m_editorSettings.showPanes.binaryDisassembly))
    {
      guiPaneBinaryDisassembly();
    }
    ImGui::End();
  }

  // Camera pane
  if(m_editorSettings.showPanes.camera && m_resources->hasCameraUniform)
  {
    if(ImGui::Begin(kUIPaneCamera, &m_editorSettings.showPanes.camera))
    {
      nvgui::CameraWidget(m_cameraControl);
    }
    ImGui::End();
  }

  // Code pane
  if(m_editorSettings.showPanes.code)
  {
    if(ImGui::Begin(kUIPaneCode, &m_editorSettings.showPanes.code))
    {
      guiPaneCode();
    }
    ImGui::End();
  }

  // Diagnostics pane
  if(m_editorSettings.showPanes.diagnostics)
  {
    if(ImGui::Begin(kUIPaneDiagnostics, &m_editorSettings.showPanes.diagnostics))
    {
      guiPaneDiagnostics();
    }
    ImGui::End();
  }

  // Editor Settings pane
  if(m_editorSettings.showPanes.editorSettings)
  {
    ImGui::SetNextWindowSize(ImVec2(400, 650), ImGuiCond_FirstUseEver);
    if(ImGui::Begin(kUIPaneEditorSettings, &m_editorSettings.showPanes.editorSettings))
    {
      guiPaneEditorSettings();
    }
    ImGui::End();
  }

  // Pipeline Statistics pane
  if(m_editorSettings.showPanes.pipelineStats)
  {
    if(ImGui::Begin(kUIPanePipelineStats, &m_editorSettings.showPanes.pipelineStats))
    {
      guiPanePipelineStats();
    }
    ImGui::End();
  }

  // Quick Reference pane
  if(m_editorSettings.showPanes.quickReference)
  {
    ImGui::SetNextWindowSize(ImVec2(400, 650), ImGuiCond_FirstUseEver);
    if(ImGui::Begin(kUIPaneQuickReference, &m_editorSettings.showPanes.quickReference))
    {
      guiPaneQuickReference();
    }
    ImGui::End();
  }

  // Reflection pane
  if(m_editorSettings.showPanes.reflection)
  {
    ImGui::SetNextWindowSize(ImVec2(400, 650), ImGuiCond_FirstUseEver);
    if(ImGui::Begin(kUIPaneReflection, &m_editorSettings.showPanes.reflection))
    {
      guiPaneReflection();
    }
    ImGui::End();
  }

  // Shader Parameters pane
  if(m_editorSettings.showPanes.parameters)
  {
    ImGui::SetNextWindowSize(ImVec2(400, 650), ImGuiCond_FirstUseEver);
    if(ImGui::Begin(kUIPaneParameters, &m_editorSettings.showPanes.parameters))
    {
      guiPaneParameters();
    }
    ImGui::End();
  }

  // Target Disassembly pane
  if(m_editorSettings.showPanes.targetDisassembly)
  {
    ImGui::SetNextWindowSize(ImVec2(400, 650), ImGuiCond_FirstUseEver);
    if(ImGui::Begin(kUIPaneTargetDisassembly, &m_editorSettings.showPanes.targetDisassembly))
    {
      guiPaneTargetDisassembly();
    }
    ImGui::End();
  }

  // Transport pane
  if(m_editorSettings.showPanes.transport)
  {
    if(ImGui::Begin(kUIPaneTransport, &m_editorSettings.showPanes.transport))
    {
      guiPaneTransport();
    }
    ImGui::End();
  }
}

void Sample::guiPaneAbout()
{
  ImGui::PushFont(nvgui::getDefaultFont(), 30.0f);
  ImGui::TextAligned(0.5f, -1.0f, TARGET_NAME);
  ImGui::PopFont();

  ImGui::TextAligned(0.5f, -1.0f, "version " TARGET_VERSION_STRING);
  ImGui::TextAligned(0.5f, -1.0f, "A shader-driven livecoding tool");
  ImGui::SetCursorPosY(ImGui::GetCursorPosY() - ImGui::GetStyle().ItemSpacing.y);
  ImGui::TextAligned(0.5f, -1.0f, "using Vulkan and the Slang shading language.");

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  ImGui::TextWrapped(TARGET_NAME
                     " is part of the NVIDIA DesignWorks Samples, a collection of projects created to help developers "
                     "learn about GPU programming. You can download the source code for " TARGET_NAME
                     " and other samples online:");

  const char* nvproSamplesUrl = "https://github.com/nvpro-samples";
  const float linkWidth       = ImGui::CalcTextSize(nvproSamplesUrl).x;
  ImGui::SetCursorPosX((ImGui::GetContentRegionAvail().x - linkWidth) * .5f);
  ImGui::TextLinkOpenURL(nvproSamplesUrl);

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();
  ImGui::TextWrapped("Written by Nia Bickford.");

  ImGui::TextWrapped(
      "Thanks to Martin-Karl Lefrançois, Pyarelal Knowles, Tristan Lorach, and the Slang team for beta-testing this "
      "tool. And to you, for reading these credits!");
}

void Sample::guiPaneBinaryDisassembly()
{
  const auto& passes = m_resources->passes;
  if(passes.empty())
  {
    ImGui::TextWrapped("No passes to display. Try adding a shader and compiling.");
    return;
  }

  static size_t selectedPass = 0;
  TextEditor&   editor       = m_editors[+Editor::eBinaryDisassembly];
  bool          needNewText  = editor.IsEmpty();
  selectedPass               = std::min(selectedPass, passes.size() - 1);
  if(passes.size() > 1)  // Don't ask if there's only one pass
  {
    if(ImGui::BeginCombo("Pipeline", passes[selectedPass].debugName.c_str()))
    {
      for(size_t i = 0; i < passes.size(); i++)
      {
        if(ImGui::Selectable(passes[i].debugName.c_str(), i == selectedPass))
        {
          selectedPass = i;
          needNewText  = true;
        }
      }
      ImGui::EndCombo();
    }
  }
  const auto& pass = passes[selectedPass];

  const auto& executables = pass.pipelineRepresentations;
  if(executables.empty())
  {
    ImGui::TextWrapped("No pipeline executables.");
    return;
  }

  static size_t selectedExecutable = 0;
  selectedExecutable               = std::min(selectedExecutable, executables.size() - 1);
  if(executables.size() > 1)  // Don't ask if there's only one executable
  {
    if(ImGui::BeginCombo("Executable", std::to_string(selectedExecutable).c_str()))
    {
      for(size_t i = 0; i < executables.size(); i++)
      {
        if(ImGui::Selectable(std::to_string(i).c_str(), i == selectedExecutable))
        {
          selectedExecutable = i;
          needNewText        = true;
        }
      }
      ImGui::EndCombo();
    }
  }

  const auto& representations = executables[selectedExecutable];
  if(representations.empty())
  {
    ImGui::TextWrapped("No pipeline representations.");
    return;
  }

  static size_t selectedRepresentation = 0;
  selectedRepresentation               = std::min(selectedRepresentation, representations.size() - 1);
  if(representations.size() > 1)  // Don't ask if there's only one representation
  {
    if(ImGui::BeginCombo("Representation", representations[selectedRepresentation].name))
    {
      for(size_t i = 0; i < representations.size(); i++)
      {
        if(ImGui::Selectable(representations[i].name, i == selectedRepresentation))
        {
          selectedRepresentation = i;
          needNewText            = true;
        }
      }
      ImGui::EndCombo();
    }
  }

  const VkPipelineExecutableInternalRepresentationKHR& repr = representations[selectedRepresentation];
  if(needNewText)
  {
    if(repr.isText)
    {
      editor.SetText(std::string_view(reinterpret_cast<const char*>(repr.pData), repr.dataSize));
    }
    else
    {
      // Format into groups of 16 bytes for now
      std::stringstream display;
      const uint8_t*    bytes = reinterpret_cast<const uint8_t*>(repr.pData);
      const char*       hex   = "0123456789ABCDEF";
      for(size_t i = 0; i < repr.dataSize; i++)
      {
        const uint8_t byte = bytes[i];
        if(i > 0 && ((i % 16) == 0))
        {
          display << '\n';
        }
        display << hex[byte >> 4] << hex[byte & 15] << ' ';
      }
      editor.SetText(display.str());
    }
  }

  ImGui::PushFont(nvgui::getMonospaceFont());
  editor.Render(kUIPaneBinaryDisassembly);
  ImGui::PopFont();
}

void Sample::guiPaneCode()
{
  bool compile = ImGui::IsKeyPressed(ImGuiKey_F3);

  ImGui::PushFont(nvgui::getMonospaceFont());
  const char*  buttonText   = "Compile (F3)";
  const ImVec2 textSize     = ImGui::CalcTextSize(buttonText, nullptr);
  const float  buttonHeight = textSize.y + 4.0f * ImGui::GetStyle().FramePadding.y;

  m_languageServer.interceptKeys();
  m_editors[+Editor::eCode].Render(kUIPaneCode, ImVec2(0.0, -buttonHeight));
  bool isCodeEditorFocused = ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows);
  ImGui::PopFont();

  ImGui::Spacing();
  {
    compile = ImGui::Button("Compile (F3)") || compile;
  }

  if(compile)
  {
    updateFromSlangCode(true);
    updateDiagnosticMarkers();
  }

  // Language server processing
  m_languageServer.doUI(isCodeEditorFocused);
}

void Sample::guiPaneDiagnostics()
{
  if(m_diagnostics.empty())
  {
    ImGui::Text("Shaders compiled successfully.");
  }
  else
  {
    ImGui::PushFont(nvgui::getMonospaceFont());
    for(size_t i = 0; i < m_diagnostics.size(); i++)
    {
      ImGui::PushID(static_cast<int>(i));
      const Diagnostic& diagnostic = m_diagnostics[i];
      // TODO: Can we replace this with GetContentRegionAvail?
      const float  wrapWidth = ImGui::CalcWrapWidthForPos(ImGui::GetCursorScreenPos(), 0.0f);
      const ImVec2 textSize  = ImGui::CalcTextSize(diagnostic.displayText.c_str(), nullptr, false, wrapWidth);
      const ImVec2 cursorPos = ImGui::GetCursorPos();
      ImGui::SetNextItemAllowOverlap();
      if(ImGui::Selectable("##diagnostic", false, ImGuiSelectableFlags_AllowOverlap, textSize))
      {
        m_editors[+Editor::eCode].ScrollToLine(diagnostic.line, TextEditor::Scroll::alignMiddle);
      }
      ImGui::SetCursorPos(cursorPos);
      ImGui::TextWrapped("%s", diagnostic.displayText.c_str());
      ImGui::PopID();
    }
    ImGui::PopFont();
  }
}

void Sample::guiPaneEditorSettings()
{
  // We use nvgui::PropertyEditor for the custom nvpro-samples ImGui style.
  namespace PE = nvgui::PropertyEditor;
  PE::begin();

  {
    float uiScale = m_editorSettings.uiScale;
    PE::InputFloat("UI scale", &uiScale);
    if(uiScale != m_editorSettings.uiScale && uiScale != 0.0f)
    {
      scaleUI(uiScale / m_editorSettings.uiScale);
    }
  }

  {
    int                        selectedTheme = 0;
    std::array<const char*, 3> comboThemes{"Select...", "Light", "Dark"};
    if(PE::Combo("Theme", &selectedTheme, comboThemes.data(), static_cast<int>(comboThemes.size()), -1,
                 "Selecting a theme will overwrite all colors."))
    {
      if(selectedTheme == 1)
      {
        ImGui::StyleColorsLight();
        m_editorSettings.textPalette = getNiasLightPalette();
      }
      else if(selectedTheme == 2)
      {
        nvgui::setStyle(false);
        m_editorSettings.textPalette = getNiasDarkPalette();
      }
      updateTextEditorPalettes();
    }
  }

  if(PE::treeNode("Code Colors"))
  {
    bool changed     = false;
    auto handleColor = [&](TextEditor::Color e, const char* name) {
      const size_t i   = static_cast<size_t>(e);
      ImVec4       col = ImGui::ColorConvertU32ToFloat4(m_editorSettings.textPalette[i]);
      if(PE::ColorEdit4(name, reinterpret_cast<float*>(&col), ImGuiColorEditFlags_AlphaBar))
      {
        changed                         = true;
        m_editorSettings.textPalette[i] = ImGui::ColorConvertFloat4ToU32(col);
      }
    };
#define EDITOR_COLOR(NAME) handleColor(TextEditor::Color::NAME, #NAME);
    FOR_EACH_EDITOR_COLOR(EDITOR_COLOR)
#undef EDITOR_COLOR

    if(changed)
    {
      updateTextEditorPalettes();
    }

    PE::treePop();
  }

  if(PE::treeNode("UI Colors"))
  {
    ImGuiStyle& style = ImGui::GetStyle();
    for(ImGuiCol i = 0; i < ImGuiCol_COUNT; i++)
    {
      PE::ColorEdit4(ImGui::GetStyleColorName(i), reinterpret_cast<float*>(&style.Colors[i]), ImGuiColorEditFlags_AlphaBar);
    }

    PE::treePop();
  }

  if(PE::treeNode("Code Editor"))
  {
    EditorSettings::Text& t = m_editorSettings.text;
    PE::InputFloat("Line spacing", &t.lineSpacing);
    PE::Checkbox("Auto-indent", &t.autoIndent);
    PE::Checkbox("Show whitespace", &t.showWhitespaces);
    PE::Checkbox("Show line numbers", &t.showLineNumbers);
    PE::Checkbox("Show scrollbar mini-map", &t.showScrollbarMiniMap);
    PE::Checkbox("Show matching brackets", &t.showMatchingBrackets);
    PE::Checkbox("Complete paired glyphs", &t.completePairedGlyphs);
    PE::Checkbox("Middle mouse pans instead of scrolls", &t.middleMousePan);
    updateTextEditorSettings();

    PE::treePop();
  }

  if(PE::treeNode("Language Server"))
  {
    LanguageServer::Settings settings = m_languageServer.settings();  // Note copy
    PE::Checkbox("Enabled", &settings.enabled);
    PE::InputFloat("Minimum restart interval", &settings.restartInterval, 0.0f, 0.0f, "%.1f", 0,
                   "If slangd crashes, how many seconds should vk_slang_editor wait before starting it again?");
    m_languageServer.settings(&settings);
    PE::treePop();
  }

  PE::end();
}

void Sample::guiPanePipelineStats()
{
  m_resources->updatePipelineStats(m_ctx.getDevice());
  if(m_resources->passes.empty())
  {
    ImGui::TextWrapped("No passes to display. Try adding a shader and compiling.");
    return;
  }

  const auto&   passes       = m_resources->passes;
  static size_t selectedPass = 0;
  selectedPass               = std::min(selectedPass, passes.size() - 1);
  if(passes.size() > 1)  // Don't ask if there's only one pass
  {
    if(ImGui::BeginCombo("Pipeline", passes[selectedPass].debugName.c_str()))
    {
      for(size_t i = 0; i < passes.size(); i++)
      {
        if(ImGui::Selectable(passes[i].debugName.c_str(), i == selectedPass))
        {
          selectedPass = i;
        }
      }
      ImGui::EndCombo();
    }
  }
  const auto& pass = passes[selectedPass];

  if(pass.pipelineProperties.size() == 0)
  {
    ImGui::TextWrapped("No pipeline statistics to display. That's odd.");
    return;
  }

  static size_t selectedExecutable = 0;
  selectedExecutable               = std::min(selectedExecutable, pass.pipelineProperties.size() - 1);
  if(pass.pipelineProperties.size() > 1)  // Don't ask if there's only one executable
  {
    if(ImGui::BeginCombo("Executable", std::to_string(selectedExecutable).c_str()))
    {
      for(size_t i = 0; i < pass.pipelineProperties.size(); i++)
      {
        if(ImGui::Selectable(std::to_string(i).c_str(), i == selectedExecutable))
        {
          selectedExecutable = i;
        }
      }
      ImGui::EndCombo();
    }
  }

  const VkPipelineExecutablePropertiesKHR&             props = pass.pipelineProperties[selectedExecutable];
  const std::vector<VkPipelineExecutableStatisticKHR>& stats = pass.pipelineStatistics[selectedExecutable];
  ImGui::Text("%s (%s)", props.name, props.description);
  ImGui::Text("Subgroup Size: %u", props.subgroupSize);
  for(const VkPipelineExecutableStatisticKHR& stat : stats)
  {
    // "Local Memory Size" is a special case.
    if(strcmp(stat.name, "Local Memory Size") == 0)
    {
      ImGui::Text("%s: %" PRIu64 " high, %" PRIu64 " low", stat.name, stat.value.u64 >> 32, stat.value.u64 & 0xFFFF'FFFF);
    }
    else
    {
      switch(stat.format)
      {
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
          ImGui::Text("%s: %s", stat.name, (stat.value.b32 ? "true" : "false"));
          break;
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
          ImGui::Text("%s: %" PRId64, stat.name, stat.value.i64);
          break;
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
          ImGui::Text("%s: %" PRIu64, stat.name, stat.value.u64);
          break;
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR:
          ImGui::Text("%s: %f", stat.name, stat.value.f64);
          break;
      }
    }
    nvgui::tooltip(stat.description, true, 0.F);
  }
}

void Sample::guiPaneQuickReference()
{
  ImGui::TextWrapped(
      "This window lists the most common Slang features and syntax. For the complete reference, please see the "
      "Standard Modules Reference at https://docs.shader-slang.org.");

  auto header = [](const char* text) {
    ImGui::PushFont(nvgui::getDefaultFont(), 30.0f);
    ImGui::TextWrapped("%s", text);
    ImGui::PopFont();
    ImGui::Separator();
  };

  header("vk_slang_editor specific:");
  ImGui::TextWrapped("vk_slang_editor recognizes some variable names and will automatically write values to them.\n");
  ImGui::PushFont(nvgui::getMonospaceFont());
  ImGui::TextWrapped(
      "* uniform float    iTime: Time in seconds, starting from 0.0\n"
      "* uniform int      iFrame: Frame count, starting from 0\n"
      "* uniform uint2    iResolution: Viewport resolution in pixels\n"
      "* uniform int2     iMouse: .xy = mouse position in pixels, .zw = left/right mouse buttons\n"
      "* uniform float4x4 iView, iViewInverse, iProj, iProjInverse, iProjView, iProjViewInverse: Camera matrices\n"
      "* uniform float3   iEye: Camera position\n"
      "* uniform float3x3 iFragCoordToDirection: mul(float3(pixel, 1), iFragCoordToDirection) returns camera ray "
      "direction\n"
      "* RWTexture2D      texFrame: Texture that will be displayed\n");
  ImGui::PopFont();
  ImGui::TextWrapped(
      "If you use a different type for the above, vk_slang_editor will convert and write to the type you wrote.\n"
      "Textures: Write `texFoo` to load a file named `foo` (extension does not matter). Unrecognized textures will "
      "be allocated screen-sized.\n"
      "Shader parameters: Unrecognized shader parameters and buffers will appear in the Shader Parameters menu.\n"
      "Write multiple Slang entrypoints on the left to use multi-pass. Entrypoints will be grouped together and "
      "run in sequence.\n"
      "Compute shader dispatches are screen-sized.\n"
      "Rasterization shaders will automatically load a shaderball.\n");


  header("Operators:");
  ImGui::TextWrapped(
      "Arithmetic: ( ) + - * / %%\n"
      "Bitwise arithmetic: ~ & ^ | << >>\n"
      "Casts: float(x), (float)x, bit_cast<float>(x), reinterpret<float>(x)\n"
      "Comments: // /* */\n"
      "Comparison: < > <= >= == != && ||\n");

  header("Types:");
  ImGui::TextWrapped(
      "Scalar types: void bool int uint int(8|16|32|64)_t uint(8|16|32|64)_t half float double\n"
      "Vectors: float1 float2 float3 float4 int1 int2 int3 int4 bool1 bool2 bool3 bool4... or vector<T,N>\n"
      "Vector components: .xyzw .rgba .stpq [0] [1] [2] [3]\n"
      "Matrices: Row-major. float3x4 == matrix<T,3,4> has 3 rows, 4 columns; use [row][col] and mul(v, M)\n"
      "Arrays: int a[16] = {1, 2, 3, ...} or Array<int,16> a = {1, 2, 3, ...}.\n"
      "Textures: Texture2D, TextureCubeArray, RWTexture2D, etc. Sampler2D == Texture2D + SamplerState.\n"
      "Buffers: ConstantBuffer<T> StructuredBuffer<T> ByteAddressBuffer<T> ParameterBuffer<T>...\n"
      "Ray tracing: RaytracingAccelerationStructure RayDesc RayQuery<Flags:uint>\n"
      "Globals: `float a`, `const float a` are shader parameters; `static float a` is a global; `static const "
      "float a` is a compile-time constant.\n"
      "Functions: float add(int x, float y) { return x + y; }. Qualifiers: [none] in out inout\n"
      "Structs: Support member functions, static functions, __init() constructors. Use [mutating] for non-const "
      "functions.\n"
      "Properties: struct MyType{ property uint highBits { get{ ... } set(uint x){ ... } }}\n"
      "Operator overloads: MyType operator+(MyType a, MyType b){ ... }; operator()(...) and __subscript(...) "
      "member methods supported.\n"
      "Enums: Scoped by default. [UnscopedEnum] makes them unscoped. [Flags] uses powers of 2.\n"
      "Tuples: Tuple<int, float, bool> t0 = makeTuple(3, 1.f, true). Access using _0, _1, ... Swizzle with "
      "_0_0_1.\n"
      "Pointers to global memory: MyType*, Ptr<MyType>, DescriptorHandle<Texture2D>, loadAligned(), "
      "storeAligned()\n");

  header("Language:");
  ImGui::TextWrapped(
      "Flow control: if else for do while return break continue switch/case\n"
      "Multi-level break: outer: for(...){ for(...){ if(...){ break outer; }}}\n"
      "Including modules: module YourLibrary; import YourLibrary; __include \"path/to/file.slang\"\n"
      "Preprocessor: #include #define #undef #if #ifdef #ifndef #else #elif #endif #error #warning #line #pragma\n"
      "More features: namespace foo{ ... }; var a : int = 1; if(let x = ...)\n");

  header("Generics and Interfaces:");
  ImGui::TextWrapped(
      "Define an interface: interface IFoo{ int bar(float arg); int baz(This other); }\n"
      "Implement an interface: `struct MyType : IFoo { ... }` or after-definition using `extension MyType : IFoo { "
      "... }`\n"
      "Test/cast interfaces: is as\n"
      "Generic functions: T foo<T>(T arg) where T : IFoo { ... }\n"
      "Generic structs: struct MyType<T, U> where T: IFoo, IBar where U: IBaz<T> { ...}\n"
      "Extend a generic: `extension<T:IFoo> MyType<T> : IBar { ... }` extends `MyType<T>` to conform to `IBar`.\n"
      "Variadic generics: void printAll<each T>(expand each T args){ expand printf(\"%%d\\n\", each args); }\n"
      "Built-in interfaces: IArithmetic IArray IRWArray IAtomicable IComparable IDifferentiable "
      "IDifferentiableMutatingFunc IFloat IFunc IInteger ...\n");

  header("Attributes:");
  ImGui::TextWrapped(
      "Shaders: [shader(\"compute\")] [shader(\"vertex\")] [shader(\"fragment\")] [shader(\"raygeneration\")]...\n"
      "Compute workgroup size: [numthreads]\n"
      "Autodifferentiation: [Differentiable] [ForwardDifferentiable] [BackwardDifferentiable] "
      "[ForwardDerivativeOf] [BackwardDerivativeOf] [PrimalSubstituteOf] [PreferCheckpoint] [PreferRecompute]\n"
      "Compiler: [ForceInline] [ForceUnroll] [MaxIters(n)]\n"
      "Other: [earlydepthstencil] [raypayload] [shader_record]\n"
      "Vulkan-specific: [format(\"rgba16f\")] [MaximallyReconverges] [push_constant] [vk_binding] [vk_constant_id] "
      "[vk_location] [vk_offset] [vk_spirv_instruction]...\n");

  header("Intrinsics:");
  ImGui::TextWrapped(
      "Math: abs acos acosh asin asinh atan atan2 atanh ceil clamp copysign cos cosh cospi degrees exp exp10 exp2 "
      "fabs fdim floor fma fmax fmax3 fmedian3 fmin fmin3 fmod frac frexp isfinite isinf isnan ldexp lerp log "
      "log10 log2 mad max max3 median3 min min3 modf msad4 mul nextafter pow powr radians rcp rint round rsqrt "
      "saturate sign sin sincos sinh sinpi smoothstep sqrt step tan tanh tanpi trunc\n"
      "Vector math: cross determinant distance dot dot2add dot4add_i8packed dot4add_u8packed faceforward length "
      "normalize reflect refract select transpose\n"
      "Vector packing: [un]pack{Half2x16, Int4x8, Int4x8Clamp, Snorm2x16, Snorm4x8, Uint4x8, Uint4x8Clamp, "
      "Unorm2x16, Unorm4x8, clamp_s8, clamp_u8, _s8, _u8}\n"
      "Bits: bit_cast bitfieldExtract bitfieldInsert copysign countbits firstbithigh firstbitlow reversebits "
      "Atomic: Interlocked{Add, And, CompareExchange, CompareExchangeFloatBitwise, CompareStore, "
      "CompareStoreFloatBitwise, Exchange, Max, Min, Or, Xor}\n"
      "Bindless: NonUniformResourceIndex(...)\n"
      "Cooperative vectors: coopVec{Load, LoadGroupshared, MatMul, MatMulAdd, MatMulAddPacked, MatMulPacked, "
      "OuterProductAccumulate, ReduceSumAccumulate}\n"
      "Discard fragment: clip. Stop draw call: abort\n"
      "Vector->bool: all any\n"
      "Timing: clock2x32ARB clockARB getRealtimeClock getRealtimeClockLow\n"
      "Tuples: concat\n"
      "Ray tracing: ObjectRayDirection ObjectRayOrigin ObjectToWorld PrimitiveIndex WorldRayDirection "
      "WorldRayOrigin WorldToObject\n"
      "Autodiff: detach diffPair isDifferentialNull updateDiff updatePair updatePrimal\n"
      "Finite differences: ddx ddx_coarse ddx_fine ddy ddy_coarse ddy_fine fwidth fwidth_coarse fwidth_fine\n"
      "Quad invocations: QuadAll QuadAny QuadReadAcrossDiagonal QuadReadAcrossX QuadReadAcrossY QuadReadLaneAt\n"
      "Wave operations: Wave{Active{AllEqual, AllTrue, AnyTrue, Ballot, BitAnd, BitOr, BitXor, CountBits, Max, "
      "Min, Product, Sum}, BroadcastLaneAt, ClusteredRotate, GetLaneCount, GetLaneIndex, IsFirstLane, Match, "
      "PrefixCountBits, PrefixSum, ReadLaneAt, ReadLaneFirst, Rotate, Shuffle}\n"
      "SPV_NV_shader_subgroup_partitioned: Same as wave operations but using WaveMulti*.\n");
  // debugBreak?
}

void Sample::guiPaneParameters()
{
  namespace PE = nvgui::PropertyEditor;

  ImGui::TextWrapped(
      "Click and drag to change shader parameters. Alt-click and drag to make fine edits. "
      "Control-click to type in a value.");

  PE::begin();
  PE::Combo("Clear texFrame when:", reinterpret_cast<int*>(&m_shaderParams.clearColorWhen), "Never\0Every frame\0\0");
  PE::ColorEdit4("Clear color", m_shaderParams.clearColor.data(), ImGuiColorEditFlags_Float,
                 "texFrame (if it exists) will be cleared to this color at the start of every frame.");
  PE::Combo("Clear depth/stencil when:", reinterpret_cast<int*>(&m_shaderParams.clearDepthStencilWhen),
            "Never\0Every frame\0\0", -1,
            "Only applies if there is a depth/stencil texture (e.g. Sampler2D<float4> texDepth) in the shader.");
  PE::InputFloat("Clear depth value", &m_shaderParams.clearDepth);
  PE::InputInt("Clear stencil value", reinterpret_cast<int*>(&m_shaderParams.clearStencil), 0, 0);
  PE::end();

  if(m_resources)
  {
    PE::begin();
    // Draw sliders for all unknown uniforms
    for(const UniformWrite& uniform : m_resources->uniformUpdates)
    {
      if(uniform.source != Source::eUnknown)
      {
        continue;
      }

      if(uniform.cols > 4)
      {
        PE::entry(uniform.name, [] {
          ImGui::Text("Too many columns to display");
          return false;
        });
        continue;
      }

      // Get raw data for this uniform; this access also creates an entry
      // if it doesn't exist.
      void*    data  = m_shaderParams.uniforms[uniform.name].data();
      uint8_t* bytes = reinterpret_cast<uint8_t*>(data);

      // Convert from a scalar type to an ImGui type so that we can use
      // ImGui::DragScalarN
      ImGuiDataType dataType = ImGuiDataType_COUNT;
      // Note: This could be combined with slangScalarTypeBitSize in sample_render.cpp
      size_t byteStride = 4;
      switch(uniform.scalarType)
      {
        case SLANG_SCALAR_TYPE_INT32:
          dataType = ImGuiDataType_S32;
          break;
        case SLANG_SCALAR_TYPE_UINT32:
          dataType = ImGuiDataType_U32;
          break;
        case SLANG_SCALAR_TYPE_INT64:
          dataType = ImGuiDataType_S32;
          break;
        case SLANG_SCALAR_TYPE_UINT64:
          dataType   = ImGuiDataType_U64;
          byteStride = 8;
          break;
        case SLANG_SCALAR_TYPE_FLOAT32:
          dataType = ImGuiDataType_Float;
          break;
        case SLANG_SCALAR_TYPE_FLOAT64:
          dataType   = ImGuiDataType_Double;
          byteStride = 8;
          break;
        case SLANG_SCALAR_TYPE_INT8:
          dataType   = ImGuiDataType_S8;
          byteStride = 1;
          break;
        case SLANG_SCALAR_TYPE_UINT8:
          dataType   = ImGuiDataType_U8;
          byteStride = 1;
          break;
        case SLANG_SCALAR_TYPE_INT16:
          dataType   = ImGuiDataType_S16;
          byteStride = 2;
          break;
        case SLANG_SCALAR_TYPE_UINT16:
          dataType   = ImGuiDataType_U16;
          byteStride = 2;
          break;
        default:
          break;  // COUNT can happen for some types
      }

      if(SLANG_SCALAR_TYPE_BOOL == uniform.scalarType)
      {
        PE::entry(uniform.name, [&] {
          bool changed = false;
          for(uint32_t row = 0; row < uniform.rows; row++)
          {
            for(uint32_t col = 0; col < uniform.cols; col++)
            {
              if(col != 0)
              {
                ImGui::SameLine();
              }
              const uint32_t i = row * uniform.cols + col;
              ImGui::PushID(i);
              changed = ImGui::Checkbox("##hidden", reinterpret_cast<bool*>(bytes + 4 * i)) || changed;
              ImGui::PopID();
            }
          }
          return changed;
        });
      }
      else if(SLANG_SCALAR_TYPE_FLOAT16 == uniform.scalarType)
      {
        // ImGui doesn't implement float16, so we need to convert to and from `float`
        PE::entry(uniform.name, [&] {
          bool      changed = false;
          uint16_t* halves  = reinterpret_cast<uint16_t*>(bytes);
          for(uint32_t row = 0; row < uniform.rows; row++)
          {
            std::array<float, 4> floats{};
            for(uint32_t col = 0; col < uniform.cols; col++)
            {
              const uint32_t i = row * uniform.cols + col;
              floats[col]      = glm::detail::toFloat32(halves[i]);
            }
            ImGui::PushID(row);
            ImGui::SetNextItemWidth(-FLT_MIN);
            changed = ImGui::DragScalarN("##hidden", ImGuiDataType_Float, floats.data(), static_cast<int>(uniform.cols), 0.025f);
            ImGui::PopID();
            for(uint32_t col = 0; col < uniform.cols; col++)
            {
              const uint32_t i = row * uniform.cols + col;
              halves[i]        = glm::detail::toFloat16(floats[col]);
            }
          }
          return changed;
        });
      }
      else if(ImGuiDataType_COUNT != dataType)
      {
        PE::entry(uniform.name, [&] {
          bool changed = false;
          for(uint32_t row = 0; row < uniform.rows; row++)
          {
            const float vSpeed = (dataType == ImGuiDataType_Float || dataType == ImGuiDataType_Double) ? 0.025f : 0.25f;
            ImGui::PushID(row);
            ImGui::SetNextItemWidth(-FLT_MIN);
            changed = ImGui::DragScalarN("##hidden", dataType, bytes + byteStride * uniform.cols * row,
                                         static_cast<int>(uniform.cols), vSpeed)
                      || changed;
            ImGui::PopID();
          }
          return changed;
        });
      }
    }
    PE::end();

    // See if we have any storage buffers for which to display parameters
    bool displayStorageBuffers = false;
    for(const StorageBuffer& storageBuffer : m_resources->storageBuffers)
    {
      if(storageBuffer.source == Source::eUnknown)
      {
        displayStorageBuffers = true;
        break;
      }
    }

    if(displayStorageBuffers && ImGui::CollapsingHeader("Storage Buffers", ImGuiTreeNodeFlags_DefaultOpen))
    {
      PE::begin();
      for(const StorageBuffer& storageBuffer : m_resources->storageBuffers)
      {
        if(storageBuffer.source != Source::eUnknown)
        {
          continue;
        }

        // Note that this creates an entry if it doesn't exist.
        StorageBufferParameters& params = m_shaderParams.storageBuffers[storageBuffer.name];

        PE::entry(storageBuffer.name + ": Info", [&] {
          const VkExtent2D resolution   = m_app->getViewportSize();
          const size_t     requiredSize = params.computeBufferSize(resolution, storageBuffer.elementStride);
          ImGui::Text("Element stride: %zu bytes", storageBuffer.elementStride);
          ImGui::Text("Required: %zu bytes", requiredSize);
          ImGui::Text("Allocated: %zu bytes", storageBuffer.buffer.bufferSize);
          if(requiredSize > m_physicalDeviceProperties.limits.maxStorageBufferRange)
          {
            ImGui::PushFont(nvgui::getIconicFont());
            ImGui::Text("%s", nvgui::icon_warning);
            ImGui::PopFont();
            ImGui::SameLine();
            ImGui::TextWrapped(
                "This is too large for a storage buffer -- it is larger than "
                "VkPhysicalDeviceLimits::maxStorageBufferRange (%u)!",
                m_physicalDeviceProperties.limits.maxStorageBufferRange);
          }
          return false;
        });

        PE::entry(storageBuffer.name + ": Element Count", [&] {
          bool changed = false;
          changed      = ImGui::InputInt("##elements", &params.base, 0, 0) || changed;

          ImGui::AlignTextToFramePadding();
          ImGui::Text("+");
          ImGui::SameLine();
          ImGui::SetNextItemWidth(-FLT_MIN);
          changed = ImGui::InputInt("##perTile", &params.perTile, 0, 0) || changed;

          ImGui::AlignTextToFramePadding();
          ImGui::Text("per");
          ImGui::SameLine();

          // The math here is a bit complex since we need to set both InputInts
          // to the same size while leaving space for the "x" and "px" text.
          const float textWidth = ImGui::CalcTextSize("x").x + ImGui::CalcTextSize("px").x;
          const float itemWidth = (ImGui::GetContentRegionAvail().x - textWidth - 3.f * ImGui::GetStyle().ItemSpacing.x) * .5f;

          ImGui::SetNextItemWidth(itemWidth);
          changed = ImGui::InputInt("##tileSizeX", &params.tileSizeX, 0, 0) || changed;
          ImGui::SameLine();
          ImGui::Text("x");
          ImGui::SameLine();
          ImGui::SetNextItemWidth(itemWidth);
          changed = ImGui::InputInt("##tileSizeY", &params.tileSizeY, 0, 0) || changed;
          ImGui::SameLine();
          ImGui::Text("px");

          return changed;
        });
      }
      PE::end();
    }

    // TODO: Texture selection GUI
  }
}

void Sample::guiPaneTargetDisassembly()
{
  TextEditor& editor = m_editors[+Editor::eTargetDisassembly];
  if(editor.IsEmpty())
  {
    const uint32_t* spirv          = m_compiler.getSpirv();
    const size_t    spirvSizeBytes = m_compiler.getSpirvSize();
    assert(spirvSizeBytes % 4 == 0);  // Make sure this is bytes and not words
    if(spirvSizeBytes == 0)
    {
      editor.SetLanguage(nullptr);
      editor.SetText("No SPIR-V to disassemble.");
    }
    else
    {
      const uint32_t options = SPV_BINARY_TO_TEXT_OPTION_INDENT | SPV_BINARY_TO_TEXT_OPTION_COMMENT | SPV_BINARY_TO_TEXT_OPTION_NESTED_INDENT
                               | SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES | SPV_BINARY_TO_TEXT_OPTION_REORDER_BLOCKS;
      // We call the C API here, because of an ABI issue: the SPIRV-Tools
      // C++ API cannot correctly allocate std::string objects on MSVC in
      // Debug mode.
      // Specifically, in Debug mode when _ITERATOR_DEBUG_LEVEL is nonzero,
      // std::string contains 8 additional bytes at the start for iterator
      // debug info. Incidentally, this means that std::string isn't a
      // standard layout type -- but more directly means that if we wanted
      // to use the SPIRV-Tools C++ API in Debug mode, we'd need to link
      // with the debug SPIRV-Tools library, which is not guaranteed to be
      // installed.
      // For more info on this, see https://stackoverflow.com/a/67042132 .
      spv_text spvText = nullptr;
      const spv_result_t result = spvBinaryToText(m_spirvTools.CContext(), spirv, spirvSizeBytes / 4, options, &spvText, nullptr);
      if(SPV_SUCCESS != result)
      {
        editor.SetLanguage(nullptr);
        editor.SetText("Disassembly failed with SPIRV-Tools error code " + std::to_string(result) + ".");
      }
      else
      {
        editor.SetLanguage(getSpirVLanguageInfo());
        editor.SetText(std::string_view(spvText->str, spvText->length));
      }
      spvTextDestroy(spvText);
    }
  }
  else
  {
    ImGui::PushFont(nvgui::getMonospaceFont());
    editor.Render(kUIPaneTargetDisassembly);
    ImGui::PopFont();
  }
}

void Sample::guiPaneTransport()
{
  {
    ImGui::PushFont(nvgui::getIconicFont());

    if(ImGui::Button(nvgui::icon_media_step_backward))
    {
      m_shaderParams.time = 0.0;
      m_frame             = 0;
    }

    ImGui::SameLine();
    const char* pausePlayIcon = m_shaderParams.paused ? nvgui::icon_media_play : nvgui::icon_media_pause;
    if(ImGui::Button(pausePlayIcon))
    {
      m_shaderParams.paused = !m_shaderParams.paused;
    }

    ImGui::PopFont();
  }

  {
    ImGui::PushFont(nvgui::getMonospaceFont());

    ImGui::SameLine();
    const float remainingWidth = ImGui::GetContentRegionAvail().x;
    ImGui::SetNextItemWidth(remainingWidth * .5f);
    const float inputTime  = static_cast<float>(m_shaderParams.time);
    float       outputTime = inputTime;
    const bool  timeEdited = ImGui::DragFloat("##time", &outputTime, 0.1f, 0.0f, 0.0f, "Time: %.3f");
    m_shaderParams.time += static_cast<double>(outputTime - inputTime);
    // Don't have time move while someone's editing it:
    m_temporarilyPaused = timeEdited || (ImGui::IsMouseDown(ImGuiMouseButton_Left) && ImGui::IsItemHovered());

    ImGui::SameLine();
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::SliderFloat("##speed", &m_shaderParams.timeSpeed, -10.0f, 10.0f, "Speed: %.3f", ImGuiSliderFlags_Logarithmic);

    ImGui::PopFont();
  }
}

void Sample::guiPaneReflection()
{
  slang::IModule* slangModule = m_compiler.getSlangModule();
  if(!slangModule)
  {
    ImGui::TextWrapped("Slang file did not compile successfully.");
    return;
  }

  if(ImGui::CollapsingHeader("Data", ImGuiTreeNodeFlags_DefaultOpen))
  {
    ImGui::AlignTextToFramePadding();
    ImGui::Text("Display:");
    ImGui::SameLine();
    if(ImGui::RadioButton("JSON", m_editorSettings.reflectionShowJson))
    {
      m_editorSettings.reflectionShowJson = true;
    }
    ImGui::SameLine();
    if(ImGui::RadioButton("Detailed", !m_editorSettings.reflectionShowJson))
    {
      m_editorSettings.reflectionShowJson = false;
    }

    if(m_editorSettings.reflectionShowJson)
    {
      if(!m_reflectionJson)
      {
        ImGui::Text("No reflection JSON available.");
      }
      else
      {
        const char* text = reinterpret_cast<const char*>(m_reflectionJson->getBufferPointer());
        if(ImGui::Button("Copy"))
        {
          ImGui::SetClipboardText(text);
        }
        ImGui::PushFont(nvgui::getMonospaceFont());
        ImGui::Text("%s", text);
        ImGui::PopFont();
      }
    }
    else
    {
      guiProgramReflection(m_compiler.getSlangProgram());
      guiModuleReflection(slangModule);
    }
  }

  if(ImGui::CollapsingHeader("Computed", ImGuiTreeNodeFlags_DefaultOpen))
  {
    std::stringstream text;
    text << "Descriptor sets:\n";
    for(const DescriptorWrite& write : m_resources->descriptorSetUpdates)
    {
      text << "\t";
      switch(write.descriptorType)
      {
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
          text << "Texture `" << m_resources->textures[write.resourceIndex].name << "`";
          break;
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
          text << "Uniform buffer `" << m_resources->uniformBuffers[write.resourceIndex].name << "`";
          break;
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
          text << "Storage buffer `" << m_resources->storageBuffers[write.resourceIndex].name << "`";
          break;
      }
      text << " (" << write.resourceIndex << ") -> set " << write.index.set << ", binding " << write.index.binding << "\n";
    }
    text << "Uniforms:\n";
    for(const UniformWrite& write : m_resources->uniformUpdates)
    {
      text << "\t" << write.name << " -> buffer " << write.bufferIndex << " ("
           << m_resources->uniformBuffers[write.bufferIndex].name << "), byte offset " << write.byteOffset << "\n";
    }
    const std::string textStr = text.str();

    if(ImGui::Button("Copy"))
    {
      ImGui::SetClipboardText(textStr.c_str());
    }
    ImGui::PushFont(nvgui::getMonospaceFont());
    ImGui::Text("%s", textStr.c_str());
    ImGui::PopFont();
  }
}

void Sample::onUIMenu()
{
  bool                  newFile  = false;
  bool                  openFile = false;
  std::filesystem::path openFilename;
  bool                  saveShaderAndConfig = false;
  bool                  saveShader          = false;
  bool                  saveViewport        = false;
  bool                  saveScreen          = false;

  bool uiLayoutReset = false;
  bool uiScaleUp     = false;
  bool uiScaleDown   = false;
  bool uiScaleReset  = false;
  bool vsync         = m_app->isVsync();

  if(ImGui::BeginMenu("File"))
  {
    newFile |= ImGui::MenuItem("New");
    openFile |= ImGui::MenuItem("Open...", "Ctrl+O");
    saveShaderAndConfig |= ImGui::MenuItem("Save Shader...", "Ctrl+S");
    saveViewport |= ImGui::MenuItem("Save Viewport...");
    saveScreen |= ImGui::MenuItem("Save Screen...");
    ImGui::EndMenu();
  }

  if(const auto maybeFile = m_exampleShaderCache.doUI())
  {
    openFilename = maybeFile.value();
  }

  if(ImGui::BeginMenu("Editor"))
  {
    uiLayoutReset |= ImGui::MenuItem("UI: Layout Reset");
    uiScaleUp |= ImGui::MenuItem("UI: Scale Up", "Ctrl+Plus", nullptr);
    uiScaleDown |= ImGui::MenuItem("UI: Scale Down", "Ctrl+Minus", nullptr);
    uiScaleReset |= ImGui::MenuItem("UI: Scale Reset", "Ctrl+0", nullptr);
    ImGui::MenuItem("V-Sync", "Ctrl+Shift+V", &vsync);
    ImGui::MenuItem(kUIPaneEditorSettings, "", &m_editorSettings.showPanes.editorSettings);
    ImGui::EndMenu();
  }

  if(ImGui::MenuItem("Shader Parameters"))
  {
    m_editorSettings.showPanes.parameters = true;
  }

  if(ImGui::BeginMenu("View"))
  {
    m_resources->updatePipelineStats(m_ctx.getDevice());
    if(m_resources->hadAnyPipelineRepresentations)
    {
      ImGui::MenuItem(kUIPaneBinaryDisassembly, "", &m_editorSettings.showPanes.binaryDisassembly);
    }
    ImGui::MenuItem(kUIPaneCamera, "", &m_editorSettings.showPanes.camera, m_resources->hasCameraUniform);
    if(!m_resources->hasCameraUniform)
    {
      nvgui::tooltip("Add a camera uniform like `uniform float3 eye;` to your shader to access camera controls.", false, 0.f);
    }
    ImGui::MenuItem(kUIPaneDiagnostics, "", &m_editorSettings.showPanes.diagnostics);
    ImGui::MenuItem(kUIPaneEditorSettings, "", &m_editorSettings.showPanes.editorSettings);
    ImGui::MenuItem(kUIPanePipelineStats, "", &m_editorSettings.showPanes.pipelineStats);
    ImGui::MenuItem(kUIPaneQuickReference, "", &m_editorSettings.showPanes.quickReference);
    ImGui::MenuItem(kUIPaneReflection, "", &m_editorSettings.showPanes.reflection);
    ImGui::MenuItem(kUIPaneParameters, "", &m_editorSettings.showPanes.parameters);
    ImGui::MenuItem(kUIPaneTargetDisassembly, "", &m_editorSettings.showPanes.targetDisassembly);
    ImGui::MenuItem(kUIPaneTransport, "", &m_editorSettings.showPanes.transport);
    ImGui::EndMenu();
  }

  if(ImGui::BeginMenu("Help"))
  {
    if(ImGui::MenuItem("Slang Docs Online"))
    {
      ImGui::GetPlatformIO().Platform_OpenInShellFn(nullptr, "https://docs.shader-slang.org");
    }
    ImGui::MenuItem(kUIPaneQuickReference, "", &m_editorSettings.showPanes.quickReference);
    ImGui::MenuItem(kUIPaneAbout, "", &m_editorSettings.showPanes.about);
    ImGui::EndMenu();
  }

  openFile |= ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_O);
  saveShaderAndConfig |= ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_S);
  saveShader |= ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_S);

  uiScaleUp |= ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_Equal);
  uiScaleDown |= ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_Minus);
  uiScaleReset |= ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_0);
  vsync ^= ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_V);

  if(newFile)
  {
    m_modal.action             = ModalState::Action::eLoadOrNew;
    m_modal.confirmationNeeded = hasCodeChanged();
    m_modal.fileToLoad.clear();
    if(m_modal.confirmationNeeded)
    {
      ImGui::OpenPopup(kUIModalShaderLoad);
    }
  }

  if(openFile)
  {
    openFilename = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Open Shader or Config...",
                                               "Slang Shader (.slang), Config (.json)|*.slang;*.json");
  }

  // Since this can be reached by both Open and Examples:
  if(!openFilename.empty())
  {
    m_modal.action             = ModalState::Action::eLoadOrNew;
    m_modal.confirmationNeeded = hasCodeChanged();
    m_modal.fileToLoad         = openFilename;
    if(m_modal.confirmationNeeded)
    {
      ImGui::OpenPopup(kUIModalShaderLoad);
    }
  }

  if(saveShaderAndConfig)
  {
    const std::filesystem::path filename =
        nvgui::windowSaveFileDialog(m_app->getWindowHandle(), "Save Shader and Config...",
                                    "Slang Shader (.slang)|*.slang");
    if(!filename.empty())
    {
      this->saveShaderAndConfig(filename, true);
    }
  }

  const Texture* displayTex = getDisplayTexture();
  if(saveViewport && displayTex)
  {
    const std::filesystem::path filename = nvgui::windowSaveFileDialog(m_app->getWindowHandle(), "Save Viewport...",
                                                                       "PNG(.png),JPG(.jpg)|*.png;*.jpg;*.jpeg");
    if(!filename.empty())
    {
      m_app->saveImageToFile(displayTex->image.image, {displayTex->size.width, displayTex->size.height}, filename);
    }
  }

  if(saveScreen)
  {
    const std::filesystem::path filename =
        nvgui::windowSaveFileDialog(m_app->getWindowHandle(), "Save Screen Including UI...",
                                    "PNG(.png),JPG(.jpg)|*.png;*.jpg;*.jpeg");
    if(!filename.empty())
    {
      m_app->screenShot(filename);
    }
  }

  if(uiLayoutReset)
  {
    // HACK: Makes use of ImGui internals
    ImGuiWindow* windowOverViewport = ImGui::FindWindowByName("WindowOverViewport_11111111");
    assert(windowOverViewport);
    if(windowOverViewport)  // In case ImGui internals change
    {
      const ImGuiID dockID = windowOverViewport->GetID("DockSpace");
      dockSetup(dockID);
    }
    m_editorSettings.showPanes = EditorSettings::ShowPanes{};
  }

  const float uiScaleTicks = (uiScaleUp ? 1.f : 0.f)       //
                             + (uiScaleDown ? -1.f : 0.f)  //
                             + (ImGui::IsKeyDown(ImGuiKey_ReservedForModCtrl) ? ImGui::GetIO().MouseWheel : 0.f);
  if(uiScaleTicks != 0.f)
  {
    const float uiScaleFactor = std::pow(1.1f, uiScaleTicks);
    scaleUI(uiScaleFactor);
  }
  if(uiScaleReset)
  {
    scaleUI(1.f / m_editorSettings.uiScale);
  }

  if(m_app->isVsync() != vsync)
  {
    m_app->setVsync(vsync);
  }

  // Popup windows
  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
  ImGui::SetNextWindowSize(ImVec2(400, 0.0f));
  if(ImGui::BeginPopupModal(kUIModalShaderLoad, nullptr, ImGuiWindowFlags_AlwaysAutoResize))
  {
    ImGui::TextWrapped(
        "You've made unsaved changes to the shader. Are you sure you want to load a new one and lose the changes "
        "you've made?");
    ImGui::Separator();
    if(ImGui::Button("Yes"))
    {
      m_modal.confirmationNeeded = false;
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if(ImGui::Button("No"))
    {
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
  if(!m_modal.confirmationNeeded && m_modal.action == ModalState::Action::eLoadOrNew)
  {
    if(m_modal.fileToLoad.empty())
    {
      openSlangCode(kDefaultComputeShader, "");
    }
    else
    {
      openSlangCodeAndConfig(m_modal.fileToLoad);
    }
  }

  if(m_modal.pleaseOpenExitModal)
  {
    ImGui::OpenPopup(kUIModalExit);
    m_modal.pleaseOpenExitModal = false;
  }
  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
  ImGui::SetNextWindowSize(ImVec2(400, 0.0f));
  if(ImGui::BeginPopupModal(kUIModalExit, nullptr, ImGuiWindowFlags_AlwaysAutoResize))
  {
    ImGui::TextWrapped("Are you sure you want to exit? You've made unsaved changes to the shader.");
    ImGui::Separator();
    if(ImGui::Button("Yes"))
    {
      m_modal.confirmationNeeded = false;
      glfwSetWindowShouldClose(m_app->getWindowHandle(), GLFW_TRUE);
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if(ImGui::Button("No"))
    {
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
}

void Sample::onFileDrop(const std::filesystem::path& filename)
{
  // TODO: Make this more elegant by delayed-destroying the old scene
  if(!(nvutils::extensionMatches(filename, ".glb") || nvutils::extensionMatches(filename, ".gltf")))
  {
    LOGI("Not loading %s: not a .glb or .gltf file\n", nvutils::utf8FromPath(filename).c_str());
    return;
  }
  vkDeviceWaitIdle(m_ctx.getDevice());

  m_currentScene.destroy();
  if(!m_currentScene.load(filename))
  {
    return;
  }

  VkCommandBuffer cmd = m_app->createTempCmdBuffer();
  m_staging.uploader.setEnableLayoutBarriers(true);
  m_currentSceneVk.create(cmd, m_staging.uploader, m_currentScene);
  m_staging.uploader.cmdUploadAppended(cmd);
  m_staging.uploader.setEnableLayoutBarriers(false);
  m_app->submitAndWaitTempCmdBuffer(cmd);
}

void Sample::updateTextEditorSettings()
{
  // All Set functions here are inexpensive, so we can do this every frame:
  const EditorSettings::Text& t = m_editorSettings.text;
  for(TextEditor& editor : m_editors)
  {
    editor.SetLineSpacing(t.lineSpacing);
    editor.SetAutoIndentEnabled(t.autoIndent);
    editor.SetShowWhitespacesEnabled(t.showWhitespaces);
    editor.SetShowLineNumbersEnabled(t.showLineNumbers);
    editor.SetShowScrollbarMiniMapEnabled(t.showScrollbarMiniMap);
    editor.SetShowMatchingBrackets(t.showMatchingBrackets);
    editor.SetCompletePairedGlyphs(t.completePairedGlyphs);
    if(t.middleMousePan)
    {
      editor.SetMiddleMousePanMode();
    }
    else
    {
      editor.SetMiddleMouseScrollMode();
    }
  }
}

void Sample::updateTextEditorPalettes()
{
  for(TextEditor& editor : m_editors)
  {
    editor.SetPalette(m_editorSettings.textPalette);
  }
}

CpuImage Sample::getThumbnailImage()
{
  CpuImage result;

  Texture* displayTex = getDisplayTexture();
  if(!displayTex)
  {
    return result;
  }

  // Load the image onto the CPU; then downscale using stb_image_resize2.
  // TODO: Do this on the GPU instead.
  const VkExtent3D inputSize         = displayTex->size;
  VkCommandBuffer  cmd               = m_app->createTempCmdBuffer();
  VkImage          linearImage       = {};
  VkDeviceMemory   linearImageMemory = {};
  const VkResult vkRes = nvvk::imageToLinear(cmd, m_ctx.getDevice(), m_ctx.getPhysicalDevice(), displayTex->image.image,
                                                  {inputSize.width, inputSize.height}, linearImage, linearImageMemory, VK_FORMAT_R8G8B8A8_UNORM);
  // Account for imageToRgba8Linear's barriers, just in case
  displayTex->image.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  displayTex->currentStages                = nvvk::INFER_BARRIER_PARAMS;
  m_app->submitAndWaitTempCmdBuffer(cmd);
  if(vkRes < 0)
  {
    LOGW("Creating downscaled image failed with VkResult %d\n", vkRes);
    return result;
  }

  // Get layout of the image (including offset and row pitch)
  VkImageSubresource  subResource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
  VkSubresourceLayout subResourceLayout;
  vkGetImageSubresourceLayout(m_ctx.getDevice(), linearImage, &subResource, &subResourceLayout);

  // Map image memory so we can start copying from it
  const char* data = nullptr;
  vkMapMemory(m_ctx.getDevice(), linearImageMemory, 0, VK_WHOLE_SIZE, 0, (void**)&data);
  data += subResourceLayout.offset;

  // Copy the data and adjust for the row pitch.
  // Note that we over-allocate; stbi_write_jpg discards alpha.
  std::vector<uint8_t> linearImageCpu(static_cast<size_t>(inputSize.width) * inputSize.height * 4);
  for(uint32_t y = 0; y < inputSize.height; y++)
  {
    memcpy(linearImageCpu.data() + y * inputSize.width * 4, data, static_cast<size_t>(inputSize.width) * 4);
    data += subResourceLayout.rowPitch;
  }

  vkUnmapMemory(m_ctx.getDevice(), linearImageMemory);
  vkFreeMemory(m_ctx.getDevice(), linearImageMemory, nullptr);
  vkDestroyImage(m_ctx.getDevice(), linearImage, nullptr);

  const float scale = std::min({1.f, 480.f / static_cast<float>(inputSize.width), 270.f / static_cast<float>(inputSize.height)});
  result.size   = VkExtent3D{.width  = static_cast<uint32_t>(inputSize.width * scale),
                             .height = static_cast<uint32_t>(inputSize.height * scale),
                             .depth  = 1};
  result.format = VK_FORMAT_R8G8B8A8_UNORM;
  result.allocate(1, 1, 1);
  result.subresource(0, 0, 0).resize(static_cast<size_t>(result.size.width) * result.size.height * 4);
  if(!stbir_resize_uint8_srgb(linearImageCpu.data(), inputSize.width, inputSize.height, inputSize.width * 4,  //
                              reinterpret_cast<uint8_t*>(result.subresource(0, 0, 0).data()), result.size.width,
                              result.size.height,
                              result.size.width * 4,  //
                              STBIR_RGBA))
  {
    LOGW("stbir_resize_uint8_srgb failed!\n");
    return result;
  }

  return result;
}

bool Sample::openSlangCode(const std::string& text, const std::filesystem::path& diskPath)
{
  // Update paths
  m_languageServer.notifyDocumentClosed();
  m_currentShaderFile = diskPath;
  if(!diskPath.empty())
  {
    m_languageServer.notifyDocumentName(diskPath);
  }

  // Load into code editor
  m_editors[+Editor::eCode].SetText(text);

  // Reset modal state
  // Note that we call GetText() so the line endings are normalized for us,
  // and we don't detect changes if only the line endings have changed.
  m_modal.originalCode       = m_editors[+Editor::eCode].GetText();
  m_modal.confirmationNeeded = true;
  m_modal.fileToLoad.clear();

  const bool success = updateFromSlangCode(false);
  updateDiagnosticMarkers();
  return success;
}

bool Sample::openSlangCodeAndConfig(const std::filesystem::path& slangOrJson)
{
  const std::filesystem::path shaderFilename = std::filesystem::path(slangOrJson).replace_extension(".slang");
  const std::filesystem::path configFilename = std::filesystem::path(slangOrJson).replace_extension(".json");

  m_shaderParams = deserializeShaderConfig(configFilename).parameters;

  const std::string slangCode = nvutils::loadFile(shaderFilename);
  if(slangCode.empty())
  {
    LOGE("Could not open Slang file or file was empty!\n");
    return false;
  }
  return openSlangCode(slangCode, shaderFilename);
}

void Sample::saveShaderAndConfig(const std::filesystem::path& filename, bool saveThumbnail)
{
  // Shader
  std::filesystem::path shaderFilename = filename;
  if(!shaderFilename.has_extension())
  {
    shaderFilename.replace_extension(".slang");
  }

  const std::string slangCode = m_editors[+Editor::eCode].GetText();
  {
    std::ofstream file(shaderFilename, std::ios::binary);
    file.write(slangCode.c_str(), slangCode.size());
  }

  // Config
  {
    ShaderConfigFile config;
    config.parameters  = m_shaderParams;
    config.description = extractDescription(slangCode);
    if(saveThumbnail)
    {
      config.thumbnail = getThumbnailImage();
    }

    // Crop shader parameters to only the ones that were used -- but if the
    // shader didn't compile, save all of them to avoid losing info:
    if(m_resources)
    {
      std::set<std::string> usedUniforms;
      for(const UniformWrite& update : m_resources->uniformUpdates)
      {
        usedUniforms.insert(update.name);
      }
      std::erase_if(config.parameters.uniforms, [&](const auto& kvp) { return !usedUniforms.contains(kvp.first); });

      std::set<std::string> usedStorageBuffers;
      for(const DescriptorWrite& update : m_resources->descriptorSetUpdates)
      {
        if(update.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
        {
          usedStorageBuffers.insert(m_resources->storageBuffers[update.resourceIndex].name);
        }
      }
      std::erase_if(config.parameters.storageBuffers,
                    [&](const auto& kvp) { return !usedStorageBuffers.contains(kvp.first); });
    }

    const std::filesystem::path configFilename = std::filesystem::path(shaderFilename).replace_extension(".json");
    serializeShaderConfig(configFilename, config);
  }

  // Reset modal status
  m_modal.originalCode = slangCode;
}

// Finds, parses, and returns Slang diagnostics in a string.
std::vector<Diagnostic> parseDiagnostics(const std::string& diagnostics)
{
  std::vector<Diagnostic> result;
  if(diagnostics.empty())
  {
    // No diagnostics, exit early
    return result;
  }

  // Slang emits diagnostics in the form `foundPath(lineNumber): errorLevel code: multi-line-message\n`.
  // The way we call the compiler, our foundPath is either empty
  // (this usually indicates redundant messages) or a 40-character hexadecimal
  // hash.

  // First, separate the string into individual diagnostics by breaking on
  // - newline or the start of the string
  // - followed by 0 characters or a 40-digit hex string
  // - followed by an open parenthesis
  static const std::regex diagnosticStartRegex(R"((^|\n)(|[0-9a-z]{40})\()");
  const auto              regexEnd = std::sregex_iterator();
  std::vector<size_t>     diagnosticMarkerStarts;
  std::vector<size_t>     diagnosticMarkerEnds;
  std::vector<bool>       diagnosticsImportant;
  for(std::sregex_iterator outer = std::sregex_iterator(diagnostics.begin(), diagnostics.end(), diagnosticStartRegex);
      outer != regexEnd; ++outer)
  {
    const std::smatch&     match   = *outer;
    const std::ssub_match& hexcode = match[2];
    diagnosticMarkerStarts.push_back(static_cast<size_t>(match.position()));
    diagnosticMarkerEnds.push_back(static_cast<size_t>(match.position() + match.length()));
    diagnosticsImportant.push_back(hexcode.length() > 0);
  }

  // Then we'll parse the individal terms:
  static const std::regex diagnosticParseRegex(R"((\d*)\): ([a-z\s]+)(?:\s(\d+))?: ([\s\S]*))");
  for(size_t i = 0; i < diagnosticsImportant.size(); i++)
  {
    if(!diagnosticsImportant[i])
    {
      continue;
    }
    const size_t diagnosticStart = diagnosticMarkerEnds[i];
    const size_t diagnosticEnd = (i + 1 == diagnosticsImportant.size() ? diagnostics.size() : diagnosticMarkerStarts[i + 1]);
    const size_t safeLength    = std::max(diagnosticStart, diagnosticEnd) - diagnosticStart;
    std::string  diagnosticStr = diagnostics.substr(diagnosticStart, safeLength);
    // Trim newlines from the end of diagnosticStr.
    while('\n' == diagnosticStr.back() || '\r' == diagnosticStr.back())
    {
      diagnosticStr.pop_back();
    }
    // Try to match the entire string against the parse regex.
    std::smatch parsed;
    Diagnostic  diagnostic;
    if(std::regex_match(diagnosticStr, parsed, diagnosticParseRegex))
    {
      // Group 1 is the line number; group 2 is the severity; group 3 is the error code; group 4 is the message.
      // The severity list comes from slang-diagnostic-sink.h's getSeverityName.
      const std::string& severity = parsed[2];
      if(severity == "ignored")
      {
        continue;
      }
      else if(severity == "note")
      {
        diagnostic.level = Diagnostic::Level::eInfo;
      }
      else if(severity == "warning")
      {
        diagnostic.level = Diagnostic::Level::eWarning;
      }
      else  // error, fatal error, internal error, unknown error
      {
        diagnostic.level = Diagnostic::Level::eError;
      }

      diagnostic.text = parsed[4];
      try
      {
        diagnostic.line = std::stoi(parsed[1]);
        if(parsed[3].matched)  // Notes about implicit conversions, for instance, have no error codes
        {
          diagnostic.errorCode = std::stoull(parsed[3]);
        }
      }
      catch(const std::exception& /* unused */)
      {
        assert(false);  // Should never happen
        diagnostic.text += "\n(" TARGET_NAME " could not parse this diagnostic)";
        diagnostic.level = Diagnostic::Level::eError;
      }
    }
    else
    {
      // We couldn't parse it. Luckily, we're in the diagnostic code so we can
      // raise this!
      diagnostic.text  = diagnosticStr + "\n(" TARGET_NAME " could not parse this diagnostic)";
      diagnostic.level = Diagnostic::Level::eError;
    }

    result.push_back(std::move(diagnostic));
  }

  return result;
}

void Sample::updateDiagnosticMarkers()
{
  // Add markers to text editor
  TextEditor& codeEditor = m_editors[+Editor::eCode];
  codeEditor.ClearMarkers();
  // The last marker placed on a line gets drawn over the rest, so
  // handle markers in order of severity.
  for(Diagnostic::Level level : {Diagnostic::Level::eInfo, Diagnostic::Level::eWarning, Diagnostic::Level::eError})
  {
    ImU32 color{};
    switch(level)
    {
      case Diagnostic::Level::eInfo:
        color = IM_COL32(102, 102, 102, 128);
        break;
      case Diagnostic::Level::eWarning:
        color = IM_COL32(142, 100, 0, 128);
        break;
      case Diagnostic::Level::eError:
      default:
        color = IM_COL32(200, 0, 0, 128);
        break;
    }

    for(Diagnostic& diagnostic : m_diagnostics)
    {
      if(diagnostic.level == level)
      {
        // This -1 here is because addMarker's lines are 0-indexed
        codeEditor.AddMarker(std::max(0, diagnostic.line - 1), color, color, diagnostic.text, diagnostic.text);
      }
    }
  }
}

//-----------------------------------------------------------------------------
// Unhandled exception filters

static void errorHandler() noexcept
{
// Something terrible has happened, probably a segfault!
// Because of that, we'll try to be safer than usual.
// Save the state of the app to files that can be read to create an issue
// either for the compiler or the vk_slang_editor sample.

// Note that because logging isn't asynchronous-safe, this function
// relies on undefined behavior. But, well, we're crashing anyways.
// And if we segfaulted, then this iat least a synchronous signal.
#if 0  // To be enabled once MR is merged
  nvutils::Logger::getInstance().setFileFlush(true);
#endif
  LOGE(
      "A fatal exception occurred! This app will now attempt to save the shader you were working on. If this was a "
      "shader compiler crash, please create an issue for it at https://github.com/shader-slang/slang/issues and attach "
      "the shader. If the sample crashed for a different reason, please create an issue at "
      "https://github.com/nvpro-samples/" TARGET_NAME ".\n");

  if(!s_sample)
  {
    LOGE("Could not get the last shader because the sample object was destroyed!\n");
  }
  else
  {
    try
    {
      const std::string filename = "crash-" + pathSafeTimeString() + ".slang";
      s_sample->saveShaderAndConfig(filename.c_str(), false);
      LOGOK("Shader saved to %s.\n", filename.c_str());
    }
    catch(const std::exception& e)
    {
      LOGE("That failed as well, with exception: %s\n", e.what());
    }
  }
}

#ifdef _WIN32
static LONG WINAPI unhandledExceptionFilter(_EXCEPTION_POINTERS* exceptionInfo)
{
  // If we get here, we need to turn off raise(SIGTRAP) in the logger because
  // we're printing to LOGE.
  // We do this here for symmetry with the Linux path.
  nvutils::Logger::getInstance().breakOnError(false);
  errorHandler();
  // Return; execute the associated exception handler
  // (usually results in process termination)
  return EXCEPTION_EXECUTE_HANDLER;
}
#else  // Linux
static void linuxSignalHandler(int signal, siginfo_t* signalInfo, void* unused)
{
  nvutils::Logger::getInstance().breakOnError(false);
  LOGE("Received signal %d:\n", signal);
  if(signalInfo)
  {
    LOGE("si_errno: %d\n", signalInfo->si_errno);
    LOGE("si_code: %d\n", signalInfo->si_code);
    LOGE("si_addr: %p\n", signalInfo->si_addr);
  }
  errorHandler();
  exit(EXIT_FAILURE);
}
#endif

//-----------------------------------------------------------------------------
// Application entrypoint

int main(int argc, const char** argv)
{
// Set up exception callbacks
#ifdef _WIN32
  SetUnhandledExceptionFilter(unhandledExceptionFilter);
#else
  struct sigaction sigInfo;
  sigInfo.sa_flags = SA_SIGINFO;
  sigemptyset(&sigInfo.sa_mask);
  sigInfo.sa_sigaction = linuxSignalHandler;
  sigaction(SIGSEGV, &sigInfo, NULL);
#endif

// On release builds, don't break on errors, since users might cause them with
// custom shaders and sometimes the driver can keep going.
#ifdef NDEBUG
  nvutils::Logger::getInstance().breakOnError(false);
#else
  nvutils::Logger::getInstance().setFileFlush(true);
#endif

  s_sample                = std::make_shared<Sample>();
  const VkResult exitcode = s_sample->run();
  s_sample->deinit();

  return (VK_SUCCESS == exitcode) ? EXIT_SUCCESS : EXIT_FAILURE;
}
