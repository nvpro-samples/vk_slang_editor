cmake_minimum_required(VERSION 3.21...3.31)
# vk_slang_editor uses semantic versioning.
project(vk_slang_editor LANGUAGES CXX VERSION 0.9.0)

# Add the cmake folder to the module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(NvproCore2 REQUIRED)

find_package(VulkanExtra) # TODO: Figure out why this makes SPIRV-Tools-shared exist when it didn't before

download_files(FILENAMES daytime.hdr shaderball.glb starmap.dds std_env.hdr) # Default resources

# ImGuiColorTextEdit
set(ImGuiColorTextEdit_DIR ${CMAKE_CURRENT_LIST_DIR}/extern/ImGuiColorTextEdit)
add_library(ImGuiColorTextEdit STATIC
  ${ImGuiColorTextEdit_DIR}/TextEditor.cpp
  ${ImGuiColorTextEdit_DIR}/TextEditor.h
)
target_include_directories(ImGuiColorTextEdit
  PUBLIC ${NvproCore2_ROOT}/nvpro_core2/third_party/imgui # Since it #includes <imgui.h>
  PUBLIC ${NvproCore2_ROOT}/nvpro_core2/third_party # Since it #includes <GLFW/glfw3.h>
  INTERFACE ${CMAKE_CURRENT_LIST_DIR}/extern
)
set_property(TARGET ImGuiColorTextEdit PROPERTY FOLDER "ThirdParty")

# Source files for this project
set(SOURCE_FILES
  src/gui_reflection.cpp
  src/gui_reflection.h
  src/io_image.cpp
  src/io_image.h
  src/io_params.cpp
  src/io_params.h
  src/language.cpp
  src/language.h
  src/language_highlight_autogen.cpp
  src/language_highlight_autogen.h
  src/resources.cpp
  src/resources.h
  src/sample.h
  src/sample_gui.cpp
  src/sample_render.cpp
  src/utilities.cpp
  src/utilities.h
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
target_compile_definitions(${PROJECT_NAME} PRIVATE TARGET_VERSION_STRING="${CMAKE_PROJECT_VERSION}")

# Link with nvpro_core2 libraries we need
target_link_libraries(${PROJECT_NAME} PRIVATE
  ImGuiColorTextEdit
  nvpro2::nvapp
  nvpro2::nvgui
  nvpro2::nvimageformats
  nvpro2::nvslang
  nvpro2::nvvk
  nvpro2::nvvkgltf
  SPIRV-Tools-shared
)
# nvaftermath?
# nvgpu_monitor?

add_project_definitions(${PROJECT_NAME})

# Copy runtime dependencies to build and install directories
copy_to_runtime_and_install(${PROJECT_NAME}
  LOCAL_DIRS "${CMAKE_CURRENT_LIST_DIR}/examples" "${CMAKE_CURRENT_LIST_DIR}/media"
  FILES ${Slang_GLSLANG} ${Slang_GLSL_MODULE} ${Slang_DLL}
  PROGRAMS ${Slang_SLANGD_EXECUTABLE}
  AUTO
)
