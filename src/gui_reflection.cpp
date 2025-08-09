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

// Slang reflection GUI

#include "gui_reflection.h"

#include <imgui/imgui.h>
#include <slang.h>

#include <array>
#include <inttypes.h>
#include <string>
#include <unordered_map>

//-------------------------------------------------------------------------------------------------------------------//
// To-string functions

const std::string slangBindingTypeToString(slang::BindingType bindingType)
{
  std::string            result = "";
  const SlangBindingType raw    = static_cast<SlangBindingType>(bindingType);
  if(raw & SLANG_BINDING_TYPE_MUTABLE_FLAG)
  {
    result = "mutable";
  }

  switch(raw & SLANG_BINDING_TYPE_BASE_MASK)
  {
    case SLANG_BINDING_TYPE_SAMPLER:
      result += " sampler";
      break;
    case SLANG_BINDING_TYPE_TEXTURE:
      result += " texture";
      break;
    case SLANG_BINDING_TYPE_CONSTANT_BUFFER:
      result += " constant buffer";
      break;
    case SLANG_BINDING_TYPE_PARAMETER_BLOCK:
      result += " parameter block";
      break;
    case SLANG_BINDING_TYPE_TYPED_BUFFER:
      result += " typed buffer";
      break;
    case SLANG_BINDING_TYPE_RAW_BUFFER:
      result += " raw buffer";
      break;
    case SLANG_BINDING_TYPE_COMBINED_TEXTURE_SAMPLER:
      result += " combined texture sampler";
      break;
    case SLANG_BINDING_TYPE_INPUT_RENDER_TARGET:
      result += " input render target";
      break;
    case SLANG_BINDING_TYPE_INLINE_UNIFORM_DATA:
      result += " inline uniform data";
      break;
    case SLANG_BINDING_TYPE_RAY_TRACING_ACCELERATION_STRUCTURE:
      result += " ray tracing acceleration structure";
      break;
    case SLANG_BINDING_TYPE_VARYING_INPUT:
      result += " varying input";
      break;
    case SLANG_BINDING_TYPE_VARYING_OUTPUT:
      result += " varying output";
      break;
    case SLANG_BINDING_TYPE_EXISTENTIAL_VALUE:
      result += " existential value";
      break;
    case SLANG_BINDING_TYPE_PUSH_CONSTANT:
      result += " push constant";
      break;
  }
  if(result.empty())
  {
    return "unknown";
  }
  return result;
}

const char* slangCategoryToString(slang::ParameterCategory category)
{
  const static std::unordered_map<SlangParameterCategoryIntegral, const char*> s_slangCategoryToString{
      {SLANG_PARAMETER_CATEGORY_NONE, "none"},
      {SLANG_PARAMETER_CATEGORY_MIXED, "mixed"},
      {SLANG_PARAMETER_CATEGORY_CONSTANT_BUFFER, "constant buffer"},
      {SLANG_PARAMETER_CATEGORY_SHADER_RESOURCE, "shader resource"},
      {SLANG_PARAMETER_CATEGORY_UNORDERED_ACCESS, "unordered access"},
      {SLANG_PARAMETER_CATEGORY_VARYING_INPUT, "varying input"},
      {SLANG_PARAMETER_CATEGORY_VARYING_OUTPUT, "varying output"},
      {SLANG_PARAMETER_CATEGORY_SAMPLER_STATE, "sampler state"},
      {SLANG_PARAMETER_CATEGORY_UNIFORM, "uniform"},
      {SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT, "descriptor table slot"},
      {SLANG_PARAMETER_CATEGORY_SPECIALIZATION_CONSTANT, "specialization constant"},
      {SLANG_PARAMETER_CATEGORY_PUSH_CONSTANT_BUFFER, "push constant buffer"},
      {SLANG_PARAMETER_CATEGORY_REGISTER_SPACE, "register space"},
      {SLANG_PARAMETER_CATEGORY_GENERIC, "generic resource"},
      {SLANG_PARAMETER_CATEGORY_RAY_PAYLOAD, "ray payload"},
      {SLANG_PARAMETER_CATEGORY_HIT_ATTRIBUTES, "hit attributes"},
      {SLANG_PARAMETER_CATEGORY_CALLABLE_PAYLOAD, "callable payload"},
      {SLANG_PARAMETER_CATEGORY_SHADER_RECORD, "shader record"},
      {SLANG_PARAMETER_CATEGORY_EXISTENTIAL_TYPE_PARAM, "existential type parameter"},
      {SLANG_PARAMETER_CATEGORY_EXISTENTIAL_OBJECT_PARAM, "existential object parameter"},
      {SLANG_PARAMETER_CATEGORY_SUB_ELEMENT_REGISTER_SPACE, "sub-element register space"},
      {SLANG_PARAMETER_CATEGORY_SUBPASS, "input attachment index"},
      {SLANG_PARAMETER_CATEGORY_METAL_TEXTURE, "Metal texture"},
      {SLANG_PARAMETER_CATEGORY_METAL_ARGUMENT_BUFFER_ELEMENT, "Metal argument buffer"},
      {SLANG_PARAMETER_CATEGORY_METAL_ATTRIBUTE, "Metal attribute"},
      {SLANG_PARAMETER_CATEGORY_METAL_PAYLOAD, "Metal payload"},
  };
  const auto& it = s_slangCategoryToString.find(static_cast<SlangParameterCategoryIntegral>(category));
  if(it == s_slangCategoryToString.end())
  {
    return "<unknown>";
  }
  return it->second;
}

const char* slangImageFormatToString(SlangImageFormat imageFormat)
{
  const static std::unordered_map<SlangImageFormat, const char*> s_slangImageFormatToString{
#define SLANG_FORMAT(NAME, DESC) {SLANG_IMAGE_FORMAT_##NAME, #NAME},
#include <slang-image-format-defs.h>
#undef SLANG_FORMAT
  };
  const auto& it = s_slangImageFormatToString.find(imageFormat);
  if(it == s_slangImageFormatToString.end())
  {
    return "<unknown>";
  }
  return it->second;
}

const char* slangKindToString(slang::TypeReflection::Kind kind)
{
  const static std::unordered_map<SlangTypeKindIntegral, const char*> s_slangKindToString{
      {SLANG_TYPE_KIND_NONE, "none"},
      {SLANG_TYPE_KIND_STRUCT, "struct"},
      {SLANG_TYPE_KIND_ARRAY, "array"},
      {SLANG_TYPE_KIND_MATRIX, "matrix"},
      {SLANG_TYPE_KIND_VECTOR, "vector"},
      {SLANG_TYPE_KIND_SCALAR, "scalar"},
      {SLANG_TYPE_KIND_CONSTANT_BUFFER, "constant buffer"},
      {SLANG_TYPE_KIND_RESOURCE, "resource"},
      {SLANG_TYPE_KIND_SAMPLER_STATE, "sampler state"},
      {SLANG_TYPE_KIND_TEXTURE_BUFFER, "texture buffer"},
      {SLANG_TYPE_KIND_SHADER_STORAGE_BUFFER, "shader storage buffer"},
      {SLANG_TYPE_KIND_PARAMETER_BLOCK, "parameter block"},
      {SLANG_TYPE_KIND_GENERIC_TYPE_PARAMETER, "generic type parameter"},
      {SLANG_TYPE_KIND_INTERFACE, "interface"},
      {SLANG_TYPE_KIND_OUTPUT_STREAM, "output stream"},
      {SLANG_TYPE_KIND_SPECIALIZED, "specialized"},
      {SLANG_TYPE_KIND_FEEDBACK, "feedback"},
      {SLANG_TYPE_KIND_POINTER, "pointer"},
      {SLANG_TYPE_KIND_DYNAMIC_RESOURCE, "dynamic resource"},
  };
  const auto& it = s_slangKindToString.find(static_cast<SlangTypeKindIntegral>(kind));
  if(it == s_slangKindToString.end())
  {
    return "<unknown>";
  }
  return it->second;
}

const char* slangScalarTypeToString(slang::TypeReflection::ScalarType scalar)
{
  const static std::unordered_map<SlangScalarTypeIntegral, const char*> s_slangScalarTypeToString{
      {SLANG_SCALAR_TYPE_NONE, "none"},       {SLANG_SCALAR_TYPE_VOID, "void"},
      {SLANG_SCALAR_TYPE_BOOL, "bool"},       {SLANG_SCALAR_TYPE_INT32, "int32"},
      {SLANG_SCALAR_TYPE_UINT32, "uint32"},   {SLANG_SCALAR_TYPE_INT64, "int64"},
      {SLANG_SCALAR_TYPE_UINT64, "uint64"},   {SLANG_SCALAR_TYPE_FLOAT16, "float16"},
      {SLANG_SCALAR_TYPE_FLOAT32, "float32"}, {SLANG_SCALAR_TYPE_FLOAT64, "float64"},
      {SLANG_SCALAR_TYPE_INT8, "int8"},       {SLANG_SCALAR_TYPE_UINT8, "uint8"},
      {SLANG_SCALAR_TYPE_INT16, "int16"},     {SLANG_SCALAR_TYPE_UINT16, "uint16"},
  };
  const auto& it = s_slangScalarTypeToString.find(static_cast<SlangScalarTypeIntegral>(scalar));
  if(it == s_slangScalarTypeToString.end())
  {
    return "<unknown>";
  }
  return it->second;
}

const char* slangStageToString(SlangStage stage)
{
  static const std::unordered_map<SlangStage, const char*> s_slangStageToString{
      {SLANG_STAGE_NONE, "none"},
      {SLANG_STAGE_VERTEX, "vertex"},
      {SLANG_STAGE_HULL, "hull"},
      {SLANG_STAGE_DOMAIN, "domain"},
      {SLANG_STAGE_GEOMETRY, "geometry"},
      {SLANG_STAGE_FRAGMENT, "fragment"},
      {SLANG_STAGE_COMPUTE, "compute"},
      {SLANG_STAGE_RAY_GENERATION, "ray generation"},
      {SLANG_STAGE_INTERSECTION, "intersection"},
      {SLANG_STAGE_ANY_HIT, "any hit"},
      {SLANG_STAGE_CLOSEST_HIT, "closest hit"},
      {SLANG_STAGE_MISS, "miss"},
      {SLANG_STAGE_CALLABLE, "callable"},
      {SLANG_STAGE_MESH, "mesh"},
      {SLANG_STAGE_AMPLIFICATION, "amplification"},
  };
  const auto& it = s_slangStageToString.find(stage);
  if(it == s_slangStageToString.end())
  {
    return "<unknown>";
  }
  return it->second;
}

//-------------------------------------------------------------------------------------------------------------------//
// These functions can all recursively call each other, so we need to forward-declare them:
void guiAttributeReflection(slang::Attribute* attribute);
void guiDeclReflection(slang::DeclReflection* decl);
void guiEntryPointReflection(slang::EntryPointReflection* entryRefl);
void guiFunctionReflection(slang::FunctionReflection* funcRefl);
void guiGenericReflection(slang::GenericReflection* genRefl);
void guiTypeReflection(slang::TypeReflection* typeRefl);
void guiTypeLayoutReflection(slang::TypeLayoutReflection* typeRefl);
void guiVariableReflection(slang::VariableReflection* varRefl);
void guiVariableLayoutReflection(slang::VariableLayoutReflection* varRefl);

template <class T>
void guiListModifiers(T* obj)
{
  assert(obj);
  // TODO: Surely there must be a better way to do this...
  std::string                                                 label = "Modifiers:";
  std::array<std::pair<slang::Modifier::ID, const char*>, 11> modifiers{
      std::pair<slang::Modifier::ID, const char*>{slang::Modifier::Shared, "shared"},
      {slang::Modifier::NoDiff, "no_diff"},
      {slang::Modifier::Static, "static"},
      {slang::Modifier::Const, "const"},
      {slang::Modifier::Export, "export"},
      {slang::Modifier::Extern, "extern"},
      {slang::Modifier::Differentiable, "Differentiable"},
      {slang::Modifier::Mutating, "Mutating"},
      {slang::Modifier::In, "in"},
      {slang::Modifier::Out, "out"},
      {slang::Modifier::InOut, "inout"},
  };
  size_t modifiersFound = 0;
  for(const auto& modifier : modifiers)
  {
    if(obj->findModifier(modifier.first))
    {
      label = label + " " + modifier.second;
      modifiersFound++;
    }
  }

  if(modifiersFound != 0)
  {
    ImGui::Text("%s", label.c_str());
  }
}

template <class T>
void guiListUserAttributes(T* obj)
{
  const unsigned int userAttributeCount = obj->getUserAttributeCount();
  if(userAttributeCount > 0)
  {
    ImGui::Text("User attributes (%u):", userAttributeCount);
    for(unsigned int attribIdx = 0; attribIdx < userAttributeCount; attribIdx++)
    {
      slang::Attribute* attrib     = obj->getUserAttributeByIndex(attribIdx);
      const std::string attribName = std::to_string(attribIdx) + ": " + std::string(attrib->getName());
      if(ImGui::TreeNode(attribName.c_str()))
      {
        guiAttributeReflection(attrib);
        ImGui::TreePop();
      }
    }
  }
}

// Sometimes types don't have names!
template <class T>
const char* guiGetName(T* obj)
{
  if(!obj)
    return "<null>";
  const char* name = obj->getName();
  if(!name)
    return "<nameless>";
  return name;
}

// The ordering rule within each function is to list names first, followed by
// leaf properties, then tree properties, each alphabetically.

void guiAttributeReflection(slang::Attribute* attribute)
{
  assert(attribute);
  const uint32_t argumentCount = attribute->getArgumentCount();
  ImGui::Text("Arguments (%u):", argumentCount);
  for(uint32_t i = 0; i < argumentCount; i++)
  {
    slang::TypeReflection* argument  = nullptr;
    int                    tempInt   = 0;
    float                  tempFloat = 0.f;
    // TODO: Is there a way to check the type in advance?
    if(const char* str = attribute->getArgumentValueString(i, nullptr))
    {
      ImGui::Text("%u: %s", i, str);
    }
    else if(SLANG_SUCCEEDED(attribute->getArgumentValueInt(i, &tempInt)))
    {
      ImGui::Text("%u: %i", i, tempInt);
    }
    else if(SLANG_SUCCEEDED(attribute->getArgumentValueFloat(i, &tempFloat)))
    {
      ImGui::Text("%u: %f", i, tempFloat);
    }
    else if(!(argument = attribute->getArgumentType(i)))
    {
      assert(false);
      ImGui::Text("%u: <nullptr>", i);
    }
    else
    {
      const std::string nodeLabel = std::to_string(i) + ": " + argument->getName() + " [...]";
      if(ImGui::TreeNode(nodeLabel.c_str()))
      {
        guiTypeReflection(argument);
        ImGui::TreePop();
      }
    }
  }
}

void guiDeclReflection(slang::DeclReflection* decl)
{
  assert(decl);

  using KindT = slang::DeclReflection::Kind;
  switch(decl->getKind())
  {
    case KindT::Struct:
      ImGui::Text("Kind: struct");
      break;
    case KindT::Func:
      ImGui::Text("Kind: function");
      guiFunctionReflection(decl->asFunction());
      break;
    case KindT::Module:
      ImGui::Text("Kind: module");
      break;
    case KindT::Generic:
      ImGui::Text("Kind: generic");
      guiGenericReflection(decl->asGeneric());
      break;
    case KindT::Variable:
      ImGui::Text("Kind: variable");
      guiVariableReflection(decl->asVariable());
      break;
    case KindT::Namespace:
      ImGui::Text("Kind: namespace");
      break;
    case KindT::Unsupported:
      ImGui::Text("Kind: unsupported");
      break;
  }

  const unsigned int childrenCount  = decl->getChildrenCount();
  const std::string  childLabelName = std::string("Children (") + std::to_string(childrenCount) + ")";
  if(ImGui::TreeNode(childLabelName.c_str()))
  {
    for(unsigned i = 0; i < childrenCount; i++)
    {
      slang::DeclReflection* child = decl->getChild(i);
      const std::string      label = std::to_string(i) + ": " + guiGetName(child);
      if(ImGui::TreeNode(label.c_str()))
      {
        guiDeclReflection(child);
        ImGui::TreePop();
      }
    }
    ImGui::TreePop();
  }
}


void guiEntryPointReflection(slang::EntryPointReflection* entryRefl)
{
  assert(entryRefl);
  ImGui::Text("Name override: %s", entryRefl->getNameOverride());

  ImGui::Text("Stage: %s", slangStageToString(entryRefl->getStage()));

  std::array<SlangUInt, 3> groupSize{};
  entryRefl->getComputeThreadGroupSize(groupSize.size(), groupSize.data());
  ImGui::Text("Group size: [%" PRIu64 ", %" PRIu64 ", %" PRIu64 "]", groupSize[0], groupSize[1], groupSize[2]);

  SlangUInt waveSize{};
  entryRefl->getComputeWaveSize(&waveSize);
  ImGui::Text("Wave size: %" PRIu64, waveSize);

  ImGui::Text("Uses any sample-rate input: %s", entryRefl->usesAnySampleRateInput() ? "true" : "false");
  ImGui::Text("Has default constant buffer: %s", entryRefl->hasDefaultConstantBuffer() ? "true" : "false");

  slang::FunctionReflection* functionRefl = entryRefl->getFunction();
  if(functionRefl && ImGui::TreeNode("Function"))
  {
    guiFunctionReflection(functionRefl);
    ImGui::TreePop();
  }

  const unsigned    parameterCount = entryRefl->getParameterCount();
  const std::string paramLabelName = std::string("Parameters (") + std::to_string(parameterCount) + ")";
  if(ImGui::TreeNode(paramLabelName.c_str()))
  {
    for(unsigned i = 0; i < parameterCount; i++)
    {
      slang::VariableLayoutReflection* parameter = entryRefl->getParameterByIndex(i);
      const std::string                nodeLabel = std::to_string(i) + ": " + parameter->getName();
      if(ImGui::TreeNode(nodeLabel.c_str()))
      {
        guiVariableLayoutReflection(parameter);
        ImGui::TreePop();
      }
    }
    ImGui::TreePop();
  }

  slang::VariableLayoutReflection* resultVarLayoutRefl = entryRefl->getResultVarLayout();
  if(resultVarLayoutRefl && ImGui::TreeNode("Result variable layout"))
  {
    guiVariableLayoutReflection(resultVarLayoutRefl);
    ImGui::TreePop();
  }

  slang::TypeLayoutReflection* typeLayoutRefl = entryRefl->getTypeLayout();
  if(typeLayoutRefl && ImGui::TreeNode("Type layout"))
  {
    guiTypeLayoutReflection(typeLayoutRefl);
    ImGui::TreePop();
  }

  slang::VariableLayoutReflection* varLayoutRefl = entryRefl->getVarLayout();
  if(varLayoutRefl && ImGui::TreeNode("Variable layout"))
  {
    guiVariableLayoutReflection(varLayoutRefl);
    ImGui::TreePop();
  }
}

void guiFunctionReflection(slang::FunctionReflection* funcRefl)
{
  assert(funcRefl);
  guiListModifiers(funcRefl);

  slang::TypeReflection* returnType      = funcRefl->getReturnType();
  const std::string      returnTypeLabel = std::string("Return type: ") + returnType->getName() + " [...]";
  if(ImGui::TreeNode(returnTypeLabel.c_str()))
  {
    guiTypeReflection(returnType);
    ImGui::TreePop();
  }

  const unsigned int parameterCount = funcRefl->getParameterCount();
  const std::string  paramLabelName = std::string("Parameters (") + std::to_string(parameterCount) + ")";
  if(ImGui::TreeNode(paramLabelName.c_str()))
  {
    for(unsigned int parameterIdx = 0; parameterIdx < parameterCount; parameterIdx++)
    {
      slang::VariableReflection* param    = funcRefl->getParameterByIndex(parameterIdx);
      const std::string          nodeName = std::to_string(parameterIdx) + ": " + std::string(param->getName());
      if(ImGui::TreeNode(nodeName.c_str()))
      {
        guiVariableReflection(param);
        ImGui::TreePop();
      }
    }
    ImGui::TreePop();
  }

  guiListUserAttributes(funcRefl);
}

void guiGenericReflection(slang::GenericReflection* genRefl)
{
  assert(genRefl);

  const unsigned int numTypeParameters = genRefl->getTypeParameterCount();
  const std::string  typeParamLabel    = std::string("Type parameters (") + std::to_string(numTypeParameters) + ")";
  if(ImGui::TreeNode(typeParamLabel.c_str()))
  {
    for(unsigned int i = 0; i < numTypeParameters; i++)
    {
      slang::VariableReflection* typeParam = genRefl->getTypeParameter(i);

      slang::TypeReflection* typeConcrete = genRefl->getConcreteType(typeParam);
      if(typeConcrete)
      {
        const std::string typeConcreteLabel = std::string("Concrete type: ") + typeConcrete->getName() + " [...]";
        if(ImGui::TreeNode(typeConcreteLabel.c_str()))
        {
          guiTypeReflection(typeConcrete);
          ImGui::TreePop();
        }
        const int64_t concreteVal = genRefl->getConcreteIntVal(typeParam);
        ImGui::Text("Concrete int value: %" PRId64, concreteVal);
      }

      const std::string nodeName = std::to_string(i) + ": " + std::string(typeParam->getName());
      if(ImGui::TreeNode(nodeName.c_str()))
      {
        unsigned int      numConstraints   = genRefl->getTypeParameterConstraintCount(typeParam);
        const std::string constraintsLabel = std::string("Constraints (") + std::to_string(numConstraints) + ")";
        if(ImGui::TreeNode(constraintsLabel.c_str()))
        {
          for(unsigned int j = 0; j < numConstraints; j++)
          {
            slang::TypeReflection* constraintType = genRefl->getTypeParameterConstraintType(typeParam, j);
            const std::string      constNodeName  = std::to_string(j) + ": " + constraintType->getName() + " [...]";
            if(ImGui::TreeNode(constNodeName.c_str()))
            {
              guiTypeReflection(constraintType);
              ImGui::TreePop();
            }
          }
          ImGui::TreePop();
        }

        guiVariableReflection(typeParam);

        ImGui::TreePop();
      }
    }
    ImGui::TreePop();
  }

  slang::DeclReflection* innerDecl = genRefl->getInnerDecl();
  if(innerDecl)
  {
    const std::string label = std::string("Inner decl: ") + innerDecl->getName();
    if(ImGui::TreeNode(label.c_str()))
    {
      guiDeclReflection(innerDecl);
      ImGui::TreePop();
    }
  }

  // TODO: Is getOuterGenericContainer interesting?
}

void guiTypeReflection(slang::TypeReflection* typeRefl)
{
  if(!typeRefl)
    return;  // Untyped

  using KindT = slang::TypeReflection::Kind;

  {  // Getting the full name requires some work
    ISlangBlob* fullName = nullptr;
    SlangResult result   = typeRefl->getFullName(&fullName);
    if(!fullName || SLANG_FAILED(result))
    {
      assert(false);
      ImGui::Text("Full name: <Slang error code %i>", result);
    }
    else
    {
      ImGui::Text("Full name: %s", reinterpret_cast<const char*>(fullName->getBufferPointer()));
    }

    if(fullName)
    {
      fullName->Release();
    }
  }

  const KindT kind = typeRefl->getKind();
  ImGui::Text("Kind: %s", slangKindToString(kind));

  if(kind == KindT::Struct)
  {
    const unsigned int fieldCount = typeRefl->getFieldCount();
    ImGui::Text("Fields (%u):", fieldCount);
    for(unsigned int fieldIdx = 0; fieldIdx < fieldCount; fieldIdx++)
    {
      slang::VariableReflection* field      = typeRefl->getFieldByIndex(fieldIdx);
      const std::string          fieldLabel = std::to_string(fieldIdx) + ": " + field->getName();
      if(ImGui::TreeNode(fieldLabel.c_str()))
      {
        guiVariableReflection(field);
        ImGui::TreePop();
      }
    }
  }
  if(kind == KindT::Array)
  {
    const size_t elementCount = typeRefl->getElementCount();
    ImGui::Text("Element count: %zu", elementCount);
  }
  slang::TypeReflection* elementType = typeRefl->getElementType();
  if(elementType)  // TODO: determine exact conditions
  {
    const std::string elementLabel = std::string("Element type: ") + elementType->getName();
    if(ImGui::TreeNode(elementLabel.c_str()))
    {
      guiTypeReflection(elementType);
      ImGui::TreePop();
    }
  }
  if(kind == KindT::Matrix || kind == KindT::Vector)
  {
    ImGui::Text("Row count: %u", typeRefl->getRowCount());
    ImGui::Text("Column count: %u", typeRefl->getColumnCount());
    const char* scalarTypeName = slangScalarTypeToString(typeRefl->getScalarType());
    ImGui::Text("Scalar type: %s", scalarTypeName);
  }
  // TODO: Determine exact conditions
  if(kind == KindT::Resource || kind == KindT::Feedback || kind == KindT::Pointer || kind == KindT::DynamicResource)
  {
    // Resource shape: "multisample shadow feedback 2D texture array"
    const SlangResourceShape shape      = typeRefl->getResourceShape();
    std::string              shapeLabel = "Resource shape:";
    if(shape == SlangResourceShape::SLANG_RESOURCE_NONE)
    {
      shapeLabel += " <none>";
    }
    else
    {
      if(shape & SLANG_TEXTURE_MULTISAMPLE_FLAG)
      {
        shapeLabel += " multisample";
      }
      if(shape & SLANG_TEXTURE_SHADOW_FLAG)  // TODO : Find out when this can happen
      {
        shapeLabel += " shadow";
      }
      if(shape & SLANG_TEXTURE_FEEDBACK_FLAG)
      {
        shapeLabel += " feedback";
      }
      switch(shape & SLANG_RESOURCE_BASE_SHAPE_MASK)
      {
        case SLANG_TEXTURE_1D:
          shapeLabel += " 1D texture";
          break;
        case SLANG_TEXTURE_2D:
          shapeLabel += " 2D texture";
          break;
        case SLANG_TEXTURE_3D:
          shapeLabel += " 3D texture";
          break;
        case SLANG_TEXTURE_CUBE:
          shapeLabel += " cube texture";
          break;
        case SLANG_TEXTURE_BUFFER:
          shapeLabel += " buffer texture";
          break;
        case SLANG_STRUCTURED_BUFFER:
          shapeLabel += " structured buffer";
          break;
        case SLANG_BYTE_ADDRESS_BUFFER:
          shapeLabel += " byte address buffer";
          break;
        case SLANG_ACCELERATION_STRUCTURE:
          shapeLabel += " acceleration structure";
          break;
        case SLANG_TEXTURE_SUBPASS:
          shapeLabel += " subpass";
          break;
        case SLANG_RESOURCE_UNKNOWN:
        default:
          shapeLabel += " unknown";
          break;
      }
      if(shape & SLANG_TEXTURE_ARRAY_FLAG)
      {
        shapeLabel += " array";
      }
    }
    ImGui::Text("%s", shapeLabel.c_str());

    // Resource access
    const SlangResourceAccess access      = typeRefl->getResourceAccess();
    std::string               accessLabel = "Resource access:";
    switch(access)
    {
      case SLANG_RESOURCE_ACCESS_NONE:
        accessLabel += " none";
        break;
      case SLANG_RESOURCE_ACCESS_READ:
        accessLabel += " read";
        break;
      case SLANG_RESOURCE_ACCESS_READ_WRITE:
        accessLabel += " read/write";
        break;
      case SLANG_RESOURCE_ACCESS_RASTER_ORDERED:
        accessLabel += " raster ordered";
        break;
      case SLANG_RESOURCE_ACCESS_APPEND:
        accessLabel += " append";
        break;
      case SLANG_RESOURCE_ACCESS_CONSUME:
        accessLabel += " consume";
        break;
      case SLANG_RESOURCE_ACCESS_WRITE:
        accessLabel += " write";
        break;
      case SLANG_RESOURCE_ACCESS_FEEDBACK:
        accessLabel += " feedback";
        break;
      case SLANG_RESOURCE_ACCESS_UNKNOWN:
      default:
        accessLabel += " unknown";
        break;
    }
    ImGui::Text("%s", accessLabel.c_str());

    // Result type
    slang::TypeReflection* resultType = typeRefl->getResourceResultType();
    if(resultType)
    {
      const std::string resultTypeLabel = std::string("Result type: ") + resultType->getName() + " [...]";
      if(ImGui::TreeNode(resultTypeLabel.c_str()))
      {
        guiTypeReflection(resultType);
        ImGui::TreePop();
      }
    }
  }

  guiListUserAttributes(typeRefl);
}

void guiTypeLayoutReflection(slang::TypeLayoutReflection* typeRefl)
{
  assert(typeRefl);

  // Type...
  // Optional explicit counter + binding range offset
  // Generic param index (only for generics)
  // Matrix layout (only for matrices)
  // For each category:
  // - size
  // - stride
  // - alignment
  // - element stride (for arrays)
  // For arrays:
  // - element type layout
  // - element variable layout
  // For each field (for structs):
  // - binding range offset
  // - VariableLayoutReflection
  // For each binding range:
  // - BindingType
  // - Count
  // - Descriptor set index
  // - Descriptor range count
  // - First descriptor range index
  // - Image format
  // - Is specializable
  // - Leaf type layout
  // - Leaf type variable
  // For each descriptor set:
  // - Space offset
  // - For each descriptor range:
  //   - index offset
  //   - descriptor count
  //   - Category
  // For each sub-object range:
  // - binding range index
  // - space offset
  // - range offset
  // TODO: What are binding ranges vs. descriptor sets vs. sub-objects?

  const std::string typeReflLabel = std::string("Type: ") + guiGetName(typeRefl) + " [...]";
  if(ImGui::TreeNode(typeReflLabel.c_str()))
  {
    guiTypeReflection(typeRefl->getType());
    ImGui::TreePop();
  }

  const slang::TypeReflection::Kind kind = typeRefl->getKind();

  if(slang::VariableLayoutReflection* variableLayout = typeRefl->getContainerVarLayout())
  {
    if(ImGui::TreeNode("Container variable layout"))
    {
      guiVariableLayoutReflection(variableLayout);
      ImGui::TreePop();
    }
  }

  // Try ElementVarLayout first; if that doesn't work (e.g. structured buffers),
  // then do ElementTypeLayout.
  // See https://docs.shader-slang.org/en/latest/external/slang/docs/user-guide/09-reflection.html#pitfalls-to-avoid
  // and https://github.com/shader-slang/slang/blob/d39590228241cb42d72f493f6f484c5ea93df934/source/slang/slang-reflection-json.cpp#L726
  if(slang::VariableLayoutReflection* variableLayout = typeRefl->getElementVarLayout())
  {
    if(ImGui::TreeNode("Element variable layout"))
    {
      guiVariableLayoutReflection(variableLayout);
      ImGui::TreePop();
    }
  }
  else if(slang::TypeLayoutReflection* elementTypeLayout = typeRefl->getElementTypeLayout())
  {
    const std::string elementTypeLabel = std::string("Element type layout: ") + elementTypeLayout->getName() + " [...]";
    if(ImGui::TreeNode(elementTypeLabel.c_str()))
    {
      guiTypeLayoutReflection(elementTypeLayout);
      ImGui::TreePop();
    }
  }

  unsigned int categoryCount = typeRefl->getCategoryCount();
  if(categoryCount > 0)
  {
    const std::string categoryNodeLabel = std::string("Categories (") + std::to_string(categoryCount) + ")";
    if(ImGui::TreeNode(categoryNodeLabel.c_str()))
    {
      for(unsigned int i = 0; i < categoryCount; i++)
      {
        const slang::ParameterCategory category  = typeRefl->getCategoryByIndex(i);
        const size_t                   size      = typeRefl->getSize(category);
        const size_t                   stride    = typeRefl->getStride(category);
        const size_t                   alignment = typeRefl->getAlignment(category);
        if(slang::TypeReflection::Kind::Array == kind)
        {
          const size_t elementStride = typeRefl->getElementStride(static_cast<SlangParameterCategory>(category));
          ImGui::Text("%s: size %zu, stride %zu, alignment %zu, element stride %zu", slangCategoryToString(category),
                      size, stride, alignment, elementStride);
        }
        else
        {
          ImGui::Text("%s: size %zu, stride %zu, alignment %zu", slangCategoryToString(category), size, stride, alignment);
        }
      }
      ImGui::TreePop();
    }
  }

  if(slang::TypeReflection::Kind::Struct == kind)
  {
    unsigned int      fieldCount      = typeRefl->getFieldCount();
    const std::string fieldsNodeLabel = std::string("Field layouts (") + std::to_string(fieldCount) + ")";
    if(ImGui::TreeNode(fieldsNodeLabel.c_str()))
    {
      for(unsigned int i = 0; i < fieldCount; i++)
      {
        slang::VariableLayoutReflection* field  = typeRefl->getFieldByIndex(i);
        const SlangInt                   offset = typeRefl->getFieldBindingRangeOffset(i);
        const std::string                fieldNodeLabel =
            std::to_string(i) + ": binding range offset " + std::to_string(offset) + ", name " + field->getName();
        if(ImGui::TreeNode(fieldNodeLabel.c_str()))
        {
          guiVariableLayoutReflection(field);
          ImGui::TreePop();
        }
      }
      ImGui::TreePop();
    }
  }

  const SlangInt bindingRangeCount = typeRefl->getBindingRangeCount();
  if(bindingRangeCount > 0)
  {
    const std::string bindingRangesLabel = std::string("Binding ranges (") + std::to_string(bindingRangeCount) + ")";
    if(ImGui::TreeNode(bindingRangesLabel.c_str()))
    {
      for(SlangInt i = 0; i < bindingRangeCount; i++)
      {
        const std::string nodeLabel = std::to_string(i);
        if(ImGui::TreeNode(nodeLabel.c_str()))
        {
          ImGui::Text("Binding type: %s", slangBindingTypeToString(typeRefl->getBindingRangeType(i)).c_str());
          ImGui::Text("Binding count: %" PRIu64, typeRefl->getBindingRangeBindingCount(i));
          ImGui::Text("Descriptor set index: %" PRIu64, typeRefl->getBindingRangeDescriptorSetIndex(i));
          ImGui::Text("Descriptor range count: %" PRIu64, typeRefl->getBindingRangeDescriptorRangeCount(i));
          ImGui::Text("First descriptor range index: %" PRIu64, typeRefl->getBindingRangeFirstDescriptorRangeIndex(i));
// FIXME: This crashes Slang
#if 0
          const SlangImageFormat imageFormat = typeRefl->getBindingRangeImageFormat(i);
          if(SLANG_IMAGE_FORMAT_unknown != imageFormat)
          {
            ImGui::Text("Image format: %s", slangImageFormatToString(imageFormat));
          }
#endif
          ImGui::Text("Is specializable: %s", typeRefl->isBindingRangeSpecializable(i) ? "true" : "false");
          slang::TypeLayoutReflection* leafTypeLayout = typeRefl->getBindingRangeLeafTypeLayout(i);
          if(leafTypeLayout)
          {
            const std::string leafTypeLabel = std::string("Leaf type layout: ") + guiGetName(leafTypeLayout);
            if(ImGui::TreeNode(leafTypeLabel.c_str()))
            {
              guiTypeLayoutReflection(leafTypeLayout);
              ImGui::TreePop();
            }
          }
          slang::VariableReflection* variableRefl = typeRefl->getBindingRangeLeafVariable(i);
          if(variableRefl)
          {
            const std::string varLabel = std::string("Variable: ") + guiGetName(variableRefl);
            if(ImGui::TreeNode(varLabel.c_str()))
            {
              guiVariableReflection(variableRefl);
              ImGui::TreePop();
            }
          }
          ImGui::TreePop();
        }
      }
      ImGui::TreePop();
    }
  }

  const SlangInt descriptorSetCount = typeRefl->getDescriptorSetCount();
  if(descriptorSetCount > 0)
  {
    const std::string descriptorSetsLabel = std::string("Descriptor sets (") + std::to_string(descriptorSetCount) + ")";
    if(ImGui::TreeNode(descriptorSetsLabel.c_str()))
    {
      for(SlangInt i = 0; i < descriptorSetCount; i++)
      {
        const SlangInt    spaceOffset = typeRefl->getDescriptorSetSpaceOffset(i);
        const SlangInt    rangeCount  = typeRefl->getDescriptorSetDescriptorRangeCount(i);
        const std::string nodeLabel =
            std::to_string(i) + ": space offset " + std::to_string(spaceOffset) + ", ranges: " + std::to_string(rangeCount);
        if(ImGui::TreeNode(nodeLabel.c_str()))
        {
          for(SlangInt range = 0; range < rangeCount; range++)
          {
            const slang::BindingType       bindingType = typeRefl->getDescriptorSetDescriptorRangeType(i, range);
            const slang::ParameterCategory category    = typeRefl->getDescriptorSetDescriptorRangeCategory(i, range);
            const SlangInt descriptorCount = typeRefl->getDescriptorSetDescriptorRangeDescriptorCount(i, range);
            const SlangInt indexOffset     = typeRefl->getDescriptorSetDescriptorRangeIndexOffset(i, range);
            ImGui::Text("%" PRId64 ": binding type `%s`, category `%s`, descriptor count %" PRId64 ", index offset %" PRId64,
                        range, slangBindingTypeToString(bindingType).c_str(), slangCategoryToString(category),
                        descriptorCount, indexOffset);
          }
          ImGui::TreePop();
        }
      }
      ImGui::TreePop();
    }
  }

  // For each sub-object range:
  // - binding range index
  // - space offset
  // - range offset
  const SlangInt subObjectRangeCount = typeRefl->getSubObjectRangeCount();
  if(subObjectRangeCount > 0)
  {
    const std::string subObjectRangesLabel =
        std::string("Sub-object ranges (") + std::to_string(subObjectRangeCount) + ")";
    if(ImGui::TreeNode(subObjectRangesLabel.c_str()))
    {
      for(SlangInt i = 0; i < subObjectRangeCount; i++)
      {
        const SlangInt                   bindingRangeIndex = typeRefl->getSubObjectRangeBindingRangeIndex(i);
        const SlangInt                   spaceOffset       = typeRefl->getSubObjectRangeSpaceOffset(i);
        slang::VariableLayoutReflection* rangeOffset       = typeRefl->getSubObjectRangeOffset(i);
        assert(rangeOffset);
        const std::string nodeLabel = std::to_string(i) + ": binding range index " + std::to_string(bindingRangeIndex)
                                      + ", space offset " + std::to_string(spaceOffset) + ", name " + guiGetName(rangeOffset);
        if(ImGui::TreeNode(nodeLabel.c_str()))
        {
          guiVariableLayoutReflection(rangeOffset);
          ImGui::TreePop();
        }
      }
      ImGui::TreePop();
    }
  }
}

void guiVariableReflection(slang::VariableReflection* varRefl)
{
  assert(varRefl);
  const bool hasDefaultValue = varRefl->hasDefaultValue();
  ImGui::Text("Has default value: %s", hasDefaultValue ? "true" : "false");
  if(hasDefaultValue)
  {
    int64_t     defaultValue = 0;
    SlangResult result       = varRefl->getDefaultValueInt(&defaultValue);
    if(SLANG_SUCCEEDED(result))
    {
      ImGui::Text("Default value (bytes): 0x%16" PRIx64, defaultValue);
    }
  }

  guiListModifiers(varRefl);

  slang::TypeReflection* typeRefl = varRefl->getType();
  if(typeRefl)
  {
    const std::string typeLabel = std::string("Type: ") + typeRefl->getName() + " [...]";
    if(ImGui::TreeNode(typeLabel.c_str()))
    {
      guiTypeReflection(typeRefl);
      ImGui::TreePop();
    }
  }

  guiListUserAttributes(varRefl);
}

void guiVariableLayoutReflection(slang::VariableLayoutReflection* varLayoutRefl)
{
  assert(varLayoutRefl);
  // Binding index - undocumented, no mention of use
  // Per category:
  //   Offset
  //   Binding space
  // Image format
  // Semantic name
  // Semantic index
  // Stage
  // Variable...
  // Type layout...

  unsigned int categoryCount = varLayoutRefl->getCategoryCount();
  if(categoryCount > 0)
  {
    const std::string nodeLabel = std::string("Categories (") + std::to_string(categoryCount) + ")";
    if(ImGui::TreeNode(nodeLabel.c_str()))
    {
      for(unsigned int i = 0; i < categoryCount; i++)
      {
        const slang::ParameterCategory category  = varLayoutRefl->getCategoryByIndex(i);
        const size_t                   relOffset = varLayoutRefl->getOffset(category);
        const size_t                   relSpace  = varLayoutRefl->getBindingSpace(category);
        ImGui::Text("%s: relative offset %zu, relative space %zu", slangCategoryToString(category), relOffset, relSpace);
      }
      ImGui::TreePop();
    }
  }

  const SlangImageFormat imageFormat = varLayoutRefl->getImageFormat();
  if(imageFormat != SLANG_IMAGE_FORMAT_unknown)
  {
    ImGui::Text("Image format: %s", slangImageFormatToString(imageFormat));
  }

  const char* semanticName = varLayoutRefl->getSemanticName();
  if(semanticName)
  {
    ImGui::Text("Semantic name: %s", semanticName);
    ImGui::Text("Semantic index: %zu", varLayoutRefl->getSemanticIndex());
  }

  const SlangStage stage = varLayoutRefl->getStage();
  if(stage != SLANG_STAGE_NONE)
  {
    ImGui::Text("Stage: %s", slangStageToString(stage));
  }

  slang::VariableReflection* varRefl = varLayoutRefl->getVariable();
  if(varRefl && ImGui::TreeNode("Variable reflection"))
  {
    guiVariableReflection(varRefl);
    ImGui::TreePop();
  }

  slang::TypeLayoutReflection* typeLayoutRefl = varLayoutRefl->getTypeLayout();
  if(typeLayoutRefl && ImGui::TreeNode("Type layout reflection"))
  {
    guiTypeLayoutReflection(typeLayoutRefl);
    ImGui::TreePop();
  }
}

void guiProgramReflection(slang::IComponentType* program)
{
  if(!program)
  {
    ImGui::TextWrapped("No program reflection info available.");
    return;
  }

  if(ImGui::TreeNodeEx("Layout", ImGuiTreeNodeFlags_DefaultOpen))
  {
    slang::ProgramLayout* programLayout = program->getLayout();

    const SlangUInt entrypointCount = programLayout->getEntryPointCount();
    ImGui::Text("Entrypoints (%" PRIu64 "):", entrypointCount);
    for(SlangUInt entryIdx = 0; entryIdx < entrypointCount; entryIdx++)
    {
      slang::EntryPointReflection* entrypoint = programLayout->getEntryPointByIndex(entryIdx);
      const std::string            nodeName   = std::to_string(entryIdx) + ": " + entrypoint->getName();
      if(ImGui::TreeNode(nodeName.c_str()))
      {
        guiEntryPointReflection(entrypoint);
        ImGui::TreePop();
      }
    }

    ImGui::Text("Global constant buffer binding: %" PRIu64, programLayout->getGlobalConstantBufferBinding());
    ImGui::Text("Global constant buffer size: %zu", programLayout->getGlobalConstantBufferSize());
    slang::VariableLayoutReflection* globalLayout = programLayout->getGlobalParamsVarLayout();
    if(ImGui::TreeNode("Global parameters"))
    {
      guiVariableLayoutReflection(globalLayout);
      ImGui::TreePop();
    }

    const unsigned    parameterCount = programLayout->getParameterCount();
    const std::string paramLabelName = std::string("Parameters (") + std::to_string(parameterCount) + ")";
    if(ImGui::TreeNode(paramLabelName.c_str()))
    {
      for(unsigned paramIdx = 0; paramIdx < parameterCount; paramIdx++)
      {
        slang::VariableLayoutReflection* param    = programLayout->getParameterByIndex(paramIdx);
        const std::string                nodeName = std::to_string(paramIdx) + ": " + param->getName();
        if(ImGui::TreeNode(nodeName.c_str()))
        {
          guiVariableLayoutReflection(param);
          ImGui::TreePop();
        }
      }
      ImGui::TreePop();
    }

#if 0
    const SlangUInt hashedStringCount = programLayout->getHashedStringCount();
    if(hashedStringCount > 0)
    {
      ImGui::Text("Hashed strings (%" PRIu64 "):", hashedStringCount);
      for(SlangUInt i = 0; i < hashedStringCount; i++)
      {
        ImGui::Text("%" PRIu64 ": %s", i, programLayout->getHashedString(i, nullptr));
      }
    }
#endif

    ImGui::TreePop();
  }
}

void guiModuleReflection(slang::IModule* module)
{
  slang::DeclReflection* moduleReflection = nullptr;
  if(!module || !(moduleReflection = module->getModuleReflection()))
  {
    ImGui::TextWrapped("No module reflection info available.");
    return;
  }

  if(ImGui::TreeNodeEx("Module", ImGuiTreeNodeFlags_DefaultOpen))
  {
    guiDeclReflection(moduleReflection);
    ImGui::TreePop();
  }
}
