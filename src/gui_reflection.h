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

namespace slang {
struct IComponentType;
struct IModule;
}

// Runs ImGui commands for the Reflection pane.
// Displaying all reflection info in a human-readable way requires a lot of
// formatting code (toJson() is easier to use if you need something simpler),
// so it's separated into its own file for cleanliness.
// `program` may be null.
void guiProgramReflection(slang::IComponentType* program);
void guiModuleReflection(slang::IModule* module);
