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

// General C++ utilities.

#include <filesystem>
#include <initializer_list>
#include <string_view>

// Implementation of std::hash<std::filesystem::path>, since Visual Studio 2019's
// standard library doesn't include it;
// see https://github.com/microsoft/STL/issues/2556 .
// Note that implementations usually hash something like the string value of
// the path, so we're only guaranteed to get identical hashes if the string
// values are the same.
// This is equivalent to std::hash<std::filesystem::path> in VS 2022.
struct PathHash
{
  std::size_t operator()(const std::filesystem::path& p) const { return std::filesystem::hash_value(p); }
};

// Returns true iff the given element is contained in the given set,
// using the given predicate to test equality.
template <class Item, class Iterable, class BinaryPredicate>
static bool isIn(const Item& item, const Iterable& container, BinaryPredicate predicate)
{
  for(auto it = container.begin(); it != container.end(); ++it)
  {
    if(predicate(item, *it))
    {
      return true;
    }
  }
  return false;
}

template <class Item, class Iterable>
static bool isIn(const Item& item, const Iterable& container)
{
  for(auto it = container.begin(); it != container.end(); ++it)
  {
    if(item == *it)
    {
      return true;
    }
  }
  return false;
}

// Compares two strings, case-insensitive.
// This has the same return values as strcmp; negative if str1 appears before
// str1 in lexicographic order; zero if they're equal; and positive if str1
// appears after.
int stricompare(const std::string_view str1, const std::string_view str2);

// Tests for case-insensitive ASCII string equality. We use this to detect if
// a variable name matches one of our recognized variable names.
bool strieq(const std::string_view str1, const std::string_view str2);

// Same as strieq, except it tests against multiple names and returns
// true if any of them match.
bool strieqList(const std::string_view str1, const std::initializer_list<const std::string_view> patterns);

// Same as string_view::starts_with, but case-insensitive.
bool startsWithI(const std::string_view text, const std::string_view prefix);

// Implements tolower; doesn't look at the locale.
// This allows a compiler to see its implementation and possibly inline it.
// Otherwise, experimentally, it becomes a function call.
// See also https://gist.github.com/easyaspi314/9d31e5c0f9cead66aba2ede248b74d64
inline char myToLower(char c)
{
  if(c >= 'A' && c <= 'Z')
  {
    c += 'a' - 'A';
  }
  return c;
}

// Returns a string corresponding to the current UTC time (down to the precision
// of std:chrono::system_clock), with some characters replaced with dashes so
// it can be used in a file path.
std::string pathSafeTimeString();
