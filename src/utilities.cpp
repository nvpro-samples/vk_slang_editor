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

#include "utilities.h"

#include <nvutils/logger.hpp>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <chrono>
#include <iomanip>
#include <ranges>
#include <sstream>

int stricompare(const std::string_view str1, const std::string_view str2)
{
  const size_t common = std::min(str1.size(), str2.size());
  for(size_t i = 0; i < common; i++)
  {
    const int c1 = static_cast<int>(myToLower(str1[i]));
    const int c2 = static_cast<int>(myToLower(str2[i]));
    if(c1 != c2)
    {
      return c1 - c2;
    }
  }
  // Given identical common prefixes, shorter strings appear first in the
  // dictionary: 'A' appears before 'aardvark'
  if(str1.size() < str2.size())
  {
    return -1;
  }
  else if(str1.size() > str2.size())
  {
    return 1;
  }
  return 0;
}

bool strieq(const std::string_view str1, const std::string_view str2)
{
  return std::ranges::equal(str1, str2, [](auto c1, auto c2) { return myToLower(c1) == myToLower(c2); });
}

// Same as strieq, except it tests against multiple names and returns
// true if any of them match.
bool strieqList(const std::string_view str1, const std::initializer_list<const std::string_view> patterns)
{
  return isIn(str1, patterns, strieq);
}

// Same as string_view::starts_with, but case-insensitive.
bool startsWithI(const std::string_view text, const std::string_view prefix)
{
  if(prefix.size() > text.size())
    return false;
  return std::ranges::equal(prefix, text.substr(0, prefix.size()),
                            [](auto c1, auto c2) { return myToLower(c1) == myToLower(c2); });
}

std::string pathSafeTimeString()
{
  // Thank you to https://stackoverflow.com/a/58523115 ;
  // operator << (ostream&) doesn't appear to be available on GCC 11.
  const auto   time  = std::chrono::system_clock::now();
  const time_t timeT = std::chrono::system_clock::to_time_t(time);
#ifdef _MSC_VER
  tm            gmt{};
  const errno_t err = gmtime_s(&gmt, &timeT);
  if(0 != err)
  {
    LOGE("gmtime_s failed with errno %u! This should never happen.\n", static_cast<unsigned>(err));
    return "0000-00-00T00-00-00Z";
  }
#else
  tm gmt = *std::gmtime(&timeT);
#endif
  std::stringstream s;
  s << std::put_time(&gmt, "%FT%H-%M-%SZ");
  return s.str();
}
