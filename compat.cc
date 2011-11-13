// Copyright 2009-2011 Carl Staelin. All Rights Reserved.
// Copyright 2011 Google Inc. All Rights Reserved.
//
// Author: carl.staelin@gmail.com (Carl Staelin)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include "compat.h"

#include <omp.h>

MTRand** RandomBase::random = NULL;
RandomBase::RandomBase() : threadId(omp_get_thread_num()) {
  if (random == NULL) {
    random = new MTRand*[omp_get_max_threads()];
    for (int32 i = 0; i < omp_get_max_threads(); ++i) {
      random[i] = new MTRand;
    }
  }
}

uint32 RandomBase::Rand32() { return random[threadId]->randInt(); }

uint64 RandomBase::Rand64() {
  return ((static_cast<uint64>(random[threadId]->randInt()) << 32)
          | static_cast<uint64>(random[threadId]->randInt()));
}

nullstream cnull;
bool test_fail = false;

string StringPrintf(const char* format, ...) {
 va_list args;
 va_start(args, format);
 string result = vStringPrintf(format, args);
 va_end(args);
 return result;
}

string vStringPrintf(const char* format, va_list args) {
 const int bufferSize = 16384;
 char buffer[bufferSize];
 vsnprintf(buffer, bufferSize, format, args);
 return string(buffer);
}

namespace cognon {

RandomBase* CreateRandom() { return new RandomBase; }

Job::Job() { }
Job::~Job() { }

void RunParallel(vector<Job*>* jobs) {
  int32 N = jobs->size();
  RandomBase r;  // This is to ensure that we have initialized it
  const bool run_in_parallel = true;

  if (run_in_parallel) {
#pragma omp parallel for
    for (int32 i = 0; i < N; ++i) {
      (*jobs)[i]->Run();
    }
  } else {
    for (int32 i = 0; i < N; ++i) {
      (*jobs)[i]->Run();
    }
  }

  for (int32 i = 0; i < N; ++i) {
    delete (*jobs)[i];
  }
  jobs->clear();
}

}  // namespace
