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

#include "bob.h"

#include <stdio.h>
#include <sys/param.h>

#include "cognon.h"

#define ABS(a) ((a) < 0.0 ? -(a) : (a))
#define FLOAT_EQ(a, b) \
  (ABS((a) - (b)) / (MIN(ABS(a), ABS(b)) + 1.0e-6) < 1.0e-6)
#define EXPECT_FEQ(a, b) EXPECT_TRUE(FLOAT_EQ(a, b))

namespace cognon {

class BobTest : public testing::Test {
};

class TestBob {
 public:
  TestBob() { }
  ~TestBob() { }

  void CheckBitsPerNeuron() {
    Bob bob;

    for (int32 true_true = 0; true_true <= 100; true_true += 10) {
      for (int32 false_true = 0; false_true <= 100; false_true += 10) {
        printf("BitsPerNeuron(%d, %d, %d, %d, %d) = %f\n",
               100, true_true, 100 - true_true,
               false_true, 100 - false_true,
               bob.BitsPerNeuron(100, true_true, 100 - true_true,
                                 false_true, 100 - false_true));
      }
    }
  }
};

TEST_F(BobTest, CheckBitsPerNeuron) {
  TestBob test;

  test.CheckBitsPerNeuron();
}

TEST_F(BobTest, CheckMutualInformation) {
}

}  // namespace

int main(int argc, char **argv) {
  CALL_TEST(cognon::CheckBitsPerNeuron);
  //  CALL_TEST(cognon::CheckMututalInformation);
  return 0;
}
