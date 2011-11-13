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
// Tests Statistic, Histogram, and RunExperiment.
//

#include "cognon.h"

#include <math.h>
#include <sys/param.h>

#define ABS(a) ((a) < 0.0 ? -(a) : (a))
#define FLOAT_EQ(a, b) \
  (ABS((a) - (b)) / (MIN(ABS(a), ABS(b)) + 1.0e-6) < 1.0e-6)
#define EXPECT_FEQ(a, b) EXPECT_TRUE(FLOAT_EQ(a, b))

namespace cognon {

class CognonTest : public testing::Test {
};

TEST_F(CognonTest, CheckStatistic) {
  Statistic a;
  Statistic b;
  Statistic c;

  // Testing adding a single sample to empty Statistic a {}.
  // Expect Statistic a {3.0}.
  AddSample(3.0, &a);
  EXPECT_EQ(1, a.count())
      << "AddSample(3.0, &a): Expected count of 1: " << a.count();
  EXPECT_FEQ(3.0, Mean(a))
      << "AddSample(3.0, &a): Expected mean of 3.0: " << Mean(a);
  EXPECT_FEQ(-1.0, Stddev(a))
      << "AddSample(3.0, &a): Expected no stddev: " << Stddev(a);
  EXPECT_FEQ(3.0, a.values(0))
      << "AddSample(3.0, &a): Expected a.values(0) 3.0: " << a.values(0);

  // Add a second sample to Statistic a {3.0}.
  // Expect Statistic a {3.0, 4.0}.
  AddSample(4.0, &a);
  EXPECT_EQ(2, a.count())
      << "AddSample(4.0, &a): Expected count of 2: " << a.count();
  EXPECT_FEQ(3.5, Mean(a))
      << "AddSample(4.0, &a): Expected mean of 3.5: " << Mean(a);
  EXPECT_FEQ(sqrt(1.0/2.0), Stddev(a))
      << "AddSample(4.0, &a): Expected stddev of "
      << sqrt(1.0 / 2.0) << ": " << Stddev(a);
  EXPECT_FEQ(3.0, a.values(0))
      << "AddSample(4.0, &a): Expected a.values(0) 3.0: " << a.values(0);
  EXPECT_FEQ(4.0, a.values(1))
      << "AddSample(4.0, &a): Expected a.values(1) 4.0: " << a.values(1);

  // Add a third sample to Statistic a {3.0, 4.0}.
  // Expect Statistic a {3.0, 4.0, 5.0}.
  AddSample(5.0, &a);
  EXPECT_EQ(3, a.count())
      << "AddSample(5.0, &a): Expected count of 3: " << a.count();
  EXPECT_FEQ(4.0, Mean(a))
      << "AddSample(5.0, &a): Expected mean of 4.0: " << Mean(a);
  EXPECT_FEQ(1.0, Stddev(a))
      << "AddSample(5.0, &a): Expected stddev of 1.0: " << Stddev(a);
  EXPECT_FEQ(3.0, a.values(0))
      << "AddSample(5.0, &a): Expected a.values(0) 3.0: " << a.values(0);
  EXPECT_FEQ(4.0, a.values(1))
      << "AddSample(5.0, &a): Expected a.values(1) 4.0: " << a.values(1);
  EXPECT_FEQ(5.0, a.values(2))
      << "AddSample(5.0, &a): Expected a.values(2) 5.0: " << a.values(2);

  // Try "+=" of empty Statistic c {} with Statistic a {3.0, 4.0, 5.0}.
  // Expect Statistic c {3.0, 4.0, 5.0}.
  c += a;
  EXPECT_EQ(3, c.count())
      << "c += a: Expected count of 3: " << c.count();
  EXPECT_FEQ(4.0, Mean(c))
      << "c += a: Expected mean of 4.0: " << Mean(c);
  EXPECT_FEQ(1.0, Stddev(c))
      << "c += a: Expected stddev of 1.0: " << Stddev(c);
  EXPECT_FEQ(3.0, c.values(0))
      << "c += a: Expected c.values(0) 3.0: " << c.values(0);
  EXPECT_FEQ(4.0, c.values(1))
      << "c += a: Expected c.values(1) 4.0: " << c.values(1);
  EXPECT_FEQ(5.0, c.values(2))
      << "c += a: Expected c.values(2) 5.0: " << c.values(2);

  // Try "+" of Statistic a {3.0, 4.0, 5.0} with empty Statistic b {}.
  // Expect Statistic c {3.0, 4.0, 5.0}.
  c = a + b;
  EXPECT_EQ(3, c.count())
      << "c = a + b: Expected count of 3: " << c.count();
  EXPECT_FEQ(4.0, Mean(c))
      << "c = a + b: Expected mean of 4.0: " << Mean(c);
  EXPECT_FEQ(1.0, Stddev(c))
      << "c = a + b: Expected stddev of 1.0: " << Stddev(c);
  EXPECT_FEQ(3.0, c.values(0))
      << "c = a + b: Expected c.values(0) 3.0: " << c.values(0);
  EXPECT_FEQ(4.0, c.values(1))
      << "c = a + b: Expected c.values(1) 4.0: " << c.values(1);
  EXPECT_FEQ(5.0, c.values(2))
      << "c = a + b: Expected c.values(2) 5.0: " << c.values(2);

  // Try "+" of empty Statistic b {} with Statistic a {3.0, 4.0, 5.0}.
  // Expect Statistic c {3.0, 4.0, 5.0}.
  c = b + a;
  EXPECT_EQ(3, c.count())
      << "c = b + a: Expected count of 3: " << c.count();
  EXPECT_FEQ(4.0, Mean(c))
      << "c = b + a: Expected mean of 4.0: " << Mean(c);
  EXPECT_FEQ(1.0, Stddev(c))
      << "c = b + a: Expected stddev of 1.0: " << Stddev(c);
  EXPECT_FEQ(3.0, c.values(0))
      << "c = b + a: Expected c.values(0) 3.0: " << c.values(0);
  EXPECT_FEQ(4.0, c.values(1))
      << "c = b + a: Expected c.values(1) 4.0: " << c.values(1);
  EXPECT_FEQ(5.0, c.values(2))
      << "c = b + a: Expected c.values(2) 5.0: " << c.values(2);

  // Try "+=" of Statistic a {3.0, 4.0, 5.0} with empty Statistic b {}.
  // Expect Statistic a {3.0, 4.0, 5.0}.
  a += b;
  EXPECT_EQ(3, a.count())
      << "c += b: Expected count of 3: " << a.count();
  EXPECT_FEQ(4.0, Mean(a))
      << "c += b: Expected mean of 4.0: " << Mean(a);
  EXPECT_FEQ(1.0, Stddev(a))
      << "c += b: Expected stddev of 1.0: " << Stddev(a);
  EXPECT_FEQ(3.0, a.values(0))
      << "c += b: Expected c.values(0) 3.0: " << a.values(0);
  EXPECT_FEQ(4.0, a.values(1))
      << "c += b: Expected c.values(1) 4.0: " << a.values(1);
  EXPECT_FEQ(5.0, a.values(2))
      << "c += b: Expected c.values(2) 5.0: " << a.values(2);

  // Add a three samples to empty Statistic b.
  // Expect Statistic b {5.0, 6.0, 7.0}.
  AddSample(5.0, &b);
  AddSample(6.0, &b);
  AddSample(7.0, &b);
  EXPECT_EQ(3, a.count())
      << "b: Expected count of 3: " << b.count();
  EXPECT_FEQ(6.0, Mean(b))
      << "b: Expected mean of 6.0: " << Mean(b);
  EXPECT_FEQ(1.0, Stddev(a))
      << "b: Expected stddev of 1.0: " << Stddev(b);
  EXPECT_FEQ(5.0, b.values(0))
      << "b: Expected b.values(0) 5.0: " << b.values(0);
  EXPECT_FEQ(6.0, b.values(1))
      << "b: Expected b.values(1) 6.0: " << b.values(1);
  EXPECT_FEQ(7.0, b.values(2))
      << "b: Expected b.values(2) 7.0: " << b.values(2);

  // Try "+" of Statistic a {3.0, 4.0, 5.0} with Statistic b {5.0, 6.0, 7.0}.
  // Expect Statistic c {3.0, 4.0, 5.0, 5.0, 6.0, 7.0}.
  c = a + b;
  EXPECT_EQ(6, c.count())
      << "c = a + b: Expected count of 6: " << c.count();
  EXPECT_FEQ(5.0, Mean(c))
      << "c = a + b: Expected mean of 5.0: " << Mean(c);
  EXPECT_FEQ(1.4142135623, Stddev(c))
      << "c = a + b: Expected stddev of 1.4142135623: " << Stddev(c);
  EXPECT_FEQ(3.0, c.values(0))
      << "c = a + b: Expected c.values(0) 3.0: " << c.values(0);
  EXPECT_FEQ(4.0, c.values(1))
      << "c = a + b: Expected c.values(1) 4.0: " << c.values(1);
  EXPECT_FEQ(5.0, c.values(2))
      << "c = a + b: Expected c.values(2) 5.0: " << c.values(2);
  EXPECT_FEQ(5.0, c.values(3))
      << "c = a + b: Expected c.values(3) 5.0: " << c.values(3);
  EXPECT_FEQ(6.0, c.values(4))
      << "c = a + b: Expected c.values(4) 6.0: " << c.values(4);
  EXPECT_FEQ(7.0, c.values(5))
      << "c = a + b: Expected c.values(5) 7.0: " << c.values(5);

  // Try "+" of Statistic b {5.0, 6.0, 7.0} with Statistic a {3.0, 4.0, 5.0}.
  // Expect Statistic c {5.0, 6.0, 7.0, 3.0, 4.0, 5.0}.
  c = b + a;
  EXPECT_EQ(6, c.count())
      << "c = b + a: Expected count of 6: " << c.count();
  EXPECT_FEQ(5.0, Mean(c))
      << "c = b + a: Expected mean of 4.0: " << Mean(c);
  EXPECT_FEQ(1.4142135623, Stddev(c))
      << "c = a + b: Expected stddev of 1.4142135623: " << Stddev(c);
  EXPECT_FEQ(5.0, c.values(0))
      << "c = b + a: Expected c.values(0) 5.0: " << c.values(0);
  EXPECT_FEQ(6.0, c.values(1))
      << "c = b + a: Expected c.values(1) 6.0: " << c.values(1);
  EXPECT_FEQ(7.0, c.values(2))
      << "c = b + a: Expected c.values(2) 7.0: " << c.values(2);
  EXPECT_FEQ(3.0, c.values(3))
      << "c = b + a: Expected c.values(3) 3.0: " << c.values(3);
  EXPECT_FEQ(4.0, c.values(4))
      << "c = b + a: Expected c.values(4) 4.0: " << c.values(4);
  EXPECT_FEQ(5.0, c.values(5))
      << "c = b + a: Expected c.values(5) 5.0: " << c.values(5);

  // Try "+" of Statistic c {3.0, 4.0, 5.0} with Statistic b {5.0, 6.0, 7.0}.
  // Expect Statistic c {3.0, 4.0, 5.0, 5.0, 6.0, 7.0}.
  c.CopyFrom(a);
  c += b;
  EXPECT_EQ(6, c.count())
      << "c += b: Expected count of 6: " << c.count();
  EXPECT_FEQ(5.0, Mean(c))
      << "c += b: Expected mean of 5.0: " << Mean(c);
  EXPECT_FEQ(1.4142135623, Stddev(c))
      << "c += b: Expected stddev of 1.4142135623: " << Stddev(c);
  EXPECT_FEQ(3.0, c.values(0))
      << "c += b: Expected c.values(0) 3.0: " << c.values(0);
  EXPECT_FEQ(4.0, c.values(1))
      << "c += b: Expected c.values(1) 4.0: " << c.values(1);
  EXPECT_FEQ(5.0, c.values(2))
      << "c += b: Expected c.values(2) 5.0: " << c.values(2);
  EXPECT_FEQ(5.0, c.values(3))
      << "c += b: Expected c.values(3) 5.0: " << c.values(3);
  EXPECT_FEQ(6.0, c.values(4))
      << "c += b: Expected c.values(4) 6.0: " << c.values(4);
  EXPECT_FEQ(7.0, c.values(5))
      << "c += b: Expected c.values(5) 7.0: " << c.values(5);

  // Add sample 8.0 to Statistic b {5.0, 6.0, 7.0}.
  // Expect Statistic b {5.0, 6.0, 7.0, 8.0}.
  AddSample(8.0, &b);
  EXPECT_EQ(4, b.count())
      << "b: Expected count of 4: " << b.count();
  EXPECT_FEQ(6.5, Mean(b))
      << "b: Expected mean of 6.5: " << Mean(b);
  EXPECT_FEQ(1.2909944487, Stddev(b))
      << "b: Expected stddev of 1.2909944487: " << Stddev(b);
  EXPECT_FEQ(5.0, b.values(0))
      << "b: Expected b.values(0) 5.0: " << b.values(0);
  EXPECT_FEQ(6.0, b.values(1))
      << "b: Expected b.values(1) 6.0: " << b.values(1);
  EXPECT_FEQ(7.0, b.values(2))
      << "b: Expected b.values(2) 7.0: " << b.values(2);
  EXPECT_FEQ(8.0, b.values(3))
      << "b: Expected b.values(3) 8.0: " << b.values(3);

  // Try "+" of Statistic a {3.0, 4.0, 5.0}
  //       with Statistic b {5.0, 6.0, 7.0, 8.0}.
  // Expect Statistic c {3.0, 4.0, 5.0, 5.0, 6.0, 7.0, 8.0}.
  c = a + b;
  EXPECT_EQ(7, c.count())
      << "c = a + b: Expected count of 7: " << c.count();
  EXPECT_FEQ(5.42857142857, Mean(c))
      << "c = a + b: Expected mean of 5.42857142857: " << Mean(c);
  EXPECT_FEQ(1.71824938596, Stddev(c))
      << "c = a + b: Expected stddev of 1.71824938596: " << Stddev(c);
  EXPECT_FEQ(3.0, c.values(0))
      << "c = a + b: Expected c.values(0) 3.0: " << c.values(0);
  EXPECT_FEQ(4.0, c.values(1))
      << "c = a + b: Expected c.values(1) 4.0: " << c.values(1);
  EXPECT_FEQ(5.0, c.values(2))
      << "c = a + b: Expected c.values(2) 5.0: " << c.values(2);
  EXPECT_FEQ(5.0, c.values(3))
      << "c = a + b: Expected c.values(3) 5.0: " << c.values(3);
  EXPECT_FEQ(6.0, c.values(4))
      << "c = a + b: Expected c.values(4) 6.0: " << c.values(4);
  EXPECT_FEQ(7.0, c.values(5))
      << "c = a + b: Expected c.values(5) 7.0: " << c.values(5);
  EXPECT_FEQ(8.0, c.values(6))
      << "c = a + b: Expected c.values(6) 8.0: " << c.values(6);

  // Try "+" of Statistic b {5.0, 6.0, 7.0, 8.0}
  //       with Statistic a {3.0, 4.0, 5.0}.
  // Expect Statistic c {5.0, 6.0, 7.0, 8.0, 3.0, 4.0, 5.0}.
  c = b + a;
  EXPECT_EQ(7, c.count())
      << "c = b + a: Expected count of 7: " << c.count();
  EXPECT_FEQ(5.42857142857, Mean(c))
      << "c = b + a: Expected mean of 5.42857142857: " << Mean(c);
  EXPECT_FEQ(1.71824938596, Stddev(c))
      << "c = b + a: Expected stddev of 1.71824938596: " << Stddev(c);
  EXPECT_FEQ(5.0, c.values(0))
      << "c = b + a: Expected c.values(0) 5.0: " << c.values(0);
  EXPECT_FEQ(6.0, c.values(1))
      << "c = b + a: Expected c.values(1) 6.0: " << c.values(1);
  EXPECT_FEQ(7.0, c.values(2))
      << "c = b + a: Expected c.values(2) 7.0: " << c.values(2);
  EXPECT_FEQ(8.0, c.values(3))
      << "c = b + a: Expected c.values(3) 8.0: " << c.values(3);
  EXPECT_FEQ(3.0, c.values(4))
      << "c = b + a: Expected c.values(4) 3.0: " << c.values(4);
  EXPECT_FEQ(4.0, c.values(5))
      << "c = b + a: Expected c.values(5) 4.0: " << c.values(5);
  EXPECT_FEQ(5.0, c.values(6))
      << "c = a + b: Expected c.values(6) 5.0: " << c.values(6);
}

TEST_F(CognonTest, CheckHistogram) {
  Histogram a;
  Histogram b;
  Histogram c;
  vector<int32> v;

  // Add histogram vector {0, 3, 5, 1} to empty Histogram a.
  // Expect Histogram a {{0}, {3}, {5}, {1}}.
  v.resize(4);
  v[1] = 3;
  v[2] = 5;
  v[3] = 1;
  SetHistogram(v, &a);
  EXPECT_EQ(4, a.values_size())
      << "a: Expected a.values_size() 4: " << a.values_size();
  EXPECT_FEQ(0.0, Mean(a.values(0)))
      << "a: Expected a.values(0) 0.0: " << Mean(a.values(0));
  EXPECT_FEQ(3.0, Mean(a.values(1)))
      << "a: Expected a.values(1) 3.0: " << Mean(a.values(1));
  EXPECT_FEQ(5.0, Mean(a.values(2)))
      << "a: Expected a.values(2) 5.0: " << Mean(a.values(2));
  EXPECT_FEQ(1.0, Mean(a.values(3)))
      << "a: Expected a.values(3) 1.0: " << Mean(a.values(3));
  EXPECT_EQ(1, a.values(0).count())
      << "a: Expected a.values(0).count() 1: " << a.values(0).count();
  EXPECT_EQ(1, a.values(1).count())
      << "a: Expected a.values(1).count() 1: " << a.values(1).count();
  EXPECT_EQ(1, a.values(2).count())
      << "a: Expected a.values(2).count() 1: " << a.values(2).count();
  EXPECT_EQ(1, a.values(3).count())
      << "a: Expected a.values(3).count() 1: " << a.values(3).count();

  // Try "+" of Histogram a {{0}, {3}, {5}, {1}} with empty Histogram b {}.
  // Expect Histogram c {{0}, {3}, {5}, {1}}.
  c = a + b;
  EXPECT_EQ(4, c.values_size())
      << "c = a + b: Expected c.values_size() 4: " << c.values_size();
  EXPECT_FEQ(0.0, Mean(c.values(0)))
      << "c = a + b: Expected c.values(0) 0.0: " << Mean(c.values(0));
  EXPECT_FEQ(3.0, Mean(c.values(1)))
      << "c = a + b: Expected c.values(1) 3.0: " << Mean(c.values(1));
  EXPECT_FEQ(5.0, Mean(c.values(2)))
      << "c = a + b: Expected c.values(2) 5.0: " << Mean(c.values(2));
  EXPECT_FEQ(1.0, Mean(c.values(3)))
      << "c = a + b: Expected c.values(3) 1.0: " << Mean(c.values(3));
  EXPECT_EQ(1, c.values(0).count())
      << "c = a + b: Expected c.values(0).count() 1: " << c.values(0).count();
  EXPECT_EQ(1, c.values(1).count())
      << "c = a + b: Expected c.values(1).count() 1: " << c.values(1).count();
  EXPECT_EQ(1, c.values(2).count())
      << "c = a + b: Expected c.values(2).count() 1: " << c.values(2).count();
  EXPECT_EQ(1, c.values(3).count())
      << "c = a + b: Expected c.values(3).count() 1: " << c.values(3).count();

  // Try "+" of empty Histogram b {} with Histogram a {{0}, {3}, {5}, {1}}.
  // Expect Histogram c {{0}, {3}, {5}, {1}}.
  c = b + a;
  EXPECT_EQ(4, c.values_size())
      << "c = b + a: Expected c.values_size() 4: " << c.values_size();
  EXPECT_FEQ(0.0, Mean(c.values(0)))
      << "c = b + a: Expected c.values(0) 0.0: " << Mean(c.values(0));
  EXPECT_FEQ(3.0, Mean(c.values(1)))
      << "c = b + a: Expected c.values(1) 3.0: " << Mean(c.values(1));
  EXPECT_FEQ(5.0, Mean(c.values(2)))
      << "c = b + a: Expected c.values(2) 5.0: " << Mean(c.values(2));
  EXPECT_FEQ(1.0, Mean(c.values(3)))
      << "c = b + a: Expected c.values(3) 1.0: " << Mean(c.values(3));
  EXPECT_EQ(1, c.values(0).count())
      << "c = b + a: Expected c.values(0).count() 1: " << c.values(0).count();
  EXPECT_EQ(1, c.values(1).count())
      << "c = b + a: Expected c.values(1).count() 1: " << c.values(1).count();
  EXPECT_EQ(1, c.values(2).count())
      << "c = b + a: Expected c.values(2).count() 1: " << c.values(2).count();
  EXPECT_EQ(1, c.values(3).count())
      << "c = b + a: Expected c.values(3).count() 1: " << c.values(3).count();

  // Try "+=" of Histogram a {{0}, {3}, {5}, {1}} with empty Histogram b {}.
  // Expect Histogram a {{0}, {3}, {5}, {1}}.
  a += b;
  EXPECT_EQ(4, a.values_size())
      << "a += b Expected a.values_size() 4: " << a.values_size();
  EXPECT_FEQ(0.0, Mean(a.values(0)))
      << "a += b Expected a.values(0) 0.0: " << Mean(a.values(0));
  EXPECT_FEQ(3.0, Mean(a.values(1)))
      << "a += b Expected a.values(1) 3.0: " << Mean(a.values(1));
  EXPECT_FEQ(5.0, Mean(a.values(2)))
      << "a += b Expected a.values(2) 5.0: " << Mean(a.values(2));
  EXPECT_FEQ(1.0, Mean(a.values(3)))
      << "a += b Expected a.values(3) 1.0: " << Mean(a.values(3));
  EXPECT_EQ(1, a.values(0).count())
      << "a += b Expected a.values(0).count() 1: " << a.values(0).count();
  EXPECT_EQ(1, a.values(1).count())
      << "a += b Expected a.values(1).count() 1: " << a.values(1).count();
  EXPECT_EQ(1, a.values(2).count())
      << "a += b Expected a.values(2).count() 1: " << a.values(2).count();
  EXPECT_EQ(1, a.values(3).count())
      << "a += b Expected a.values(3).count() 1: " << a.values(3).count();

  // Add histogram vector {1, 4} to empty Histogram b.
  // Expect Histogram b {{1}, {4}}.
  v.resize(2);
  v[0] = 1;
  v[1] = 4;
  SetHistogram(v, &b);
  EXPECT_EQ(2, b.values_size())
      << "a: Expected b.values_size() 2: " << b.values_size();
  EXPECT_FEQ(1.0, Mean(b.values(0)))
      << "a: Expected b.values(0) 1.0: " << Mean(b.values(0));
  EXPECT_FEQ(4.0, Mean(b.values(1)))
      << "a: Expected b.values(1) 4.0: " << Mean(b.values(1));
  EXPECT_EQ(1, b.values(0).count())
      << "a: Expected b.values(0).count() 1: " << b.values(0).count();
  EXPECT_EQ(1, b.values(1).count())
      << "a: Expected b.values(1).count() 1: " << b.values(1).count();

  // Try "+" of Histogram a {{0}, {3}, {5}, {1}}
  //       with Histogram b {{1}, {4}}.
  // Expect Histogram c {{0, 1}, {3, 4}, {5, 0}, {1, 0}}.
  c = a + b;
  EXPECT_EQ(4, c.values_size())
      << "c = a + b: Expected c.values_size() 4: " << c.values_size();
  EXPECT_FEQ(0.5, Mean(c.values(0)))
      << "c = a + b: Expected c.values(0) 0.5: " << Mean(c.values(0));
  EXPECT_FEQ(3.5, Mean(c.values(1)))
      << "c = a + b: Expected c.values(1) 3.5: " << Mean(c.values(1));
  EXPECT_FEQ(2.5, Mean(c.values(2)))
      << "c = a + b: Expected c.values(2) 2.5: " << Mean(c.values(2));
  EXPECT_FEQ(0.5, Mean(c.values(3)))
      << "c = a + b: Expected c.values(3) 0.5: " << Mean(c.values(3));
  EXPECT_EQ(2, c.values(0).count())
      << "c = a + b: Expected c.values(0).count() 2: " << c.values(0).count();
  EXPECT_EQ(2, c.values(1).count())
      << "c = a + b: Expected c.values(1).count() 2: " << c.values(1).count();
  EXPECT_EQ(2, c.values(2).count())
      << "c = a + b: Expected c.values(2).count() 2: " << c.values(2).count();
  EXPECT_EQ(2, c.values(3).count())
      << "c = a + b: Expected c.values(3).count() 2: " << c.values(3).count();

  // Try "+" of Histogram b {{1}, {4}}
  //       with Histogram a {{0}, {3}, {5}, {1}}.
  // Expect Histogram c {{1, 0}, {4, 3}, {0, 5}, {0, 1}}.
  c = b + a;
  EXPECT_EQ(4, c.values_size())
      << "c = b + a: Expected c.values_size() 4: " << c.values_size();
  EXPECT_FEQ(0.5, Mean(c.values(0)))
      << "c = b + a: Expected c.values(0) 0.5: " << Mean(c.values(0));
  EXPECT_FEQ(3.5, Mean(c.values(1)))
      << "c = b + a: Expected c.values(1) 3.5: " << Mean(c.values(1));
  EXPECT_FEQ(2.5, Mean(c.values(2)))
      << "c = b + a: Expected c.values(2) 2.5: " << Mean(c.values(2));
  EXPECT_FEQ(0.5, Mean(c.values(3)))
      << "c = b + a: Expected c.values(3) 0.5: " << Mean(c.values(3));
  EXPECT_EQ(2, c.values(0).count())
      << "c = b + a: Expected c.values(0).count() 2: " << c.values(0).count();
  EXPECT_EQ(2, c.values(1).count())
      << "c = b + a: Expected c.values(1).count() 2: " << c.values(1).count();
  EXPECT_EQ(2, c.values(2).count())
      << "c = b + a: Expected c.values(2).count() 2: " << c.values(2).count();
  EXPECT_EQ(2, c.values(3).count())
      << "c = b + a: Expected c.values(3).count() 2: " << c.values(3).count();

  // Try "+=" of Histogram c {{0}, {3}, {5}, {1}}
  //        with Histogram b {{1}, {4}}.
  // Expect Histogram c {{0, 1}, {3, 4}, {5, 0}, {1, 0}}.
  c.CopyFrom(a);
  c += b;
  EXPECT_EQ(4, c.values_size())
      << "c += b: Expected c.values_size() 4: " << c.values_size();
  EXPECT_FEQ(0.5, Mean(c.values(0)))
      << "c += b: Expected c.values(0) 0.5: " << Mean(c.values(0));
  EXPECT_FEQ(3.5, Mean(c.values(1)))
      << "c += b: Expected c.values(1) 3.5: " << Mean(c.values(1));
  EXPECT_FEQ(2.5, Mean(c.values(2)))
      << "c += b: Expected c.values(2) 2.5: " << Mean(c.values(2));
  EXPECT_FEQ(0.5, Mean(c.values(3)))
      << "c += b: Expected c.values(3) 0.5: " << Mean(c.values(3));
  EXPECT_EQ(2, c.values(0).count())
      << "c += b: Expected c.values(0).count() 2: " << c.values(0).count();
  EXPECT_EQ(2, c.values(1).count())
      << "c += b: Expected c.values(1).count() 2: " << c.values(1).count();
  EXPECT_EQ(2, c.values(2).count())
      << "c += b: Expected c.values(2).count() 2: " << c.values(2).count();
  EXPECT_EQ(2, c.values(3).count())
      << "c += b: Expected c.values(3).count() 2: " << c.values(3).count();

  // Try "+=" of Histogram c {{1}, {4}}
  //        with Histogram a {{0}, {3}, {5}, {1}}.
  // Expect Histogram c {{1, 0}, {4, 3}, {0, 5}, {0, 1}}.
  c.CopyFrom(b);
  c += a;
  EXPECT_EQ(4, c.values_size())
      << "c += a: Expected c.values_size() 4: " << c.values_size();
  EXPECT_FEQ(0.5, Mean(c.values(0)))
      << "c += a: Expected c.values(0) 0.5: " << Mean(c.values(0));
  EXPECT_FEQ(3.5, Mean(c.values(1)))
      << "c += a: Expected c.values(1) 3.5: " << Mean(c.values(1));
  EXPECT_FEQ(2.5, Mean(c.values(2)))
      << "c += a: Expected c.values(2) 2.5: " << Mean(c.values(2));
  EXPECT_FEQ(0.5, Mean(c.values(3)))
      << "c += a: Expected c.values(3) 0.5: " << Mean(c.values(3));
  EXPECT_EQ(2, c.values(0).count())
      << "c += a: Expected c.values(0).count() 2: " << c.values(0).count();
  EXPECT_EQ(2, c.values(1).count())
      << "c += a: Expected c.values(1).count() 2: " << c.values(1).count();
  EXPECT_EQ(2, c.values(2).count())
      << "c += a: Expected c.values(2).count() 2: " << c.values(2).count();
  EXPECT_EQ(2, c.values(3).count())
      << "c += a: Expected c.values(3).count() 2: " << c.values(3).count();

  // Try "+=" of Histogram c {{1}, {4}}
  //        with "+" of Histogram a {{0}, {3}, {5}, {1}}
  //               with Histogram a {{0}, {3}, {5}, {1}}.
  // Expect Histogram c {{1, 0, 0}, {4, 3, 3}, {0, 5, 5}, {0, 1, 1}}.
  c.CopyFrom(b);
  c += a + a;
  EXPECT_EQ(4, c.values_size())
      << "c += a + a: Expected c.values_size() 4: " << c.values_size();
  EXPECT_FEQ((2.0 * 0.0 + 1.0) / 3.0, Mean(c.values(0)))
      << "c += a + a: Expected c.values(0) " << (2.0 * 0.0 + 1.0) / 3.0
      << ": " << Mean(c.values(0));
  EXPECT_FEQ((2.0 * 3.0 + 4.0) / 3.0, Mean(c.values(1)))
      << "c += a + a: Expected c.values(1) " << (2.0 * 3.0 + 4.0) / 3.0
      << ": " << Mean(c.values(1));
  EXPECT_FEQ((2.0 * 5.0 + 0.0) / 3.0, Mean(c.values(2)))
      << "c += a + a: Expected c.values(2) " << (2.0 * 5.0 + 0.0) / 3.0
      << ": " << Mean(c.values(2));
  EXPECT_FEQ((2.0 * 1.0 + 0.0) / 3.0, Mean(c.values(3)))
      << "c += a + a: Expected c.values(3) " << (2.0 * 1.0 + 0.0) / 3.0
      << ": " << Mean(c.values(3));
  EXPECT_EQ(3, c.values(0).count())
      << "c += a + a: Expected c.values(0).count() 3: " << c.values(0).count();
  EXPECT_EQ(3, c.values(1).count())
      << "c += a + a: Expected c.values(1).count() 3: " << c.values(1).count();
  EXPECT_EQ(3, c.values(2).count())
      << "c += a + a: Expected c.values(2).count() 3: " << c.values(2).count();
  EXPECT_EQ(3, c.values(3).count())
      << "c += a + a: Expected c.values(3).count() 3: " << c.values(3).count();

  // Try "+=" of Histogram c {{0}, {3}, {5}, {1}}
  //        with "+" of Histogram b {{1}, {4}}
  //               with Histogram b {{1}, {4}}.
  // Expect Histogram c {{0, 1, 1}, {3, 4, 4}, {5, 0, 0}, {1, 0, 0}}.
  c.CopyFrom(a);
  c += b + b;
  EXPECT_EQ(4, c.values_size())
      << "c += b + b: Expected c.values_size() 4: " << c.values_size();
  EXPECT_FEQ((0.0 + 2.0 * 1.0) / 3.0, Mean(c.values(0)))
      << "c += b + b: Expected c.values(0) " << (0.0 + 2.0 * 1.0) / 3.0
      << ": " << Mean(c.values(0));
  EXPECT_FEQ((3.0 + 2.0 * 4.0) / 3.0, Mean(c.values(1)))
      << "c += b + b: Expected c.values(1) " << (3.0 + 2.0 * 4.0) / 3.0
      << ": " << Mean(c.values(1));
  EXPECT_FEQ((5.0 + 2.0 * 0.0) / 3.0, Mean(c.values(2)))
      << "c += b + b: Expected c.values(2) " << (5.0 + 2.0 * 0.0) / 3.0
      << ": " << Mean(c.values(2));
  EXPECT_FEQ((1.0 + 2.0 * 0.0) / 3.0, Mean(c.values(3)))
      << "c += b + b: Expected c.values(3) " << (1.0 + 2.0 * 0.0) / 3.0
      << ": " << Mean(c.values(3));
  EXPECT_EQ(3, c.values(0).count())
      << "c += b + b: Expected c.values(0).count() 3: " << c.values(0).count();
  EXPECT_EQ(3, c.values(1).count())
      << "c += b + b: Expected c.values(1).count() 3: " << c.values(1).count();
  EXPECT_EQ(3, c.values(2).count())
      << "c += b + b: Expected c.values(2).count() 3: " << c.values(2).count();
  EXPECT_EQ(3, c.values(3).count())
      << "c += b + b: Expected c.values(3).count() 3: " << c.values(3).count();
}

TEST_F(CognonTest, CheckRunExperiment) {
  TrainConfig config;

  config.set_w(5);
  config.mutable_config()->set_c(1);
  config.mutable_config()->set_d1(1);
  config.mutable_config()->set_d2(1);
  config.mutable_config()->set_h(10);
  config.mutable_config()->set_q(0.362000);
  config.mutable_config()->set_r(30);

  NeuronStatistics result;
  RunExperiment(config, &result);
  EXPECT_EQ(round(Mean(result.true_count())), result.config().w());
  EXPECT_EQ(round(Mean(result.false_count())), 100000);

  result.Clear();
  config.set_num_test_words(100);
  RunExperiment(config, &result);
  EXPECT_EQ(round(Mean(result.true_count())), result.config().w());
  EXPECT_TRUE(config.has_num_test_words());
  EXPECT_EQ(config.num_test_words(), 100);
  EXPECT_TRUE(result.config().has_num_test_words());
  EXPECT_EQ(result.config().num_test_words(), 100);
  EXPECT_EQ(round(Mean(result.false_count())), 100);
}

}  // namespace cognon

int main(int argc, char **argv) {
  CALL_TEST(cognon::CheckStatistic);
  CALL_TEST(cognon::CheckHistogram);
  CALL_TEST(cognon::CheckRunExperiment);
}
