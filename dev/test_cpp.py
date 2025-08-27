#!/usr/bin/env python3
# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.


from dev.benchmark_spot import benchmark_cpp_rollout

model_path = "/home/jzhang/Documents/judo/judo/models/xml/spot_locomotion.xml"
results = benchmark_cpp_rollout(model_path, 4, 100, 2)
print("C++ Results:", results)
