// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

// Utils.hpp first: MasterSolutionLibrary.hpp calls concatenate() from a
// template without including it, so it must already be visible.
#include <Tensile/Utils.hpp>

#include "SolutionIterator.hpp"

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/hip/HipHardware.hpp>

#include <origami/simulator/tensilelite/formocast_simulator.hpp>

#include <cstring>
#include <memory>

using namespace TensileLite;
using namespace TensileLite::Client;

// Performance prediction is only usable on the architectures Formocast carries
// hardware constants for; on the others Formocast::setHardware() throws, and
// the client must fall back to iterating solutions by index. These tests pin
// that fallback. The architecture is fabricated in memory (as in
// CuCount_test.cpp), so they exercise every architecture regardless of the GPU
// the test binary happens to run on.
namespace
{
    constexpr int kNumSolutions = 4;

    // Only `arch` is meaningful to these tests: isPredictionAvailable() reads
    // nothing else off the hardware, and the fallback path never runs the
    // simulator. The remaining fields just have to be self-consistent, so they
    // carry placeholder values rather than any real device's specifications.
    // NUM_XCD must be non-zero — hardware_t's constructor computes N_CU/NUM_XCD.
    origami::hardware_t makeAnalyticalHardware(origami::hardware_t::architecture_t arch)
    {
        constexpr size_t kNumCUs  = 1;
        constexpr size_t kNumXCDs = 1;
        return origami::hardware_t(arch,
                                   kNumCUs,
                                   /*lds_capacity=*/0,
                                   /*rf_capacity=*/0,
                                   kNumXCDs,
                                   /*mem1_perf_ratio=*/0.0,
                                   /*mem2_perf_ratio=*/0.0,
                                   /*mem3_perf_ratio=*/0.0,
                                   /*L2_capacity=*/0,
                                   /*compute_clock_ghz=*/0.0,
                                   /*parallel_mi_cu=*/0,
                                   std::make_tuple(0.0, 0.0, 0.0));
    }

    std::shared_ptr<Hardware> makeHipDevice(AMDGPU::Processor          processor,
                                            char const*                archName,
                                            origami::hardware_t const& hw)
    {
        auto device             = std::make_shared<hip::HipAMDGPU>();
        device->processor       = processor;
        device->computeUnitCount = static_cast<int>(hw.N_CU);
        device->deviceName      = archName;
        // HipAMDGPU::archName() returns properties.gcnArchName, and `properties`
        // is not zero-initialized on a default-constructed device. Set it so the
        // "prediction unavailable" warning does not read an unterminated buffer.
        std::snprintf(device->properties.gcnArchName,
                      sizeof(device->properties.gcnArchName),
                      "%s",
                      archName);
        device->analyticalHardware = std::make_shared<origami::hardware_t>(hw);
        return device;
    }

    std::shared_ptr<MasterSolutionLibrary<ContractionProblemGemm>> makeLibrary(int count)
    {
        auto library = std::make_shared<MasterSolutionLibrary<ContractionProblemGemm>>();
        for(int i = 0; i < count; i++)
        {
            auto solution               = std::make_shared<ContractionSolution>();
            solution->index             = i;
            solution->libraryLogicIndex = i;
            library->solutions[i]       = solution;
        }
        return library;
    }

    ContractionProblemGemm makeGemmProblem()
    {
        auto problem = ContractionProblemGemm::GEMM(
            false, false, 256, 256, 256, 256, 256, 256, 1.0, false, 1);
        problem.setComputeInputTypeA(rocisa::DataType::Float);
        problem.setComputeInputTypeB(rocisa::DataType::Float);
        return problem;
    }

    // Drives the iterator to completion and returns the solution indices it
    // yielded, in order.
    std::vector<int> collectVisitedIndices(AllSolutionsIterator& iter)
    {
        std::vector<int> visited;
        while(iter.moreSolutionsInProblem())
        {
            auto solution = iter.getSolution();
            if(!solution)
                break;
            visited.push_back(solution->index);
            iter.postSolution();

            if(visited.size() > static_cast<size_t>(kNumSolutions))
                break; // guard against a non-advancing iterator
        }
        return visited;
    }
}

// gfx1200 is recognized by origami but has no Formocast hardware constants.
// Requesting prediction there must degrade to plain index iteration rather than
// leaving the prediction queue empty: an empty queue makes
// moreSolutionsInProblem() false immediately, so the client would benchmark
// nothing and still exit successfully.
TEST(SolutionIterator, UnsupportedArchBenchmarksAllSolutions)
{
    auto hw     = makeAnalyticalHardware(origami::hardware_t::architecture_t::gfx1200);
    auto device = makeHipDevice(AMDGPU::Processor::gfx1200, "gfx1200", hw);

    ASSERT_FALSE(origami::Formocast::isArchSupported(hw.arch))
        << "test premise: gfx1200 must lack Formocast hardware constants";

    auto problem = makeGemmProblem();
    AllSolutionsIterator iter(makeLibrary(kNumSolutions),
                              device,
                              /*predictionThreshold=*/0.1,
                              /*firstSolutionIdx=*/0,
                              /*numSolutions=*/kNumSolutions,
                              /*printWinnerOnly=*/false);
    iter.preProblem(&problem);

    EXPECT_EQ(collectVisitedIndices(iter), (std::vector<int>{0, 1, 2, 3}))
        << "prediction is unavailable on gfx1200, so every solution should be "
           "visited by index";
}

// The same fallback must apply when the device carries no analytical hardware
// at all, which is the case origami cannot describe.
TEST(SolutionIterator, NullAnalyticalHardwareBenchmarksAllSolutions)
{
    auto device       = std::make_shared<hip::HipAMDGPU>();
    device->processor = AMDGPU::Processor::gfx1200;
    std::snprintf(device->properties.gcnArchName,
                  sizeof(device->properties.gcnArchName),
                  "%s",
                  "gfx1200");
    ASSERT_EQ(device->analyticalHardware, nullptr);

    auto problem = makeGemmProblem();
    AllSolutionsIterator iter(makeLibrary(kNumSolutions),
                              device,
                              /*predictionThreshold=*/0.1,
                              /*firstSolutionIdx=*/0,
                              /*numSolutions=*/kNumSolutions,
                              /*printWinnerOnly=*/false);
    iter.preProblem(&problem);

    EXPECT_EQ(collectVisitedIndices(iter), (std::vector<int>{0, 1, 2, 3}));
}

// A threshold above 1.0 disables prediction outright. This already used the
// index path before the architecture guard existed; it must keep doing so on an
// architecture Formocast does support.
TEST(SolutionIterator, SupportedArchWithoutPredictionBenchmarksAllSolutions)
{
    auto hw     = makeAnalyticalHardware(origami::hardware_t::architecture_t::gfx942);
    auto device = makeHipDevice(AMDGPU::Processor::gfx942, "gfx942", hw);

    ASSERT_TRUE(origami::Formocast::isArchSupported(hw.arch));

    auto problem = makeGemmProblem();
    AllSolutionsIterator iter(makeLibrary(kNumSolutions),
                              device,
                              /*predictionThreshold=*/2.0,
                              /*firstSolutionIdx=*/0,
                              /*numSolutions=*/kNumSolutions,
                              /*printWinnerOnly=*/false);
    iter.preProblem(&problem);

    EXPECT_EQ(collectVisitedIndices(iter), (std::vector<int>{0, 1, 2, 3}));
}
