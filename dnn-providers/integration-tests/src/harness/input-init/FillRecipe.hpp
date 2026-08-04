// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace hipdnn_integration_tests
{

// Describes how to fill a single input tensor.
//
// Kind taxonomy (decision test: "can fill() generate this value from
// (range + seed + distribution) alone, without seeing other tensors?"):
//
//   FREE       — yes: locally generatable. Pick a Distribution; never add
//                a new Kind for a distribution.
//   FIXED      — one constant value, broadcast to every element.
//   STRUCTURED — no: needs cross-tensor alignment or dtype-specific structure
//                that fill() cannot see. Correctly refused → test skips.
//                Do not fabricate; a green test on garbage input is worse
//                than an honest skip.
//   DERIVED    — no: must equal another op's output (e.g. fwd→bwd coupling).
//                Refused unless the producing node is in-graph.
struct FillRecipe
{
    enum class Kind
    {
        FREE,
        FIXED,
        STRUCTURED,
        DERIVED,
    };

    enum class Distribution
    {
        UNIFORM,
        POWER_OF_TWO,
    };

    static constexpr float K_DEFAULT_LO = -1.0f;
    static constexpr float K_DEFAULT_HI = 1.0f;

    Kind kind = Kind::FREE;
    Distribution distribution = Distribution::UNIFORM;
    float lo = K_DEFAULT_LO;
    float hi = K_DEFAULT_HI;
    float value = 0.0f;

    static FillRecipe free(float lo, float hi)
    {
        FillRecipe f;
        f.kind = Kind::FREE;
        f.lo = lo;
        f.hi = hi;
        return f;
    }
    static FillRecipe free(float lo, float hi, Distribution dist)
    {
        FillRecipe f;
        f.kind = Kind::FREE;
        f.distribution = dist;
        f.lo = lo;
        f.hi = hi;
        return f;
    }
    static FillRecipe fixed(float v)
    {
        FillRecipe f;
        f.kind = Kind::FIXED;
        f.value = v;
        return f;
    }
    static FillRecipe structured()
    {
        FillRecipe f;
        f.kind = Kind::STRUCTURED;
        return f;
    }
    static FillRecipe derived()
    {
        FillRecipe f;
        f.kind = Kind::DERIVED;
        return f;
    }
};

} // namespace hipdnn_integration_tests
