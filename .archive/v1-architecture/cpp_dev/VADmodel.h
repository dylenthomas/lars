#pragma once

#include <iostream>
#include <vector>
#include <sstream>
#include <cstring>
#include <limits>
#include <chrono>
#include <iomanip>
#include <memory>
#include <string>
#include <stdexcept>
#include <cstdio>
#include <cstdarg>
#if __cplusplus < 201703L
#include <memory>
#endif

#include "onnxruntime_cxx_api.h"

// https://github.com/snakers4/silero-vad/blob/master/examples/cpp/silero-vad-onnx.cpp#L4

class VAD {
private:
    // ONNX Runtime resources
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::shared_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, 
        OrtMemTypeCPU
    );

    // context
    const int context_samples = 64; // for 16kHz use 64 samples as context
    std::vector<float> _context;

    int window_size_samples;
    int effective_window_size; // window_size_samples + context_samples

    // ONNX Runtime input/output buffers
    std::vector<Ort::Value> ort_inputs;
    std::vector<const char*> input_node_names = {"input", "state", "sr"};
    std::vector<float> input;
    unsigned int size_state = 2 * 1 * 128;
    std::vector<float> _state;
    std::vector<int64_t> sr;
    int64_t input_node_dims[2] = {};
    const int64_t state_node_dims[3] = {2, 1, 128};
    const int64_t sr_node_dims[1] = {1};
    std::vector<Ort::Value> ort_outputs;
    std::vector<const char*> output_node_names = {"output", "stateN"};

    int sample_rate;

    void init_onnx_model(const char* model_path) {
        init_engine_threads(1, 1);
        session = std::make_shared<Ort::Session>(env, model_path, session_options);
    }

    void init_engine_threads(int inter_threads, int intra_threads) {
        session_options.SetIntraOpNumThreads(intra_threads);
        session_options.SetInterOpNumThreads(inter_threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    }

    void reset_states() {
        std::memset(_state.data(), 0, _state.size() * sizeof(float));
        std::fill(_context.begin(), _context.end(), 0.0f);
    }

    // run inference on a single mic buffer
    float predict(const std::vector<float>& buffer) {
        // build a new input: [context, buffer]
        std::vector<float> new_buffer(effective_window_size, 0.0f);
        std::copy(_context.begin(), _context.end(), new_buffer.begin());
        std::copy(buffer.begin(), buffer.end(), new_buffer.begin() + context_samples);
        input = new_buffer;

        Ort::Value input_ort = Ort::Value::CreateTensor<float>(
            memory_info, input.data(), input.size(), input_node_dims, 2);
        Ort::Value state_ort = Ort::Value::CreateTensor<float>(
            memory_info, _state.data(), _state.size(), state_node_dims, 3);
        Ort::Value sr_ort = Ort::Value::CreateTensor<int64_t>(
            memory_info, sr.data(), sr.size(), sr_node_dims, 1);
        ort_inputs.clear();
        ort_inputs.emplace_back(std::move(input_ort));
        ort_inputs.emplace_back(std::move(state_ort));
        ort_inputs.emplace_back(std::move(sr_ort));

        ort_outputs = session->Run(
            Ort::RunOptions{ nullptr },
            input_node_names.data(), ort_inputs.data(), ort_inputs.size(),
            output_node_names.data(), output_node_names.size()
        );

        float speech_prob = ort_outputs[0].GetTensorMutableData<float>()[0];
        float* stateN = ort_outputs[1].GetTensorMutableData<float>();
        std::memcpy(_state.data(), stateN, size_state * sizeof(float));
        std::copy(new_buffer.begin(), new_buffer.end(), _context.begin());

#ifndef __DEBUG_SPEECH_PROB__
        printf("Chance of speech: %.3f\n", speech_prob);
#endif
        return speech_prob;
    }

public:
    float process(const std::vector<float>& input_buffer) {
        return predict(input_buffer);
    }

    void reset() {
        reset_states();
    }

public: // Constructor
    VAD(const char* model_path, const int rate, const int buffer_samples) {
        this->sample_rate = rate;
        this->effective_window_size = buffer_samples + context_samples;

        init_onnx_model(model_path);
    }
};
