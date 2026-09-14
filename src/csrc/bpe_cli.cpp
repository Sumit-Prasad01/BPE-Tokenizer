#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <sstream>
#include <iomanip>
#include "bpe_engine.hpp"

static void print_usage(const char* prog) {
    std::cout << "================================================================" << std::endl;
    std::cout << " ⚡ High-Performance Native C++ BPE Tokenizer Engine CLI ⚡ " << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << "Usage: " << prog << " --vocab <vocab.json> --merges <merges.txt> [OPTIONS]" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  --encode <text>         Encode string and print token IDs" << std::endl;
    std::cout << "  --decode <ids>          Decode comma/space separated token IDs" << std::endl;
    std::cout << "  --stream <text>         Test token-by-token streaming decoder" << std::endl;
    std::cout << "  --benchmark <file>      Run high-speed file benchmark" << std::endl;
    std::cout << "  --interactive           Launch interactive terminal encoder REPL" << std::endl;
    std::cout << "  --threads <N>           Thread count for batch operations (default: auto)" << std::endl;
    std::cout << "  --help                  Show this help message" << std::endl;
    std::cout << "================================================================" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string vocab_path = "experiments/runs/20260914_023959_exp_vocab_64k/vocab.json";
    std::string merges_path = "experiments/runs/20260914_023959_exp_vocab_64k/merges.txt";
    std::string encode_text;
    std::string decode_ids;
    std::string stream_text;
    std::string benchmark_file;
    bool interactive = false;
    int num_threads = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--vocab" && i + 1 < argc) {
            vocab_path = argv[++i];
        } else if (arg == "--merges" && i + 1 < argc) {
            merges_path = argv[++i];
        } else if (arg == "--encode" && i + 1 < argc) {
            encode_text = argv[++i];
        } else if (arg == "--decode" && i + 1 < argc) {
            decode_ids = argv[++i];
        } else if (arg == "--stream" && i + 1 < argc) {
            stream_text = argv[++i];
        } else if (arg == "--benchmark" && i + 1 < argc) {
            benchmark_file = argv[++i];
        } else if (arg == "--interactive") {
            interactive = true;
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::atoi(argv[++i]);
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    bpe::BPEEngine engine;
    auto t0 = std::chrono::high_resolution_clock::now();
    if (!engine.load_vocab(vocab_path)) {
        std::cerr << "Error: Failed to load vocab from: " << vocab_path << std::endl;
        return 1;
    }
    if (!engine.load_merges(merges_path)) {
        std::cerr << "Error: Failed to load merges from: " << merges_path << std::endl;
        return 1;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[bpe_engine] Loaded vocab (" << engine.vocab_size() 
              << " tokens) and merges in " << std::fixed << std::setprecision(2) 
              << load_ms << " ms\n" << std::endl;

    // Mode 1: Encode string
    if (!encode_text.empty()) {
        auto t_start = std::chrono::high_resolution_clock::now();
        auto tokens = engine.encode(encode_text);
        auto t_end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

        std::cout << "Input (" << encode_text.size() << " bytes): " << encode_text << std::endl;
        std::cout << "Tokens (" << tokens.size() << "): [";
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::cout << tokens[i] << (i + 1 < tokens.size() ? ", " : "");
        }
        std::cout << "]" << std::endl;
        double cr = tokens.empty() ? 0.0 : (double)encode_text.size() / tokens.size();
        std::cout << "Compression Ratio: " << std::fixed << std::setprecision(2) << cr << " Bytes/Token" << std::endl;
        std::cout << "Latency: " << std::fixed << std::setprecision(2) << us << " us" << std::endl;
        return 0;
    }

    // Mode 2: Decode tokens
    if (!decode_ids.empty()) {
        std::vector<int> tokens;
        std::stringstream ss(decode_ids);
        std::string item;
        while (std::getline(ss, item, ',')) {
            std::stringstream item_ss(item);
            int tok;
            while (item_ss >> tok) {
                tokens.push_back(tok);
            }
        }
        std::string decoded = engine.decode(tokens);
        std::cout << "Decoded (" << decoded.size() << " bytes): " << decoded << std::endl;
        return 0;
    }

    // Mode 3: Streaming token-by-token simulation
    if (!stream_text.empty()) {
        std::cout << "Streaming input: " << stream_text << std::endl;
        auto tokens = engine.encode(stream_text);
        auto decoder = engine.create_streaming_decoder();
        std::cout << "Token-by-token emission: ";
        for (int tok : tokens) {
            std::string chunk = decoder->feed(tok);
            if (!chunk.empty()) {
                std::cout << "[" << chunk << "]" << std::flush;
            }
        }
        std::string tail = decoder->flush();
        if (!tail.empty()) {
            std::cout << "[" << tail << "]" << std::flush;
        }
        std::cout << "\nStream completed successfully." << std::endl;
        return 0;
    }

    // Mode 4: File benchmark
    if (!benchmark_file.empty()) {
        std::ifstream bf(benchmark_file, std::ios::binary);
        if (!bf.is_open()) {
            std::cerr << "Error: Cannot open benchmark file: " << benchmark_file << std::endl;
            return 1;
        }
        std::string content((std::istreambuf_iterator<char>(bf)),
                             std::istreambuf_iterator<char>());
        bf.close();

        std::cout << "Benchmarking on: " << benchmark_file 
                  << " (" << (content.size() / 1024.0 / 1024.0) << " MB)" << std::endl;

        // Split into lines
        std::vector<std::string> lines;
        std::stringstream ss(content);
        std::string line;
        while (std::getline(ss, line)) {
            if (!line.empty()) lines.push_back(line);
        }

        auto b_start = std::chrono::high_resolution_clock::now();
        auto batch_tokens = engine.encode_batch(lines, 0.0f, num_threads);
        auto b_end = std::chrono::high_resolution_clock::now();

        size_t total_tokens = 0;
        for (const auto& tvec : batch_tokens) {
            total_tokens += tvec.size();
        }

        double sec = std::chrono::duration<double>(b_end - b_start).count();
        double tok_per_sec = total_tokens / sec;
        double mb_per_sec = (content.size() / 1024.0 / 1024.0) / sec;

        std::cout << "--------------------------------------------------------" << std::endl;
        std::cout << "Lines processed:     " << lines.size() << std::endl;
        std::cout << "Total tokens:        " << total_tokens << std::endl;
        std::cout << "Execution time:      " << std::fixed << std::setprecision(4) << sec << " s" << std::endl;
        std::cout << "⚡ Throughput:        " << std::fixed << std::setprecision(2) << tok_per_sec << " tokens/sec" << std::endl;
        std::cout << "⚡ Speed:             " << std::fixed << std::setprecision(2) << mb_per_sec << " MB/sec" << std::endl;
        std::cout << "--------------------------------------------------------" << std::endl;
        return 0;
    }

    // Mode 5: Interactive terminal REPL
    if (interactive) {
        std::cout << "========================================================" << std::endl;
        std::cout << " ⚡ Native C++ Interactive Tokenizer REPL" << std::endl;
        std::cout << " Type text to tokenize (or 'quit' / 'exit' to leave)" << std::endl;
        std::cout << "========================================================" << std::endl;

        std::string input;
        while (true) {
            std::cout << "\n>>> " << std::flush;
            if (!std::getline(std::cin, input)) break;
            if (input == "quit" || input == "exit") break;
            if (input.empty()) continue;

            auto t_start = std::chrono::high_resolution_clock::now();
            auto tokens = engine.encode(input);
            auto t_end = std::chrono::high_resolution_clock::now();
            double us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

            std::cout << "Tokens (" << tokens.size() << "): [";
            for (size_t i = 0; i < tokens.size(); ++i) {
                std::cout << tokens[i] << (i + 1 < tokens.size() ? ", " : "");
            }
            std::cout << "]" << std::endl;

            double cr = tokens.empty() ? 0.0 : (double)input.size() / tokens.size();
            std::cout << "Bytes: " << input.size() << " | Tokens: " << tokens.size() 
                      << " | CR: " << std::fixed << std::setprecision(2) << cr << " B/T"
                      << " | Latency: " << std::fixed << std::setprecision(2) << us << " us" << std::endl;

            // Verify roundtrip
            std::string decoded = engine.decode(tokens);
            if (decoded != input) {
                std::cout << "⚠️ Warning: Roundtrip mismatch!" << std::endl;
            }
        }
        std::cout << "\nExiting REPL." << std::endl;
        return 0;
    }

    print_usage(argv[0]);
    return 0;
}
