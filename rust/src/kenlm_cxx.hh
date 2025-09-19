#pragma once

#include "cxx.h"
#include "lm/model.hh"
#include "lm/state.hh"
#include "lm/config.hh"

namespace kenlm_cxx {

// Opaque types
using Model = lm::base::Model;
using State = lm::ngram::State;
using Config = lm::ngram::Config;

// Functions
std::unique_ptr<Config> new_config();
void set_load_method(Config &config, lm::ngram::LoadMethod method);

std::unique_ptr<Model> load_model(const std::string &path, const Config &config);

unsigned int get_order(const Model &model);
bool vocab_contains(const Model &model, const std::string &word);

float score(const Model &model, const std::string &sentence, bool bos, bool eos);

struct FullScoreResult {
    float log_prob;
    int ngram_length;
    bool oov;
};

std::vector<FullScoreResult> full_scores(const Model &model, const std::string &sentence, bool bos, bool eos);

void begin_sentence_write(const Model &model, State &state);
void null_context_write(const Model &model, State &state);

float base_score(const Model &model, State &in_state, const std::string &word, State &out_state);

} // namespace kenlm_cxx
