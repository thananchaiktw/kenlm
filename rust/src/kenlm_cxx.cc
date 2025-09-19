#include "rust/src/kenlm_cxx.hh"
#include "python/score_sentence.hh"
#include <sstream>
#include <memory>

namespace kenlm_cxx {

std::unique_ptr<lm::ngram::Config> new_config() {
    return std::make_unique<lm::ngram::Config>();
}

void set_load_method(lm::ngram::Config &config, lm::ngram::LoadMethod method) {
    config.load_method = method;
}

std::unique_ptr<lm::base::Model> load_model(const std::string &path, const lm::ngram::Config &config) {
    return std::unique_ptr<lm::base::Model>(lm::ngram::LoadVirtual(path.c_str(), config));
}

unsigned int get_order(const lm::base::Model &model) {
    return model.Order();
}

bool vocab_contains(const lm::base::Model &model, const std::string &word) {
    return model.BaseVocabulary().Index(word.c_str()) != 0;
}

float score(const lm::base::Model &model, const std::string &sentence, bool bos, bool eos) {
    if (bos && eos) {
        return lm::base::ScoreSentence(&model, sentence.c_str());
    }

    lm::ngram::State state;
    if (bos) {
        model.BeginSentenceWrite(&state);
    } else {
        model.NullContextWrite(&state);
    }

    float total = 0;
    std::string word;
    std::stringstream ss(sentence);
    while (ss >> word) {
        lm::ngram::State out_state;
        total += model.BaseScore(&state, model.BaseVocabulary().Index(word.c_str()), &out_state);
        state = out_state;
    }

    if (eos) {
        lm::ngram::State out_state;
        total += model.BaseScore(&state, model.BaseVocabulary().EndSentence(), &out_state);
    }
    return total;
}

std::vector<FullScoreResult> full_scores(const lm::base::Model &model, const std::string &sentence, bool bos, bool eos) {
    std::vector<FullScoreResult> results;
    lm::ngram::State state;
    if (bos) {
        model.BeginSentenceWrite(&state);
    } else {
        model.NullContextWrite(&state);
    }

    std::string word;
    std::stringstream ss(sentence);
    while (ss >> word) {
        lm::ngram::State out_state;
        lm::FullScoreReturn ret = model.BaseFullScore(&state, model.BaseVocabulary().Index(word.c_str()), &out_state);
        results.push_back({ret.prob, (int)ret.ngram_length, model.BaseVocabulary().Index(word.c_str()) == 0});
        state = out_state;
    }

    if (eos) {
        lm::ngram::State out_state;
        lm::FullScoreReturn ret = model.BaseFullScore(&state, model.BaseVocabulary().EndSentence(), &out_state);
        results.push_back({ret.prob, (int)ret.ngram_length, false});
    }

    return results;
}

void begin_sentence_write(const lm::base::Model &model, lm::ngram::State &state) {
    model.BeginSentenceWrite(&state);
}

void null_context_write(const lm::base::Model &model, lm::ngram::State &state) {
    model.NullContextWrite(&state);
}

float base_score(const lm::base::Model &model, lm::ngram::State &in_state, const std::string &word, lm::ngram::State &out_state) {
    return model.BaseScore(&in_state, model.BaseVocabulary().Index(word.c_str()), &out_state);
}

} // namespace kenlm_cxx
