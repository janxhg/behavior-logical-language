#pragma once
#include <vector>
#include <memory>

namespace brainll {

class ContinualLearning {
public:
    ContinualLearning();
    ~ContinualLearning();
    
    void startNewTask(int task_id);
    void addExemplar(const std::vector<double>& data, int label);
    double computeEWCLoss(const std::vector<double>& old_params, const std::vector<double>& new_params);
    std::vector<std::vector<double>> getReplayExamples(size_t num_examples);
    void updateFisherInformation(const std::vector<double>& gradients);
    
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace brainll