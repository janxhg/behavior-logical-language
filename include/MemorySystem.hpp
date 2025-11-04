#ifndef BRAINLL_MEMORY_SYSTEM_HPP
#define BRAINLL_MEMORY_SYSTEM_HPP

#include <vector>

namespace brainll {

class MemorySystem {
public:
    MemorySystem(size_t capacity = 1000, double decay_rate = 0.01);
    
    void store(const std::vector<double>& data, double importance = 1.0);
    std::vector<double> recall(const std::vector<double>& query, size_t num_results = 1);
    void update(double dt);
    void consolidate();
    size_t getMemoryCount() const;
    double getAverageImportance() const;
    void clear();
    void setCapacity(size_t new_capacity);
    void setDecayRate(double new_decay_rate);
};

} // namespace brainll

#endif // BRAINLL_MEMORY_SYSTEM_HPP