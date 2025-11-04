#ifndef BRAINLL_STATE_MACHINE_HPP
#define BRAINLL_STATE_MACHINE_HPP

#include <vector>
#include <string>
#include <memory>

namespace brainll {

class StateMachineImpl;

enum class StateType {
    IDLE,
    LEARNING,
    INFERENCE,
    ADAPTATION,
    CONSOLIDATION,
    EXPLORATION,
    EXPLOITATION,
    ATTENTION_FOCUS,
    MEMORY_RECALL,
    DECISION_MAKING,
    ERROR_CORRECTION,
    OPTIMIZATION
};

class StateMachine {
public:
    StateMachine();
    ~StateMachine();
    
    void update(double dt);
    bool transition(const std::string& trigger);
    void forceTransition(StateType new_state);
    StateType getCurrentState() const;
    std::string getCurrentStateName() const;
    double getStateTime() const;
    double getTotalTime() const;
    void addEvent(const std::string& event, double delay);
    std::vector<std::string> getAvailableTransitions() const;
    void reset();
    
private:
    std::unique_ptr<StateMachineImpl> impl;
};

} // namespace brainll

#endif // BRAINLL_STATE_MACHINE_HPP