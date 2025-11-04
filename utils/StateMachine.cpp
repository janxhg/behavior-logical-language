#include "../../include/StateMachine.hpp"
#include "../../include/DebugConfig.hpp"
#include <iostream>
#include <chrono>
#include <map>
#include <queue>
#include <functional>
#include <algorithm>
#include <memory>

namespace brainll {

// State structure
struct State {
    StateType type;
    std::string name;
    std::vector<std::string> allowed_transitions;
    std::function<void()> on_enter;
    std::function<void()> on_exit;
    std::function<void(double)> on_update;
    
    State() : type(StateType::IDLE), name("") {}
    State(StateType t, const std::string& n) : type(t), name(n) {}
};

// Pending event structure
struct PendingEvent {
    std::string event;
    double trigger_time;
    
    bool operator>(const PendingEvent& other) const {
        return trigger_time > other.trigger_time;
    }
};

// StateMachine implementation
class StateMachineImpl {
public:
    std::map<StateType, State> states;
    StateType current_state = StateType::IDLE;
    double state_start_time = 0.0;
    double total_time = 0.0;
    std::priority_queue<PendingEvent, std::vector<PendingEvent>, std::greater<PendingEvent>> pending_events;
    
    void initializeStates() {
        // Initialize all states
        states[StateType::IDLE] = State(StateType::IDLE, "Idle");
        states[StateType::LEARNING] = State(StateType::LEARNING, "Learning");
        states[StateType::INFERENCE] = State(StateType::INFERENCE, "Inference");
        states[StateType::ADAPTATION] = State(StateType::ADAPTATION, "Adaptation");
        states[StateType::CONSOLIDATION] = State(StateType::CONSOLIDATION, "Consolidation");
        states[StateType::EXPLORATION] = State(StateType::EXPLORATION, "Exploration");
        states[StateType::EXPLOITATION] = State(StateType::EXPLOITATION, "Exploitation");
        states[StateType::ATTENTION_FOCUS] = State(StateType::ATTENTION_FOCUS, "Attention Focus");
        states[StateType::MEMORY_RECALL] = State(StateType::MEMORY_RECALL, "Memory Recall");
        states[StateType::DECISION_MAKING] = State(StateType::DECISION_MAKING, "Decision Making");
        states[StateType::ERROR_CORRECTION] = State(StateType::ERROR_CORRECTION, "Error Correction");
        states[StateType::OPTIMIZATION] = State(StateType::OPTIMIZATION, "Optimization");
        
        // Set up transitions
        states[StateType::IDLE].allowed_transitions = {"start", "learn", "infer"};
        states[StateType::LEARNING].allowed_transitions = {"consolidate", "adapt", "stop"};
        states[StateType::INFERENCE].allowed_transitions = {"learn", "stop", "decide"};
        states[StateType::ADAPTATION].allowed_transitions = {"continue_learning", "consolidate"};
        states[StateType::CONSOLIDATION].allowed_transitions = {"complete", "learn"};
        states[StateType::EXPLORATION].allowed_transitions = {"exploit", "learn"};
        states[StateType::EXPLOITATION].allowed_transitions = {"explore", "stop"};
        states[StateType::ATTENTION_FOCUS].allowed_transitions = {"process", "recall"};
        states[StateType::MEMORY_RECALL].allowed_transitions = {"decide", "focus"};
        states[StateType::DECISION_MAKING].allowed_transitions = {"execute", "recall"};
        states[StateType::ERROR_CORRECTION].allowed_transitions = {"adapt", "learn"};
        states[StateType::OPTIMIZATION].allowed_transitions = {"complete", "adapt"};
    }
    
    void changeState(StateType new_state) {
        if (states.find(current_state) != states.end() && states[current_state].on_exit) {
            states[current_state].on_exit();
        }
        
        current_state = new_state;
        state_start_time = total_time;
        
        if (states.find(current_state) != states.end() && states[current_state].on_enter) {
            states[current_state].on_enter();
        }
    }
    
    void transitionToNextState() {
        switch (current_state) {
            case StateType::LEARNING:
                changeState(StateType::CONSOLIDATION);
                break;
            case StateType::INFERENCE:
                changeState(StateType::IDLE);
                break;
            case StateType::ADAPTATION:
                changeState(StateType::LEARNING);
                break;
            case StateType::CONSOLIDATION:
                changeState(StateType::IDLE);
                break;
            case StateType::EXPLORATION:
                changeState(StateType::EXPLOITATION);
                break;
            case StateType::EXPLOITATION:
                changeState(StateType::IDLE);
                break;
            case StateType::ATTENTION_FOCUS:
                changeState(StateType::MEMORY_RECALL);
                break;
            case StateType::MEMORY_RECALL:
                changeState(StateType::DECISION_MAKING);
                break;
            case StateType::DECISION_MAKING:
                changeState(StateType::IDLE);
                break;
            case StateType::ERROR_CORRECTION:
                changeState(StateType::ADAPTATION);
                break;
            case StateType::OPTIMIZATION:
                changeState(StateType::IDLE);
                break;
            default:
                break;
        }
    }
    
    void processPendingEvents() {
        while (!pending_events.empty() && pending_events.top().trigger_time <= total_time) {
            auto event = pending_events.top();
            pending_events.pop();
            // Process event
        }
    }
};

// StateMachine constructor
StateMachine::StateMachine() {
    impl = std::make_unique<StateMachineImpl>();
    impl->initializeStates();
}

StateMachine::~StateMachine() = default;

void StateMachine::update(double dt) {
    impl->total_time += dt;
    
    auto& current_state_obj = impl->states[impl->current_state];
    if (current_state_obj.on_update) {
        current_state_obj.on_update(dt);
    }
    
    impl->processPendingEvents();
}

bool StateMachine::transition(const std::string& trigger) {
    auto& current_state_obj = impl->states[impl->current_state];
    
    auto it = std::find(current_state_obj.allowed_transitions.begin(), 
                       current_state_obj.allowed_transitions.end(), trigger);
    
    if (it != current_state_obj.allowed_transitions.end()) {
        // Map trigger to state
        if (trigger == "learn") impl->changeState(StateType::LEARNING);
        else if (trigger == "infer") impl->changeState(StateType::INFERENCE);
        else if (trigger == "adapt") impl->changeState(StateType::ADAPTATION);
        else if (trigger == "consolidate") impl->changeState(StateType::CONSOLIDATION);
        else if (trigger == "explore") impl->changeState(StateType::EXPLORATION);
        else if (trigger == "exploit") impl->changeState(StateType::EXPLOITATION);
        else if (trigger == "focus") impl->changeState(StateType::ATTENTION_FOCUS);
        else if (trigger == "recall") impl->changeState(StateType::MEMORY_RECALL);
        else if (trigger == "decide") impl->changeState(StateType::DECISION_MAKING);
        else if (trigger == "correct") impl->changeState(StateType::ERROR_CORRECTION);
        else if (trigger == "optimize") impl->changeState(StateType::OPTIMIZATION);
        else if (trigger == "stop" || trigger == "complete") impl->changeState(StateType::IDLE);
        else impl->transitionToNextState();
        
        return true;
    }
    
    return false;
}

void StateMachine::forceTransition(StateType new_state) {
    impl->changeState(new_state);
}

StateType StateMachine::getCurrentState() const {
    return impl->current_state;
}

std::string StateMachine::getCurrentStateName() const {
    auto it = impl->states.find(impl->current_state);
    if (it != impl->states.end()) {
        return it->second.name;
    }
    return "Unknown";
}

double StateMachine::getStateTime() const {
    return impl->total_time - impl->state_start_time;
}

double StateMachine::getTotalTime() const {
    return impl->total_time;
}

void StateMachine::addEvent(const std::string& event, double delay) {
    PendingEvent pending_event;
    pending_event.event = event;
    pending_event.trigger_time = impl->total_time + delay;
    impl->pending_events.push(pending_event);
}

std::vector<std::string> StateMachine::getAvailableTransitions() const {
    auto it = impl->states.find(impl->current_state);
    if (it != impl->states.end()) {
        return it->second.allowed_transitions;
    }
    return {};
}

void StateMachine::reset() {
    impl->current_state = StateType::IDLE;
    impl->state_start_time = 0.0;
    impl->total_time = 0.0;
    while (!impl->pending_events.empty()) {
        impl->pending_events.pop();
    }
}

// Behavioral State Manager
class BehavioralStateManager {
public:
    enum class BehaviorMode {
        PASSIVE,
        ACTIVE_LEARNING,
        PERFORMANCE,
        MAINTENANCE,
        EMERGENCY
    };
    
    BehavioralStateManager() : current_mode(BehaviorMode::PASSIVE), arousal_level(0.5), stress_level(0.0) {}
    
    void update(double dt) {
        state_machine.update(dt);
        updateArousalAndStress(dt);
        adaptBehaviorMode();
    }
    
    void setArousalLevel(double level) {
        arousal_level = std::max(0.0, std::min(1.0, level));
    }
    
    void setStressLevel(double level) {
        stress_level = std::max(0.0, std::min(1.0, level));
    }
    
    BehaviorMode getCurrentMode() const {
        return current_mode;
    }
    
    double getArousalLevel() const {
        return arousal_level;
    }
    
    double getStressLevel() const {
        return stress_level;
    }
    
    StateMachine& getStateMachine() {
        return state_machine;
    }
    
    void triggerEmergency() {
        current_mode = BehaviorMode::EMERGENCY;
        state_machine.forceTransition(StateType::ERROR_CORRECTION);
    }
    
    void resetToNormal() {
        current_mode = BehaviorMode::PASSIVE;
        stress_level = 0.0;
        arousal_level = 0.5;
        state_machine.reset();
    }
    
private:
    StateMachine state_machine;
    BehaviorMode current_mode;
    double arousal_level;
    double stress_level;
    
    void updateArousalAndStress(double dt) {
        // Natural decay of arousal and stress
        arousal_level += dt * 0.1 * (0.5 - arousal_level);
        stress_level = std::max(0.0, stress_level - dt * 0.05);
        
        // Adjust based on current state
        auto current_state = state_machine.getCurrentState();
        switch (current_state) {
            case StateType::LEARNING:
                arousal_level = std::min(1.0, arousal_level + dt * 0.1);
                break;
            case StateType::ERROR_CORRECTION:
                stress_level = std::min(1.0, stress_level + dt * 0.2);
                break;
            case StateType::CONSOLIDATION:
                arousal_level = std::max(0.0, arousal_level - dt * 0.05);
                break;
            default:
                break;
        }
    }
    
    void adaptBehaviorMode() {
        if (stress_level > 0.8) {
            current_mode = BehaviorMode::EMERGENCY;
        } else if (arousal_level > 0.7) {
            current_mode = BehaviorMode::ACTIVE_LEARNING;
        } else if (arousal_level > 0.3) {
            current_mode = BehaviorMode::PERFORMANCE;
        } else {
            current_mode = BehaviorMode::PASSIVE;
        }
    }
};

// Global state machine instance
static std::unique_ptr<BehavioralStateManager> global_behavioral_manager;

void initializeStateMachine() {
    global_behavioral_manager = std::make_unique<BehavioralStateManager>();
}

void updateStateMachine(double dt) {
    if (!global_behavioral_manager) initializeStateMachine();
    global_behavioral_manager->update(dt);
}

bool transitionState(const std::string& trigger) {
    if (!global_behavioral_manager) initializeStateMachine();
    return global_behavioral_manager->getStateMachine().transition(trigger);
}

std::string getCurrentStateName() {
    if (!global_behavioral_manager) return "Uninitialized";
    return global_behavioral_manager->getStateMachine().getCurrentStateName();
}

void setArousalLevel(double level) {
    if (!global_behavioral_manager) initializeStateMachine();
    global_behavioral_manager->setArousalLevel(level);
}

void setStressLevel(double level) {
    if (!global_behavioral_manager) initializeStateMachine();
    global_behavioral_manager->setStressLevel(level);
}

double getArousalLevel() {
    if (!global_behavioral_manager) return 0.5;
    return global_behavioral_manager->getArousalLevel();
}

double getStressLevel() {
    if (!global_behavioral_manager) return 0.0;
    return global_behavioral_manager->getStressLevel();
}

void triggerEmergencyMode() {
    if (!global_behavioral_manager) initializeStateMachine();
    global_behavioral_manager->triggerEmergency();
}

void resetStateMachine() {
    if (!global_behavioral_manager) initializeStateMachine();
    global_behavioral_manager->resetToNormal();
}

} // namespace brainll