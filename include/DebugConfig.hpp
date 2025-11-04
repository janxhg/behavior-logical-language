/*
 * Copyright (C) 2024 Behavior Logical Language (BrainLL)
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef BRAINLL_DEBUG_CONFIG_HPP
#define BRAINLL_DEBUG_CONFIG_HPP

#include <iostream>
#include <string>

namespace brainll {

    enum class DebugLevel {
        SILENT = 0,    // No debug output
        ERROR = 1,     // Only errors
        WARNING = 2,   // Errors and warnings
        INFO = 3,      // Errors, warnings, and info
        VERBOSE = 4,   // All messages including verbose debug
        DEBUG = 5      // All messages including detailed C++ debug
    };

    class DebugConfig {
    public:
        static DebugConfig& getInstance() {
            static DebugConfig instance;
            return instance;
        }

        void setDebugLevel(DebugLevel level) {
            m_debug_level = level;
        }

        DebugLevel getDebugLevel() const {
            return m_debug_level;
        }

        void enableDebugMode(bool enable = true) {
            if (enable) {
                m_debug_level = DebugLevel::DEBUG;
            } else {
                m_debug_level = DebugLevel::WARNING; // Default to warnings only
            }
        }

        bool isDebugEnabled() const {
            return m_debug_level >= DebugLevel::DEBUG;
        }

        bool isVerboseEnabled() const {
            return m_debug_level >= DebugLevel::VERBOSE;
        }

        bool isInfoEnabled() const {
            return m_debug_level >= DebugLevel::INFO;
        }

        bool isWarningEnabled() const {
            return m_debug_level >= DebugLevel::WARNING;
        }

        bool isErrorEnabled() const {
            return m_debug_level >= DebugLevel::ERROR;
        }

        // Convenience methods for logging (C++14 compatible)
        void logDebug(const std::string& message) const {
            if (isDebugEnabled()) {
                std::cout << "[DEBUG C++] " << message << std::endl;
            }
        }

        void logVerbose(const std::string& message) const {
            if (isVerboseEnabled()) {
                std::cout << "[VERBOSE] " << message << std::endl;
            }
        }

        void logInfo(const std::string& message) const {
            if (isInfoEnabled()) {
                std::cout << "[INFO] " << message << std::endl;
            }
        }

        void logWarning(const std::string& message) const {
            if (isWarningEnabled()) {
                std::cout << "[WARNING] " << message << std::endl;
            }
        }

        void logError(const std::string& message) const {
            if (isErrorEnabled()) {
                std::cerr << "[ERROR] " << message << std::endl;
            }
        }

        // Template versions for complex messages
        template<typename T>
        void logDebug(const std::string& prefix, const T& value) const {
            if (isDebugEnabled()) {
                std::cout << "[DEBUG C++] " << prefix << value << std::endl;
            }
        }

        template<typename T>
        void logInfo(const std::string& prefix, const T& value) const {
            if (isInfoEnabled()) {
                std::cout << "[INFO] " << prefix << value << std::endl;
            }
        }

    private:
        DebugConfig() : m_debug_level(DebugLevel::WARNING) {} // Default to warnings only
        DebugLevel m_debug_level;
    };

    // Convenience macros for easier usage
    #define BRAINLL_DEBUG(...) brainll::DebugConfig::getInstance().logDebug(__VA_ARGS__)
    #define BRAINLL_VERBOSE(...) brainll::DebugConfig::getInstance().logVerbose(__VA_ARGS__)
    #define BRAINLL_INFO(...) brainll::DebugConfig::getInstance().logInfo(__VA_ARGS__)
    #define BRAINLL_WARNING(...) brainll::DebugConfig::getInstance().logWarning(__VA_ARGS__)
    #define BRAINLL_ERROR(...) brainll::DebugConfig::getInstance().logError(__VA_ARGS__)

} // namespace brainll

#endif // BRAINLL_DEBUG_CONFIG_HPP