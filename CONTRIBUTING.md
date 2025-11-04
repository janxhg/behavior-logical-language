# Contributing to BrainLL

**Last updated: 17/7/2025**
**Copyright (c) 2025 Joaquín Sturtz - NetechAI. All rights reserved.**

Thank you for your interest in contributing to BrainLL! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Types of Contributions

We welcome several types of contributions:

1. **Bug Reports**: Help us identify and fix issues
2. **Feature Requests**: Suggest new functionality
3. **Code Contributions**: Implement new features or fix bugs
4. **Documentation**: Improve or expand documentation
5. **Testing**: Add or improve test coverage
6. **Performance Optimization**: Improve system efficiency

## 🛠️ Development Environment Setup

### Prerequisites
```bash
# Required tools
- C++17 compiler (GCC 9+, MSVC 2019+, Clang 10+)
- CMake 3.15+
- Git
- Python 3.8+ (for tools)
```

### Initial Setup
```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/NetechAI/behavior-logical-language.git
cd behavior-logical-language

# 3. Set up upstream
git remote add upstream https://github.com/NetechAI/behavior-logical-language.git

# 4. Build the project
mkdir build && cd build
cmake ..
cmake --build . --config Release

# 5. Run basic tests
ctest
```

## 📋 Contribution Process

### 1. Preparation
```bash
# Update your fork
git checkout main
git pull upstream main
git push origin main

# Create a new branch
git checkout -b feature/my-new-feature
# or
git checkout -b fix/bug-correction
```

### 2. Development
- Follow the [code conventions](#code-conventions)
- Write tests for your code
- Update documentation if necessary
- Make small and descriptive commits

### 3. Testing
```bash
# Build and run tests
cd build
cmake --build . --config Release
ctest --verbose

# Run specific validations
./brainll_validator ../examples/correct_advanced_network_simplified.bll

# Verify weight analysis
python ../test_weight_diversity.py
```

### 4. Submission
```bash
# Push to your fork
git push origin feature/my-new-feature

# Open a Pull Request on GitHub
```

## 📝 Code Conventions

### C++
```cpp
// Naming style
class MyClass {          // PascalCase for classes
public:
    void myMethod();      // camelCase for methods
    int my_variable_;     // snake_case with trailing _ for members
};

// Constants
const int MAX_NEURONS = 1000;  // UPPER_CASE for constants

// Namespaces
namespace brainll {
namespace core {
    // code
}
}
```

### BrainLL Files
```brainll
# Descriptive comments
network my_network {
    # Clear and documented configuration
    global_config {
        simulation_time = 1000.0  # ms
        time_step = 0.1          # ms
    }
    
    # Descriptive names
    neuron_type excitatory_pyramidal {
        model = "integrate_and_fire"
        # parameters...
    }
}
```

### Python
```python
# Follow PEP 8
def analyze_weights(file_path: str) -> Dict[str, List[float]]:
    """Analyzes weight diversity in a file.
    
    Args:
        file_path: Path to the weights file
        
    Returns:
        Dictionary with weights by connection type
    """
    pass
```

## 🧪 Testing Guidelines

### Unit Tests
```cpp
// Use Google Test or similar
TEST(NeuronTest, IntegrateAndFire) {
    Neuron neuron(NeuronType::INTEGRATE_AND_FIRE);
    neuron.setThreshold(-55.0);
    
    EXPECT_EQ(neuron.getThreshold(), -55.0);
    EXPECT_FALSE(neuron.hasSpike());
}
```

### Integration Tests
```bash
# Validate complete examples
brainll_validator examples/correct_advanced_network_simplified.bll
./brainll examples/correct_advanced_network_simplified.bll --validate-only
```

### Performance Tests
```python
# Measure execution time
import time
start = time.time()
# ... code to measure ...
duration = time.time() - start
assert duration < 1.0  # maximum 1 second
```

## 📚 Documentación

### Code
- Document all public functions
- Use comments for complex logic
- Include usage examples

### Markdown Files
- Use descriptive titles
- Include code examples
- Add links to related documentation

### Examples
- Create minimal but functional examples
- Include explanatory comments
- Validate that they work correctly

## 🏷️ Commit Conventions

### Format
```
type(scope): brief description

More detailed description if necessary.

- Specific change 1
- Specific change 2

Fixes #123
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Format changes (don't affect functionality)
- `refactor`: Code refactoring
- `test`: Add or modify tests
- `chore`: Maintenance tasks

### Examples
```bash
git commit -m "feat(parser): add support for custom neuron models"
git commit -m "fix(weights): resolve uniform weight initialization issue"
git commit -m "docs(readme): update installation instructions"
```

## 🔍 Code Review

### For Reviewers
- Verify that code follows conventions
- Run tests locally
- Review updated documentation
- Provide constructive feedback

### For Contributors
- Respond to comments constructively
- Make requested changes in separate commits
- Keep discussion focused on the code

## 🎯 Priority Areas

### High Priority
- 🧪 **Automated tests**: Expand test coverage
- 📚 **User documentation**: Step-by-step guides
- 🐛 **Bug fixes**: Resolve existing issues
- 🔧 **Tools**: Improve validator and debugger

### Medium Priority
- 🚀 **Optimization**: Improve performance
- 🎨 **Examples**: More use cases
- 🔗 **Integration**: APIs for other languages
- 📊 **Visualization**: Graphical tools

### Low Priority
- 🎮 **GUI**: Graphical interface
- 🌐 **Web**: Web version of editor
- 📱 **Mobile**: Mobile support
- ☁️ **Cloud**: Cloud simulation

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/behavior-logical-language/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/behavior-logical-language/discussions)
- **Email**: maintainer@brainll.org

## 📄 License

By contributing to BrainLL, you agree that your contributions will be licensed under the [AGPL v3 License](LICENSE).

---

Thank you for contributing to BrainLL! 🧠💻