# BrainLL Language Support for VS Code

This extension provides comprehensive language support for BrainLL (Brain Logic Language), including syntax highlighting, intelligent autocompletion, real-time validation, and debugging tools for neural network development.

## Features

### 🎨 Syntax Highlighting
- **Rich syntax highlighting** for all BrainLL constructs
- **Color-coded** regions, populations, connections, and parameters
- **Semantic highlighting** for neuron types and plasticity rules
- **Comment and string highlighting**

### 🚀 Intelligent Autocompletion
- **Context-aware suggestions** for neuron types, plasticity rules, and parameters
- **Smart snippets** for common BrainLL patterns
- **Population reference completion** with region prefixes
- **Parameter validation** and type checking

### ✅ Real-time Validation
- **Syntax error detection** as you type
- **Semantic validation** for population references
- **Configuration error checking**
- **Customizable validation rules**

### 📚 Documentation Support
- **Hover documentation** for keywords and parameters
- **Inline help** for neuron models and plasticity rules
- **Quick access** to BrainLL documentation

## Installation

### From VS Code Marketplace
1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X)
3. Search for "BrainLL Language Support"
4. Click Install

### Manual Installation
1. Download the `.vsix` file from releases
2. Open VS Code
3. Run `Extensions: Install from VSIX...` command
4. Select the downloaded file

## Configuration

### Extension Settings

This extension contributes the following settings:

- `brainll.validation.enabled`: Enable/disable real-time syntax validation
- `brainll.validation.strictMode`: Enable strict validation mode
- `brainll.autocompletion.enabled`: Enable/disable intelligent autocompletion
- `brainll.formatting.indentSize`: Number of spaces for indentation
- `brainll.validatorPath`: Path to brainll_validator executable

### Example Configuration

```json
{
  "brainll.validation.enabled": true,
  "brainll.validation.strictMode": false,
  "brainll.autocompletion.enabled": true,
  "brainll.formatting.indentSize": 2,
  "brainll.validatorPath": "/path/to/brainll_validator"
}
```

## Usage

### File Extensions
The extension automatically activates for files with these extensions:
- `.bll`
- `.brainll`

### Commands

- **BrainLL: Validate Syntax** (`Ctrl+Shift+V`): Manually validate current file
- **BrainLL: Format Document**: Format the current BrainLL document
- **BrainLL: Show Documentation**: Open BrainLL documentation

### Code Snippets

Type these prefixes and press Tab to insert code templates:

- `region` - Create a brain region
- `population` - Create a neural population
- `lif-pop` - LIF neuron population template
- `lstm-pop` - LSTM neuron population template
- `attention-pop` - Attention neuron population template
- `connection` - Create a neural connection
- `stdp-conn` - STDP plastic connection template
- `brain-region` - Complete brain region with E/I populations

### Example Usage

```brainll
// Type 'region' and press Tab
region cortex {
  // Type 'lif-pop' and press Tab
  population excitatory {
    size: 800
    neuron_type: LIF
    threshold: 1.0
    decay: 0.95
    refractory_period: 2
  }
  
  // Type 'stdp-conn' and press Tab
  connection exc_to_inh {
    from: cortex.excitatory
    to: cortex.inhibitory
    weight: 0.3
    probability: 0.2
    plasticity: STDP
    learning_rate: 0.01
  }
}
```

## Validation Features

### Error Detection
- **Syntax errors**: Missing brackets, invalid keywords
- **Reference errors**: Invalid population references
- **Type errors**: Incorrect parameter types
- **Configuration errors**: Missing required parameters

### Warning Detection
- **Unused populations**: Defined but not connected
- **Suspicious parameters**: Values outside typical ranges
- **Performance warnings**: Large networks without optimization

### Suggestions
- **Auto-correction**: Suggested fixes for common errors
- **Best practices**: Recommendations for better code
- **Optimization hints**: Performance improvement suggestions

## Requirements

### VS Code Version
- VS Code 1.74.0 or higher

### BrainLL Validator
For real-time validation, you need the `brainll_validator` executable:

1. **Automatic Detection**: The extension tries to find the validator in:
   - System PATH
   - Project `build/` directory
   - Project `bin/` directory

2. **Manual Configuration**: Set the path in settings:
   ```json
   {
     "brainll.validatorPath": "/path/to/brainll_validator"
   }
   ```

## Troubleshooting

### Validation Not Working
1. Check if `brainll_validator` is installed and accessible
2. Verify the `brainll.validatorPath` setting
3. Check the Output panel (View → Output → BrainLL) for error messages

### Autocompletion Not Working
1. Ensure `brainll.autocompletion.enabled` is true
2. Check if the file has the correct extension (`.bll` or `.brainll`)
3. Try restarting VS Code

### Syntax Highlighting Issues
1. Verify the file is recognized as BrainLL (check status bar)
2. Try changing language mode: Ctrl+K M → "BrainLL"
3. Reload the window: Ctrl+Shift+P → "Developer: Reload Window"

## Development

### Building from Source

```bash
# Clone the repository
git clone https://github.com/brainll-project/vscode-extension.git
cd vscode-extension

# Install dependencies
npm install

# Compile TypeScript
npm run compile

# Package extension
vsce package
```

### Testing

```bash
# Run tests
npm test

# Launch extension development host
F5 (in VS Code)
```

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Areas for Contribution
- Additional code snippets
- Improved syntax highlighting
- Better error messages
- Performance optimizations
- Documentation improvements

## Changelog

### 1.0.0
- Initial release
- Syntax highlighting for BrainLL
- Intelligent autocompletion
- Real-time validation
- Code snippets and templates
- Hover documentation

## License

AGPL v3 License - see [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [BrainLL Docs](https://github.com/brainll-project/docs)
- **Issues**: [GitHub Issues](https://github.com/brainll-project/vscode-extension/issues)
- **Discussions**: [GitHub Discussions](https://github.com/brainll-project/vscode-extension/discussions)

---

**Enjoy coding neural networks with BrainLL! 🧠⚡**