# VisProbe Documentation

Welcome to the VisProbe documentation! VisProbe is a property-based robustness testing framework for vision models that helps you find where your models fail under various perturbations.

## 📚 Documentation Overview

- **[Quickstart Guide](quickstart.md)** - Get up and running in 5 minutes
- **[Installation](installation.md)** - Detailed installation instructions
- **[User Guide](user-guide.md)** - Complete guide to using VisProbe
- **[API Reference](api-reference.md)** - Detailed API documentation
- **[Architecture](architecture.md)** - Understanding VisProbe's design
- **[Examples](../examples/)** - Code examples and tutorials

## 🚀 Quick Links

### For Users
- [Basic Usage Examples](../examples/basic_example.py)
- [Working with Presets](../examples/preset_comparison.py)
- [Custom Model Integration](../examples/custom_model_example.py)

### For Developers
- [Contributing Guide](../CONTRIBUTING.md)
- [Code Architecture](architecture.md)
- [Adding New Perturbations](extending.md)

## 🎯 What is VisProbe?

VisProbe is a framework for testing the robustness of vision models through:

1. **Property-based testing**: Define what properties your model should maintain
2. **Adaptive search**: Automatically find the breaking points
3. **Comprehensive perturbations**: Test against various real-world distortions
4. **Easy integration**: Works with any PyTorch model

## 📖 How to Use This Documentation

### New Users
Start with the [Quickstart Guide](quickstart.md) to understand the basics, then explore the [User Guide](user-guide.md) for more advanced features.

### Experienced Users
Jump to the [API Reference](api-reference.md) for detailed function documentation or check the [Examples](../examples/) for specific use cases.

### Contributors
Read the [Architecture](architecture.md) document to understand the codebase structure and the [Contributing Guide](../CONTRIBUTING.md) for development guidelines.

## 💡 Key Concepts

- **Perturbation**: A transformation applied to an input (e.g., blur, noise)
- **Strategy**: An implementation of a perturbation with specific parameters
- **Property**: A test condition that should hold (e.g., "prediction unchanged")
- **Threshold**: The perturbation level where the property fails
- **Preset**: A curated collection of perturbations for testing

## 🆘 Getting Help

- **Issues**: [GitHub Issues](https://github.com/bilgedemirkaya/VisProbe/issues)
- **Discussions**: [GitHub Discussions](https://github.com/bilgedemirkaya/VisProbe/discussions)
- **Examples**: Check the [examples/](../examples/) directory

## 📄 License

VisProbe is released under the [MIT License](../LICENSE).