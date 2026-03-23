# Changelog

All notable changes to VisProbe will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2024-03-23

### Changed
- **BREAKING**: Complete refactor to focus on compositional robustness testing
- **BREAKING**: Simplified API - now just `CompositionalExperiment` and `CompositionalResults`
- **BREAKING**: Removed complex search strategies (Binary, Bayesian, Adaptive)
- **BREAKING**: Removed property-based testing framework
- **BREAKING**: Removed CLI and dashboard
- Reduced codebase from 50+ files to 7 core files (~75% reduction)
- Simplified from ~10,000 lines to ~2,400 lines of clean code

### Added
- ✨ Automatic checkpointing - never lose progress on crashes
- ✨ Memory management - automatic model swapping between CPU/GPU
- ✨ Built-in AutoAttack integration with caching
- ✨ Fast APGD-CE mode (5x faster than standard AutoAttack)
- ✨ Offline analysis - load and analyze results without models/GPU
- ✨ Protection gap analysis
- ✨ Crossover detection
- ✨ Model disagreement analysis

### Improved
- 🚀 30x faster execution with optimizations
- 🚀 Skip attack when epsilon < 1e-10
- 🚀 Cached attack instances
- 🚀 Batch processing optimizations

### Removed
- ❌ Dashboard UI
- ❌ CLI interface
- ❌ Complex search strategies
- ❌ Property-based testing
- ❌ Preset system with JSON configs
- ❌ Complex strategy pattern with inheritance

## [1.0.0] - 2024-02-01

### Added
- Initial release with property-based testing framework
- Adaptive, Binary, and Bayesian search strategies
- Dashboard for visualization
- CLI for command-line usage
- Preset system for common test configurations