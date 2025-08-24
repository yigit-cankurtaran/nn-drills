# PyTorch Drills - Progress Tracker

## Overview
This project contains 18 PyTorch programming drills designed for intermediate-level programmers. Each drill focuses on different aspects of PyTorch and neural network implementation.

## Environment Setup
- **Virtual Environment**: `torch-env` (created with uv)
- **Activation**: `source torch-env/bin/activate`
- **Dependencies**: torch, torchvision, torchaudio, numpy, matplotlib, seaborn, scikit-learn, pandas, jupyter

## Drill Categories

### 📊 Basics (4 drills)
Test your fundamental understanding of PyTorch tensors and operations.

| Drill | Name | Completed | Grade | Notes |
|-------|------|-----------|-------|-------|
| 01 | `tensor_creation.py` | ✅ Completed | B+ (87%) | Create tensors, basic operations - Good implementation, remove debug prints |
| 02 | `tensor_indexing.py` | ✅ Completed | A- (91%) | Indexing, slicing, extraction - Excellent understanding, clean up debug code |
| 03 | `tensor_reshaping.py` | ✅ Completed | B+ (87%) | Reshape, transpose, dimension manipulation - Clean implementation, good defensive programming |
| 04 | `tensor_math.py` | ✅ Completed | A- (92%) | Mathematical operations, broadcasting - Good understanding of tensor ops, clean broadcasting |

### 🧠 Neural Networks (4 drills)
Build neural network components from scratch and understand their inner workings.

| Drill | Name | Completed | Grade | Notes |
|-------|------|-----------|-------|-------|
| 05 | `linear_layer.py` | ✅ Completed | A (93%) | Implement linear layer from scratch - Excellent implementation with proper Xavier initialization |
| 06 | `activation_functions.py` | ✅ Completed | A+ (96%) | Implement common activation functions - Perfect implementations, all match PyTorch exactly |
| 07 | `simple_mlp.py` | ✅ Completed | B (83%) | Build MLP for XOR problem - Works but architecture is hard-coded, needs dynamic layer building |
| 08 | `conv_layer.py` | ⭕ Not Started | - | Understanding convolutional layers |

### ⚡ Optimization (3 drills)
Master training loops, loss functions, and optimization techniques.

| Drill | Name | Completed | Grade | Notes |
|-------|------|-----------|-------|-------|
| 09 | `loss_functions.py` | ✅ Completed | A- (90%) | Implement MSE, BCE, CrossEntropy - All functions work correctly, minor style improvements needed |
| 10 | `gradient_descent.py` | ⭕ Not Started | - | Manual vs automatic differentiation |
| 11 | `training_loop.py` | ⭕ Not Started | - | Complete training with early stopping |

### 🐛 Debugging (3 drills)
Fix broken code and learn common PyTorch pitfalls.

| Drill | Name | Completed | Grade | Notes |
|-------|------|-----------|-------|-------|
| 12 | `buggy_network.py` | ⭕ Not Started | - | Fix 5+ bugs in XOR network |
| 13 | `gradient_issues.py` | ⭕ Not Started | - | Resolve gradient-related problems |
| 14 | `memory_leaks.py` | ⭕ Not Started | - | Fix memory and efficiency issues |

### 🔗 Fill in the Blanks (4 drills)
Complete partially implemented advanced concepts.

| Drill | Name | Completed | Grade | Notes |
|-------|------|-----------|-------|-------|
| 15 | `attention_mechanism.py` | ⭕ Not Started | - | Complete attention implementation |
| 16 | `batch_norm.py` | ⭕ Not Started | - | Implement batch normalization |
| 17 | `rnn_cell.py` | ⭕ Not Started | - | Complete RNN cell from scratch |
| 18 | `autograd_function.py` | ⭕ Not Started | - | Custom autograd function |

## Progress Summary
- **Total Drills**: 18
- **Completed**: 8
- **In Progress**: 0
- **Not Started**: 10
- **Overall Progress**: 44%

## Grading Scale
- **A+** (95-100%): Excellent implementation, optimal efficiency, great code style
- **A** (90-94%): Very good implementation, minor improvements possible
- **B+** (85-89%): Good implementation, some optimizations needed
- **B** (80-84%): Solid implementation, works correctly
- **C+** (75-79%): Acceptable implementation, has room for improvement
- **C** (70-74%): Basic implementation, meets minimum requirements
- **D** (60-69%): Below average, significant issues
- **F** (<60%): Does not work or major conceptual errors

## Tips for Success
1. **Read the docstrings carefully** - they contain important hints and expected outputs
2. **Test your implementations** - compare with PyTorch's built-in functions when possible
3. **Understand the math** - don't just copy formulas, understand why they work
4. **Debug systematically** - use print statements and tensor shapes to debug
5. **Optimize gradually** - get it working first, then make it efficient

## Next Steps
1. Start with the basics category to build confidence
2. Move through categories systematically
3. Come back to harder drills after building foundational knowledge
4. Update this file as you complete each drill

Good luck with your PyTorch journey! 🚀
