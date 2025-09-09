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
| 08 | `conv_layer.py` | ✅ Completed | A- (88%) | Understanding convolutional layers - Good implementation with proper formula, comprehensive testing, and visualization. Clean up debug prints. |

### ⚡ Optimization (3 drills)
Master training loops, loss functions, and optimization techniques.

| Drill | Name | Completed | Grade | Notes |
|-------|------|-----------|-------|-------|
| 09 | `loss_functions.py` | ✅ Completed | A- (90%) | Implement MSE, BCE, CrossEntropy - All functions work correctly, minor style improvements needed |
| 10 | `gradient_descent.py` | ✅ Completed | A- (91%) | Manual vs automatic differentiation - Excellent implementation, added bonus Adam optimizer, minor Adam convergence tuning needed |
| 11 | `training_loop.py` | ✅ Completed | A- (89%) | Complete training with early stopping - Good implementation with proper training/validation loops, early stopping, and visualization. Minor fixes: f-string bug in print statement, overly complex network for simple linear regression |

### 🐛 Debugging (3 drills)
Fix broken code and learn common PyTorch pitfalls.

| Drill | Name | Completed | Grade | Notes |
|-------|------|-----------|-------|-------|
| 12 | `buggy_network.py` | ✅ Completed | A- (88%) | Fix 5+ bugs in XOR network - Excellent debugging skills, identified all bugs correctly and provided clean fixed implementation with proper BCE loss, sigmoid activation, and Adam optimizer. Clean code with good comments. |
| 13 | `gradient_issues.py` | ✅ Completed | A- (91%) | Resolve gradient-related problems - Excellent identification and fixing of all gradient issues: poor initialization, vanishing gradients, incorrect gradient accumulation, gradient clipping, and activation scaling. Clean implementation with proper Xavier initialization, ReLU activations, and stable training loop. Minor improvement: Input doesn't need requires_grad for typical supervised learning |
| 14 | `memory_leaks.py` | ✅ Completed | A- (90%) | Fix memory and efficiency issues - Excellent identification and fixing of all memory leaks: removed unnecessary intermediate storage, used torch.no_grad() for inference, proper device management, and efficient gradient handling. Clean implementation with good architecture. Minor: redundant model.to() call and overly pessimistic performance comment. |

### 🔗 Fill in the Blanks (4 drills)
Complete partially implemented advanced concepts.

| Drill | Name | Completed | Grade | Notes |
|-------|------|-----------|-------|-------|
| 15 | `attention_mechanism.py` | ✅ Completed | A (93%) | Complete attention implementation - Excellent understanding of attention mechanism with clear explanatory comments. All components implemented correctly with proper dot-product attention. Code runs perfectly and demonstrates deep understanding of query expansion, attention scoring, and context computation. |
| 16 | `batch_norm.py` | ✅ Completed | A- (91%) | Implement batch normalization - Excellent understanding of batch norm theory and implementation. Perfect mathematical implementation with biased/unbiased variance handling, proper running stats updates, and learnable parameters. Code runs flawlessly and matches PyTorch exactly. Good explanatory comments throughout. Minor: Could add more detailed docstrings for the mathematical operations. |
| 17 | `rnn_cell.py` | ✅ Completed | A- (90%) | Complete RNN cell from scratch - Excellent implementation with correct RNN formula, proper weight matrices, and perfect sequence processing. All tensor dimensions match PyTorch exactly. Clean up debug print statement. |
| 18 | `autograd_function.py` | ✅ Completed | A- (92%) | Custom autograd function - Perfect implementation with correct forward/backward passes, proper gradient computation, and comprehensive testing. All gradients match PyTorch exactly for both vector and matrix inputs. |

## Progress Summary
- **Total Drills**: 18
- **Completed**: 18
- **In Progress**: 0
- **Not Started**: 0
- **Overall Progress**: 100%

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
