#!/bin/bash
# Quick validation script for GAE implementation

echo "=== GAE Implementation Validation ==="
echo ""

# 1. Check for old optimizer references
echo "1. Checking for old optimizer references..."
if grep -n "self.optimizer" policy_gradient.py | grep -v "policy_optimizer\|critic_optimizer"; then
    echo "   ❌ FAILED: Found old optimizer references"
    exit 1
else
    echo "   ✅ PASSED: No old optimizer references"
fi

# 2. Check for GAE implementation
echo ""
echo "2. Checking for GAE implementation..."
if grep -q "gae_lambda" policy_gradient.py && grep -q "Generalized Advantage Estimation" policy_gradient.py; then
    echo "   ✅ PASSED: GAE implementation found"
else
    echo "   ❌ FAILED: GAE implementation not found"
    exit 1
fi

# 3. Check for gradient clipping
echo ""
echo "3. Checking for gradient clipping..."
if grep -q "clip_grad_norm_" policy_gradient.py; then
    echo "   ✅ PASSED: Gradient clipping found"
else
    echo "   ❌ FAILED: Gradient clipping not found"
    exit 1
fi

# 4. Check for advantage normalization
echo ""
echo "4. Checking for advantage normalization..."
if grep -q "advantages_normalized" policy_gradient.py; then
    echo "   ✅ PASSED: Advantage normalization found"
else
    echo "   ❌ FAILED: Advantage normalization not found"
    exit 1
fi

# 5. Check for Huber loss
echo ""
echo "5. Checking for Huber loss..."
if grep -q "smooth_l1_loss" policy_gradient.py; then
    echo "   ✅ PASSED: Huber loss found"
else
    echo "   ❌ FAILED: Huber loss not found"
    exit 1
fi

# 6. Check for separate optimizers
echo ""
echo "6. Checking for separate optimizers..."
if grep -q "policy_optimizer" policy_gradient.py && grep -q "critic_optimizer" policy_gradient.py; then
    echo "   ✅ PASSED: Separate optimizers found"
else
    echo "   ❌ FAILED: Separate optimizers not found"
    exit 1
fi

# 7. Check for enhanced logging
echo ""
echo "7. Checking for enhanced logging..."
if grep -q "GAE Stats" policy_gradient.py; then
    echo "   ✅ PASSED: Enhanced logging found"
else
    echo "   ❌ FAILED: Enhanced logging not found"
    exit 1
fi

# 8. Check gamma value
echo ""
echo "8. Checking gamma value..."
if grep -q "self.gamma = 0.95" policy_gradient.py; then
    echo "   ✅ PASSED: Gamma set to 0.95"
else
    echo "   ⚠️  WARNING: Gamma might not be 0.95"
fi

echo ""
echo "=== All Validation Checks Passed! ==="
echo ""
echo "Next steps:"
echo "1. Run training with: python policy_gradient.py [args]"
echo "2. Monitor these metrics:"
echo "   - Critic Loss should be < 5"
echo "   - Normalized Adv Mean should be ≈ 0"
echo "   - Normalized Adv Std should be ≈ 1"
echo "3. Check GAE_IMPLEMENTATION_SUMMARY.md for details"
