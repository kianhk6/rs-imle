# Teacher Resample Scheduling Guide

## 📋 Parameter Names (Clear and Intuitive!)

```bash
--every_n_epochs_resample "800,100,50"
--change_resample_every_n_epoch "800,2000,6000"
```

## 🎯 What Your Current Schedule Means

```bash
--every_n_epochs_resample "800,100,50"
--change_resample_every_n_epoch "800,2000,6000"
```

### Timeline Breakdown:

**Phase 1: Epoch 0 → 800**
- **Action**: Wait for first resample
- **Resample at**: Epoch 800 (first time)

**Phase 2: Epoch 800 → 2000**
- **Action**: Resample every 100 epochs
- **Resamples at**: 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000

**Phase 3: Epoch 2000 → 6000**
- **Action**: Continue every 100 epochs (no change from phase 2)
- **Resamples at**: 2100, 2200, 2300, ..., 5900, 6000

**Phase 4: Epoch 6000 → End**
- **Action**: Resample every 50 epochs (more frequent!)
- **Resamples at**: 6050, 6100, 6150, 6200, ..., until training ends

## 📊 How It Works

The two lists work together:

1. **`every_n_epochs_resample`**: HOW OFTEN to resample
2. **`change_resample_every_n_epoch`**: WHEN to change the frequency

They must be **equal length** and correspond 1-to-1:

```
Index 0: every_n_epochs_resample[0] = 800  ← Used until change_resample_every_n_epoch[0] = 800
Index 1: every_n_epochs_resample[1] = 100  ← Used from 800 until change_resample_every_n_epoch[1] = 2000
Index 2: every_n_epochs_resample[2] = 50   ← Used from 2000 until change_resample_every_n_epoch[2] = 6000
                                           ← After 6000, keeps using 50 forever
```

## 💡 More Examples

### Example 1: Simple Two-Phase
```bash
--every_n_epochs_resample "1000,200"
--change_resample_every_n_epoch "1000,5000"
```
- Epoch 0-1000: First resample at 1000
- Epoch 1000-5000: Every 200 epochs
- Epoch 5000+: Continue every 200 epochs

### Example 2: Aggressive Four-Phase
```bash
--every_n_epochs_resample "500,200,100,50"
--change_resample_every_n_epoch "500,2000,5000,8000"
```
- Epoch 0-500: First resample at 500
- Epoch 500-2000: Every 200 epochs
- Epoch 2000-5000: Every 100 epochs
- Epoch 5000-8000: Every 50 epochs (very frequent!)
- Epoch 8000+: Continue every 50 epochs

### Example 3: Conservative
```bash
--every_n_epochs_resample "2000,500,300"
--change_resample_every_n_epoch "2000,8000,12000"
```
- Epoch 0-2000: First resample at 2000 (late start)
- Epoch 2000-8000: Every 500 epochs (infrequent)
- Epoch 8000-12000: Every 300 epochs
- Epoch 12000+: Continue every 300 epochs

## 🚀 Quick Reference

| Parameter | Meaning | Example |
|-----------|---------|---------|
| `every_n_epochs_resample` | How often to resample in each phase | `"800,100,50"` |
| `change_resample_every_n_epoch` | When to switch to next phase | `"800,2000,6000"` |

**Key Points:**
- Lists must be **same length**
- First value in `every_n_epochs_resample` determines when first resample happens
- Last value continues until end of training
- More phases = more control over training dynamics

