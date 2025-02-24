# Knowledge-inspired 3D Scene Graph Prediction in Point Cloud (Sample Code)

# Dependency
    - Python 3.6
    - Pytorch 1.5.0
    - CUDA 10.1
    - pointnet2_ops
    - tqdm
    - vtkplotter
    - numpy
    - importlib

# Instruction
    - Modify the 3DSSG dataset path to your local path.
    - Pretrained PointNet for object classification: log/obj_classification.
    - Pretrained PointNet for predicate classification: log/pred_classification.
    - Pretrained meta-embedding: data/meta_embedding.
    - RUN train the meta-embedding:

```python
    python train_GAE_meta_embedding.py
```
    - RUN train knowledge-inspired 3D scene graph prediction model for SGCls task:
```python
    python train_GNN_perfusion.py
```
    - RUN train knowledge-inspired 3D scene graph prediction model for PredCls task:
```python
    python train_GNN_perfusion_predcls.py
```


# 复制于 .../myn/2022-new/know-3DSSG
