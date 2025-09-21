# Heterogeneous Graph Learning for 3D Scene Graph Prediction in Point Clouds

This is a release of the code of our paper **_Heterogeneous Graph Learning for 3D Scene Graph Prediction in Point Clouds_** (ECCV2024)


# Dependencies
```bash
    - Python 3.6
    - Pytorch 1.5.0
    - CUDA 10.1
    - pointnet2_ops
    - tqdm
    - vtkplotter
    - numpy
    - importlib
```

# Prepare the data
A. Download 3Rscan and 3DSSG_Subset Annotation, you can follow [3DSSG](https://github.com/ShunChengWu/3DSSG#preparation).

B. Preprocess dataset. (Modify the 3DSSG dataset path to your local path first)
```bash
cd data/
python data_prepro.py
```

# Prepare the models
Pretrained PointNet for object and predicate classification: 
log/obj_classification(pred_classification).
```bash
python train_obj_class.py
python train_pred_class.py
```
Pretrained meta-embedding: data/meta_embedding.
```bash
python train_GAE_meta_embedding.py
```
[Obj_classification pth](https://drive.google.com/file/d/1qiE6oujdynejInkun7hSPNveWALg6UzE/view?usp=sharing).

[Pred_classification pth](https://drive.google.com/file/d/1ru_8JacDNy_cPWBNuF5fOyMR0GXPWWpD/view?usp=sharing).

# Run Code
## PredCls:
```bash
- Train HGSL:
    python train_heterG_predcls_typeW.py
- Generate the type edge weights and save them (or just use the saved Edge_weights in HGR stage):
    python genEW_heterG_predcls_results.py
- Train HGR:
    python train_heterG_predcls_newEdge.py
- Eval:
    python eval_train_heterG_predcls_newEdge.py
```
[Checkpoint](https://drive.google.com/file/d/10ozdRqCL84dlncQb95Kxv0CkE5VpJ62t/view?usp=sharing) for Eval. 

[Edge_weights](https://drive.google.com/drive/folders/1RNT_g9i-hB1ltsZJuInAuNg1ZpswCkx3?usp=sharing) saved by us. 

## SGCls: 
```bash
- We didn’t use the type edge weights in SGCls task.
- Train:
    python train_heterG_sgcls_newEdge.py
- Eval:
    python eval_train_heterG_sgcls_newEdge.py
```
[Checkpoint](https://drive.google.com/file/d/1yGpw8zCCk_Ui10oKAP-nRFqzmqR8PSwi/view?usp=sharing) for Eval.

# Paper

If you find the code useful please consider citing our [paper](https://fq.pkwyx.com/default/https/www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03785.pdf):

```
@inproceedings{ma2024heterogeneous,
  title={Heterogeneous graph learning for scene graph prediction in 3d point clouds},
  author={Ma, Yanni and Liu, Hao and Pei, Yun and Guo, Yulan},
  booktitle={European Conference on Computer Vision},
  pages={274--291},
  year={2024},
  organization={Springer}
}
```

# Acknowledgement
This project is partly built upon **_Knowledge-inspired 3D Scene Graph Prediction in Point Cloud_** ([KISGP](https://openreview.net/attachment?id=OLyhLK2eQP&name=code)).


## Contact

If you have any questions, feel free to open an issue or contact us at mayn3@mail2.sysu.edu.cn. 

