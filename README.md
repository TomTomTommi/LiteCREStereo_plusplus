<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;">
Uncertainty Guided Adaptive Warping <br> for Robust and Efficient Stereo Matching<h1>  

<div align="center">
  <a href="https://arxiv.org/abs/2307.14071" target="_blank" rel="external nofollow noopener">
  <img src="https://img.shields.io/badge/Paper-CREStereo++-red" alt="Paper arXiv"></a>

</div>
</p>


![teaser](./assets/CREStereo++.png)


## Checkpoint
We provide the checkpoint pretrained on SceneFlow from
 [google drive](https://drive.google.com/drive/folders/1mHxjzvBbsMoSDHQcg4QUNnSTOspgwaqT?usp=sharing) . 
Then place them in: `./checkpoints/`


## Benchmark Results 
To reproduce the generalization benchmark results, run:
```
sh evaluate.sh
```

## Training
To train the model from scratch, run:
```
sh train.sh
```


## MACs
To compute the model complexity (MACs), use:
```
python flops_count.py
```

## Citation 
If you find this work useful, please consider citing:
```
@inproceedings{jing2023uncertainty,
  title={Uncertainty guided adaptive warping for robust and efficient stereo matching},
  author={Jing, Junpeng and Li, Jiankun and Xiong, Pengfei and Liu, Jiangyu and Liu, Shuaicheng and Guo, Yichen and Deng, Xin and Xu, Mai and Jiang, Lai and Sigal, Leonid},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3318--3327},
  year={2023}
}
```
