## OOD Robustness of LRN
This is the implementation that tests the hyperbolic ResNet architecture using LRN against out of distribution daasets. The frameworks was adapted from the 2023 ICCV paper [*Poincare ResNet*](https://github.com/maxvanspengler/poincare-resnet). 

The main implementation is in the folder **LRN**. We provide an example of running our code in the **example** folder. The root directory must contains a *config.ini* file that points to the paths of the required datasets, similar to the one in the **example** folder. To run our code, please run
```
bash example/example.sh
```
