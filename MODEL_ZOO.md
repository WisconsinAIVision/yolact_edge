## Model Zoo

We provide baseline YOLACT and YolactEdge models trained on COCO and YouTube VIS (our sub-training split, with COCO joint training).

To evalute the model, put the corresponding weights file in the `./weights` directory and run one of the following commands.

YouTube VIS models:

| Method | Backbone&nbsp; | mAP | AGX-Xavier FPS | RTX 2080 Ti FPS | weights |
|:-------------:|:-------------:|:----:|:----:|:----:|----------------------------------------------------------------------------------------------------------------------|
| YOLACT | R-50-FPN | 44.7 | 8.5 | 59.8 | [download](https://drive.google.com/file/d/1EfoQ0OteuQdY2yU9Od8XHTHrizQVFR2w/view?usp=sharing) \| [mirror](https://1drv.ms/u/s!AkSxI62eEcpbiHLweuem6riY6lVK?e=cUGBRf) |
| YolactEdge <br>(w/o TRT) | R-50-FPN | 44.2| 10.5 | 67.0 | [download](https://drive.google.com/file/d/1qvd4W28yzzXFb2wwGfYySv5HHzGU26XP/view?usp=sharing) \| [mirror](https://1drv.ms/u/s!AkSxI62eEcpbiHGgB-KrQLubo7eZ?e=h26XJM) |
| YolactEdge | R-50-FPN | 44.0| 32.4 | 177.6 | [download](https://drive.google.com/file/d/1qvd4W28yzzXFb2wwGfYySv5HHzGU26XP/view?usp=sharing) \| [mirror](https://1drv.ms/u/s!AkSxI62eEcpbiHGgB-KrQLubo7eZ?e=h26XJM) |
| YOLACT | R-101-FPN | 47.3 | 5.9 | 42.6 | [download](https://drive.google.com/file/d/1doS5MRhpSs4puVCuzR5i3GrDMSxcw7Lx/view?usp=sharing) \| [mirror](https://1drv.ms/u/s!AkSxI62eEcpbiHOei4kogT1JCfO7?e=dLcrVg) |
| YolactEdge <br>(w/o TRT) | R-101-FPN | 46.9| 9.5 | 61.2 | [download](https://drive.google.com/file/d/1mSxesVaMmYc13cPHiEnRvubPxy8WBjJW/view?usp=sharing) \| [mirror](https://1drv.ms/u/s!AkSxI62eEcpbiHAqrmvsL1RMH9WK?e=Tnlu7p) |
| YolactEdge | R-101-FPN | 46.2 | 30.8 | 172.7 | [download](https://drive.google.com/file/d/1mSxesVaMmYc13cPHiEnRvubPxy8WBjJW/view?usp=sharing) \| [mirror](https://1drv.ms/u/s!AkSxI62eEcpbiHAqrmvsL1RMH9WK?e=Tnlu7p) |

COCO models:

| Method | &nbsp;&nbsp;&nbsp;Backbone&nbsp;&nbsp;&nbsp;&nbsp; | mAP | Titan Xp FPS | AGX-Xavier FPS | RTX 2080 Ti FPS | weights |
|:-------------:|:-------------:|:----:|:----:|:----:|:----:|----------------------------------------------------------------------------------------------------------------------|
| YOLACT | MobileNet-V2 | 22.1 | - | 15.0 | 35.7 | [download](https://drive.google.com/file/d/1L4N4VcykqE-D5JUgWW9zBd6WKmZPBAZQ/view?usp=sharing) \| [mirror](https://1drv.ms/u/s!AkSxI62eEcpbiG8nFXtvgAkI-c1H?e=RraXLv) |
| YolactEdge | MobileNet-V2 | 20.8 | - | 35.7 | 161.4 | [download](https://drive.google.com/file/d/1L4N4VcykqE-D5JUgWW9zBd6WKmZPBAZQ/view?usp=sharing) \| [mirror](https://1drv.ms/u/s!AkSxI62eEcpbiG8nFXtvgAkI-c1H?e=RraXLv) |
| YOLACT | R-50-FPN | 28.2 | 42.5 | 9.1 | 45.0 | [download](https://drive.google.com/file/d/15TRS8MNNe3pmjilonRy9OSdJdCPl5DhN/view?usp=sharing) \| [mirror](https://1drv.ms/u/s!AkSxI62eEcpbiG5ZnhPTSkqBCURo?e=lNOaXr) |
| YolactEdge | R-50-FPN | 27.0| - | 30.7 | 140.3 | [download](https://drive.google.com/file/d/15TRS8MNNe3pmjilonRy9OSdJdCPl5DhN/view?usp=sharing) \| [mirror](https://1drv.ms/u/s!AkSxI62eEcpbiG5ZnhPTSkqBCURo?e=lNOaXr) |
| YOLACT | R-101-FPN | 29.8 | 33.5 | 6.6 | 36.5 | [download](https://drive.google.com/file/d/1EAzO-vRDZ2hupUJ4JFSUi40lAZ5Jo-Bp/view?usp=sharing) \| [mirror](https://1drv.ms/u/s!AkSxI62eEcpbiG8nFXtvgAkI-c1H?e=HyfH8Z) |
| YolactEdge | R-101-FPN | 29.5 | - | 27.3 | 124.8 | [download](https://drive.google.com/file/d/1EAzO-vRDZ2hupUJ4JFSUi40lAZ5Jo-Bp/view?usp=sharing) \| [mirror](https://1drv.ms/u/s!AkSxI62eEcpbiG8nFXtvgAkI-c1H?e=HyfH8Z) |