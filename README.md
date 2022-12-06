# CardiacLab
* deepeidt
```shell
monailabel start_server --app radiology --studies "D:\dataset\chgh\CardiacLab\exp_imgs" --conf models deepedit
```
* segmentation
```shell
monailabel start_server --app radiology --studies "D:\dataset\chgh\CardiacLab\exp_imgs" --conf models segmentation_cardiac --conf use_pretrained_model true
```
