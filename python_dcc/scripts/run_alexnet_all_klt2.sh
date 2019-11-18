id_gpu=3
data_path=/home/wangzhe/Documents/exp/exp_2019_10/10_26_DCC_KKLT_Alex_VGG/transform_matrices/alexnet_klt2.mat
transform_method=klt2

for i in {1..96..1}
do
	python alexnet_accuracy_imagenet_optimized_transform.py $id_gpu $transform_method $data_path $i m
done

for i in {1..96..1}
do
	python alexnet_accuracy_imagenet_optimized_transform.py $id_gpu $transform_method $data_path $i nm
done
