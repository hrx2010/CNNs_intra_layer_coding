id_gpu=2
data_path=/home/wangzhe/Documents/exp/exp_2019_10/10_26_DCC_KKLT_Alex_VGG/transform_matrices/alexnet_klt1.mat
transform_method=klt1

for i in {1..96..1}
do
	python alexnet_accuracy_imagenet_given_matrix_given_slope.py $id_gpu $transform_method $data_path $i 
done
