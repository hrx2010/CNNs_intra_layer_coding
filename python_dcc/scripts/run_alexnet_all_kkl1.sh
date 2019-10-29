id_gpu=2
data_path=/home/wangzhe/Documents/exp/exp_2019_10/10_26_DCC_KKLT_Alex_VGG/transform_matrices/alexnet_kkt1.mat
transform_method=kkt1

for i in {1..80..1}
do
	python alexnet_accuracy_imagenet_given_matrix.py $id_gpu $i $transform_method $data_path
done
