id_gpu=0
transform_method=kkt1
data_path=/home/wangzhe/Documents/exp/exp_2019_10/10_26_DCC_KKLT_Alex_VGG/transform_matrices/alexnet_kkt1.mat

for i in 1 3 4
do
	for j in {0..120..1}
	do
		python generate_RD_curves_alexnet_given_matrix.py $id_gpu $i $j $transform_method $data_path
	done
done
