id_gpu=2
transform_method=klt1
data_path=/home/wangzhe/Documents/exp/exp_2019_10/10_26_DCC_KKLT_Alex_VGG/transform_matrices/alexnet_klt1.mat

for i in 0 1 2 3 4
do
	for j in {0..120..1}
	do
		python generate_RD_curves_alexnet_optimized_transforms.py $id_gpu $i $j $transform_method $data_path nm
	done
done

for i in 0 1 2 3 4
do
	for j in {0..120..1}
	do
		python generate_RD_curves_alexnet_optimized_transforms.py $id_gpu $i $j $transform_method $data_path m
	done
done
