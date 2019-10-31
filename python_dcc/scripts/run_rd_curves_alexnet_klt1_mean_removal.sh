id_gpu=0
transform_method=klt1
data_path=./transform_matrices/alexnet_klt1.mat

for i in 0 1 2 3 4 
do
	for j in {0..120..1}
	do
		python generate_RD_curves_alexnet_given_matrix.py $id_gpu $i $j $transform_method $data_path
	done
done
