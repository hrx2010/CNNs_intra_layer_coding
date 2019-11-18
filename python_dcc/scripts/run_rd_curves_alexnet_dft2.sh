id_gpu=2


for i in 0 1 2 3 4 
do
	for j in {0..120..1}
	do
		python generate_RD_curves_alexnet_fixed_transforms.py $id_gpu $i $j dft2 nm
	done
done
