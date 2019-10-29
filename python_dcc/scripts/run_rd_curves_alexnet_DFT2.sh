id_gpu=2

for i in 2 
do
	for j in {0..120..1}
	do
		python generate_RD_curves_alexnet.py $id_gpu $i $j
	done
done
