id_gpu=0

for i in {1..96..1}
do
	python alexnet_accuracy_imagenet_fixed_transform.py $id_gpu dst2 $i m
done

for i in {1..96..1}
do
	python alexnet_accuracy_imagenet_fixed_transform.py $id_gpu dst2 $i nm
done
