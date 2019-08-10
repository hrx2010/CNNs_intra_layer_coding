figure(2);
plot(sort(squeeze(sort(layers(3).Bias))))
hold on;
%xticks(1:6);
yticks(-0.1:0.05:0.1);
axis([1,384,-0.1,0.1]);
xlabel('Sort index');
ylabel('Sorted bias');
pdfprint('temp3.pdf','Width',21,'Height',12,'Position',[4,3,16.0,8]);
