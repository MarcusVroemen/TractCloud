%addpath('CPM/matlab/')
T = readtable('HCP_behavioral_data.csv');
y = T.('PMAT24_A_CR');
y = T.('PicVocab_Unadj');


RM = randi([0, 1000000], 84*84, 1206);


x = readtable('combined_connectomes.csv');
x = table2array(x(1:end, 2:end)).';
T = readtable('HCP_behavioral_data_subj.csv');
y = T.('PMAT24_A_CR');
y = T.('PicVocab_Unadj');

[y_predict, performance] = cpm_main(x, y, 'pthresh', 0.5, 'kfolds', 10);
%[y_predict, performance] = cpm_main(x, y, 'kfolds', 10);
