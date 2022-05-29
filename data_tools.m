mat2csv()

function mat2csv()
rng(63)
assignments = [randperm(63, 63)]';

% train data
labels_full = zeros(53*60, 1);
sample_nums = [randperm(53*60, 53*60)]';
for i = 1:53
    i
    [ecg, labels] = load_patient(assignments(i));
    mult = 30000;
    for k = 1:60
        ecg_minute = ecg((k-1)*mult+1:k*mult, :);
        writematrix(ecg_minute, append('aws_bucket/data/sample', string(sample_nums(60*(i-1)+k)-1 + 1894), '.csv'))
        labels_full(sample_nums(60*(i-1)+k)) = labels(k);
    end
end
writematrix(labels_full, 'aws_bucket/data/labels2.csv')

% test data
labels_full = zeros(10*60, 1);
sample_nums = [randperm(10*60, 10*60)]';
for i = 1:10
    i
    [ecg, labels] = load_patient(assignments(i+53));
    mult = 30000;
    for k = 1:60
        ecg_minute = ecg((k-1)*mult+1:k*mult, :);
        writematrix(ecg_minute, append('aws_bucket/test/sample', string(sample_nums(60*(i-1)+k)-1), '.csv'))
        labels_full(sample_nums(60*(i-1)+k)) = labels(k);
    end
end
writematrix(labels_full, 'aws_bucket/test/labels.csv')
end

function [ecg, labels] = load_patient(num)
data = load(append('aws_bucket/matlab_data/patient', string(num), '.mat')).CaseData;
ecg = timetable2table(data.ECG, "ConvertRowTimes", false);
ecg = ecg{:,:};
labels = convert_ann(data);
end

function labels = convert_ann(data)
ann = data.ann;
anntype = data.anntype;
afib = strcmp(anntype, '(AFIB');
normal = strcmp(anntype, '(N'); % add all other anntypes to exclude here
% populate missing anntype entries
anntype = zeros(size(anntype));
for i = 1:length(anntype)
    if afib(i) == 1
        anntype(i) = 1;
    elseif normal(i) == 1
        anntype(i) = 0;
    else
        if i-1 == 0
            anntype(i) = 0;
        else
            anntype(i) = anntype(i - 1);
        end
    end
end

% expand to fullsize vector of anntype
labels_full = zeros(1.8e6, 1);
for i = 1:length(ann)-1
    labels_full(ann(i):ann(i+1)) = anntype(i);
end
labels_full(ann(end):end) = anntype(end);
labels_full(1:ann(1)) = anntype(1);

% collapse to minute-chunk labels vector
labels = zeros(60, 1);
mult = 30000;
for i = 1:length(labels)
    if sum(labels_full((i-1)*mult+1:i*mult)) / mult > 0.5
        labels(i) = 1;
    end
end
end
