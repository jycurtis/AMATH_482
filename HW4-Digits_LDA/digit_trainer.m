function [thresh,w,sortV,order] = digit_trainer(digits)
    
    count = length(digits);
    means = cell(count,1);
    for i = 1:count
        means{i} = mean(digits{i},2);
    end
    overallMean = mean(cat(2,means{:}),2);

    Sw = 0;
    Sb = 0;
    for i = 1:count
        d = digits{i};
        for j = 1:size(d,2)
            Sw = Sw + (means{i}-d(:,j))*(means{i}-d(:,j))';
        end
        Sb = Sb + (means{i}-overallMean)*(means{i}-overallMean)';
    end

    [V2, D] = eig(Sb,Sw);
    [lambda, ind] = max(abs(diag(D)));
    w = V2(:,ind);
    w = w/norm(w,2);
    v = cell(count,1);
    sorts = cell(count,1);
    temp = zeros(count,1);
    for i = 1:count
        v{i} = w'*digits{i};
        sorts{i} = sort(v{i});
        temp(i) = mean(v{i});
    end
    [sortMeans,order] = sort(temp);
    sortV = sorts(order);

    thresh = cell(count-1,1);
    for i = 1:count-1
        t1 = length(sortV{i});
        t2 = 1;
        while sortV{i}(t1) > sortV{i+1}(t2)
            t1 = t1 - 1;
            t2 = t2 + 1;
        end
        thresh{i} = (sortV{i}(t1) + sortV{i+1}(t2))/2;
    end
    