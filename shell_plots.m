function shell_plots(truth, test, gap, sub, nc)

%% Display Result for first 4 coils
figure(100);clf;

% subsample spokes for display
idx = 1:sub:size(truth, 2);
h = tiledlayout(gap, 2*nc, 'TileSpacing', 'compact', 'Padding', 'none');
for i = 1:gap
    maxMag = max(abs(truth(gap+1-i,:,1:nc)),[],'all');
    for j = 1:nc
        % Magnitude
        nexttile;
        hold on;
        scatter(idx,real(truth(gap+1-i,idx,j)), 5, 'filled');
        scatter(idx,real(test(gap+1-i,idx,j)), 5, 'filled');
        xlim([0 max(idx)]);
        ylim([-maxMag maxMag]);
        if j==1
            ylabel(sprintf('k_{%d}',gap-i));
        else
        set(gca,'yticklabel',[])
        end
        if i==1
            title(sprintf('Channel %d Real',j));
        end
        if i==gap
             xlabel('Spoke Number');
         end
        % Phase
        nexttile;
        hold on;
        scatter(idx,imag(truth(gap+1-i,idx,j)), 5, 'filled');
        scatter(idx,imag(test(gap+1-i,idx,j)), 5, 'filled');
        xlim([0 max(idx)]);
        set(gca,'yticklabel',{[]})
%         ylim([-pi pi]);
        ylim([-maxMag maxMag]);
        if i==1
            title(sprintf('Channel %d Imag',j));
        end
        if i==gap
            xlabel('Spoke Number');
        end
    end
end