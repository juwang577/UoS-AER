% =================================================================
% ELE302 Machine Learning Coursework Assignment
% Author: King Fai Wang | University of Sheffield | 2026
%
% REQUIRED TOOLBOXES:
%   - Statistics and Machine Learning Toolbox
%     (required for 'fitrensemble' function in Task 3)
%
% USAGE: Run this file with 'household_energy_data.csv'
%        in the same directory. 

% ---------------IMPORTANT INFORMATION: --------------------------
% The entire code may take several minutes to run, especially 
% the task 3 part. Please be patient.
% =================================================================

clear; close all; clc;

%% Load Data and Explore
% ----- Task 0: Read table ------------------------
T = readtable("household_energy_data.csv");

fprintf('Dataset loaded: %d rows, %d columns\n\n', height(T), width(T)); % diaplay table size

% Figure 1 — Distribution of target variable
figure(1);
histogram(T.EnergyRequestedFromGrid_kW_);
title('Figure 1:Distribution of Energy Requested from Grid');
xlabel('kW'); ylabel('Frequency');

%% Task 1 - Dataset Preprocessing
% ------ Task 1.1: Dealing with WeatherIcon --------------------------------------
% Categorical weather classifications can not demonstrate actual power demand. 
% One-hot encoding no longer requested for WeatherIcon.

categories = unique(T.WeatherIcon);

fprintf('=====Unique weather categories======\n');
for i = 1:length(categories) % this should list 9 unique weather categoreis
    fprintf('  %d. %s\n', i, categories{i});
end
fprintf('====================================\n\n');

T.WeatherIcon = []; % remove WeatherIcon

% identify target and feature columns  (seperate target and feature columns)
Cols = T.Properties.VariableNames;
TargetCol = "EnergyRequestedFromGrid_kW_";
FeatureCols = Cols(~strcmp(Cols, TargetCol)); % all columns except target columns

fprintf('=========Target and Feature Counts========\n');
fprintf('Remaining features: %d\n', length(TargetCol)); % counts how many times target columns 
fprintf('Remaining features: %d\n', length(FeatureCols)); % counts how many times features columns
fprintf('==========================================\n\n');



% ------ Task 1.2: Replace NaN and Inf with column median ------------
TargetNaN = 0; % count from zero
TargetInf = 0; % count from zero

% Target Part:
for i = 1:length(TargetCol) % loop though each column
    Col_1 = TargetCol;
    ColData = T.(Col_1);

    nanCount = sum(isnan(ColData));
    infCount = sum(isinf(ColData));
    TargetNaN = TargetNaN + nanCount; % count how many NaN
    TargetInf = TargetInf + infCount; % count how many Inf

    colMedian = median(ColData(isfinite(ColData)));
    ColData(isnan(ColData)) = colMedian; % replace NaN with median
    ColData(isinf(ColData)) = colMedian; % replace Inf with median
    T.(Col_1) = ColData; % rewrite new values back to table
end

fprintf('==========Missing data (target)==========\n');
fprintf('NaN values replaced with median: %d\n', TargetNaN);
fprintf('Inf values replaced with median: %d\n', TargetInf);
fprintf('==========================================\n\n');

% Features Part:
FeatureNaN = 0; % counts from zero
FeatureInf = 0; % counts from zero

for i = 1:length(FeatureCols) % loop though each column
    col_2 = FeatureCols{i};
    ColData = T.(col_2);

    nanCount = sum(isnan(ColData));
    infCount = sum(isinf(ColData));
    FeatureNaN = FeatureNaN + nanCount; % count how many NaN
    FeatureInf = FeatureInf + infCount; % count how many Inf

    colMedian = median(ColData(isfinite(ColData)));
    ColData(isnan(ColData)) = colMedian; % replace NaN with median
    ColData(isinf(ColData)) = colMedian; % replace Inf with median
    T.(col_2) = ColData; % rewrite new values back to table
end

fprintf('========== Missing data (feature) ==========\n');
fprintf('NaN values replaced with median: %d\n', FeatureNaN);
fprintf('Inf values replaced with median: %d\n', FeatureInf);
fprintf('==========================================\n\n');


% ------ Task 1.3: Using Z-Score to detect outliers then replace with median ------------
% Incorrectly flagged genuine high-demand events.
% Preserving row count and avoiding bias from row removal.

T_clean = T; % save state after NaN/Inf handling, before outlier removal
Z = 3:15; % Iterative Z thresholds from 3 to 15 (with one interval).
results  = zeros(length(Z), 4); % diaplay RMSE_train, RMSE_test, R2_train, R2_test

stored(length(Z)) = struct('Z', [], 'X_train', [], 'Y_train', [], 'X_test', [], 'Y_test', []...
, 'featureCols', [], 'theta_hat', [],'rmse_train', [], 'rmse_test', [], 'r2_train', [], 'r2_test', []);

% ------ Task 1.4: Correlation Analysis --------
X_all = table2array(T_clean(:, FeatureCols));
Y_all = T_clean.(TargetCol);

corr_all = zeros(length(FeatureCols), 1); % pre-allocated as a 28×1 vector of zeros
for i = 1:length(FeatureCols) % loop though each feature
    r = corrcoef(X_all(:, i), Y_all); % return a 2x2 matrix
    corr_all(i) = r(1, 2); 
end

% Arranged from highest to lowest absolute value
[~, sort_idx] = sort(abs(corr_all), 'descend');
sorted_corr_all = corr_all(sort_idx);
sorted_names_all = FeatureCols(sort_idx);

% Figure 1.2: Feature Correlation with EnergyRequestedFromGrid
figure(2);
bar(sorted_corr_all, 'FaceColor', [0.2 0.5 0.8]);
set(gca, 'XTick', 1:length(sorted_names_all), 'XTickLabel', sorted_names_all,'XTickLabelRotation', 45);
ylabel('Correlation with Target');
title('Figure 1.2: Feature Correlation with EnergyRequestedFromGrid');
yline(0, 'r--', 'LineWidth', 1);
ylim([-0.3 1]);
grid on;

for k = 1:length(Z) % loop though each Z once at a time
    T = T_clean; % reset table each iteration
    Z_threshold = Z(k); % set Z threshold
    TotalOutliers = 0;

    % This loop remove outliers
    for i = 1:length(FeatureCols)
        Col_3 = FeatureCols{i};
        ColData = T.(Col_3); % output in vector form
        mu = mean(ColData); % compute mean
        sigma = std(ColData); % compute std
        if sigma > 0
            Z_Scores = (ColData - mu) / sigma;
            Outliers = abs(Z_Scores) > Z_threshold; % find outliers
            ColData(Outliers) = median(ColData); % replace outliers
            TotalOutliers = TotalOutliers + sum(Outliers);
        end
        T.(Col_3) = ColData;
    end

    % ------ Task 1.5: Features dropping----------------
    % T.WeatherIcon       = [];    rejected already (see task 1.1.)
    T.windBearing         = [];  % physically irrelevant to energy demand
    T.dewPoint            = [];  % redundant with temperature + humidity
    T.humidity            = [];  % less than 0.5% change in demand (see reference[1])
    T.pressure            = [];  % no established link to household electricity consumption
    T.windSpeed           = [];  % less than 0.5% change in demand (see reference[1])
    T.apparentTemperature = [];  % highly collinear with temperature and humidity
    T.visibility          = [];  % no established link to household electricity consumption
    % 8 features are removed, 21 features remaining (29-8=21)

    % ------ Task 1.6; Train/Test split--------------
    % Train:80%, Test:20%
    % No need for cross-validation 
    Cols = T.Properties.VariableNames;
    TargetCol = "EnergyRequestedFromGrid_kW_";
    FeatureCols = Cols(~strcmp(Cols, TargetCol));

    X = table2array(T(:, FeatureCols)); % convert features to matrix X
    Y = T.(TargetCol); % extract target as vector Y

    %rng() is fixed seed generator, this make sure every test result is idencial
    rng(10); % using rng function with a set seed of 10 to generate random numbers
    N = height(T);
    cv = cvpartition(N, 'HoldOut', 0.2); % create hold-out partition, 20% to test
    X_train = X(training(cv), :);
    X_test = X(test(cv), :);
    Y_train = Y(training(cv));
    Y_test = Y(test(cv));

    % ---- Task 1.7: Standardisation (training set only) ----
    % Only for training set to avoid data leakage
    % Mean and std need to be calculated first
    for i = 1:size(X_train, 2)
        ColMean = mean(X_train(:, i)); % compute mean
        ColStd  = std(X_train(:, i)); % compute std
        if ColStd > 0 % skip if std is 0 (we can't deviding by 0)
            X_train(:, i) = (X_train(:, i) - ColMean) / ColStd; % standardises train set
            X_test(:, i) = (X_test(:, i) - ColMean) / ColStd; % standardises test set
        end
    end
%% Task 2 - Linear Regression Model
% ----- Task 2.1: Normal Equation ------------
    Psi_train = [ones(size(X_train, 1), 1), X_train]; % adding one col to train set
    Psi_test = [ones(size(X_test,  1), 1), X_test]; % adding one col to test set
    theta_hat = (Psi_train' * Psi_train) \ (Psi_train' * Y_train); % apply normal equation

    Y_hat_train = Psi_train * theta_hat; % ŷ_train = Ψθ
    Y_hat_test = Psi_test * theta_hat; % ŷ_test = Ψθ

    rmse_train = sqrt(mean((Y_train - Y_hat_train).^2)); % calculate RMSE train
    rmse_test = sqrt(mean((Y_test  - Y_hat_test ).^2)); % calculate RMSE test
    r2_train = 1 - sum((Y_train - Y_hat_train).^2) / sum((Y_train - mean(Y_train)).^2); % calculate R^2_train
    r2_test = 1 - sum((Y_test - Y_hat_test ).^2) / sum((Y_test - mean(Y_test )).^2); % calculate R^2_test

    results(k, :) = [rmse_train, rmse_test, r2_train, r2_test]; % save all four results

    % Store all data for a chosen Z (see task 2.2)
    stored(k).Z = Z(k);
    stored(k).X_train = X_train;
    stored(k).Y_train = Y_train;
    stored(k).X_test = X_test;
    stored(k).Y_test = Y_test;
    stored(k).featureCols = FeatureCols;
    stored(k).theta_hat = theta_hat;
    stored(k).rmse_train = rmse_train;
    stored(k).rmse_test = rmse_test;
    stored(k).r2_train = r2_train;
    stored(k).r2_test = r2_test;
end % finally... loop ended

% ----- Display 2.1: Z-Score Analysis (display) ----------
fprintf('\n============ Z-Score Result Analysis ==============\n');
fprintf('%-5s  %-12s  %-12s  %-10s  %-10s\n', 'Z', 'RMSE Train', 'RMSE Test', 'R2 Train', 'R2 Test');
fprintf('%s\n', repmat('-', 1, 55));

for k = 1:length(Z)
    fprintf('%-5d  %-12.4f  %-12.4f  %-10.4f  %-10.4f\n', ...
        Z(k), results(k,1), results(k,2), results(k,3), results(k,4));
end
fprintf('%s\n', repmat('=', 1, 55));

% ----- Display 2.2: Train/Test Gap (display)----------
fprintf('\n===== Train/Test Gap Analysis =====\n');
fprintf('%-5s  %-12s\n', 'Z', 'RMSE Gap');
fprintf('%s\n', repmat('-', 1, 32));
for k = 1:length(Z)
    rmse_gap = abs(results(k,2) - results(k,1));
    fprintf('%-5d  %-12.4f\n', Z(k), rmse_gap);
end
fprintf('%s\n', repmat('=', 1, 32));

% ------ Task 2.2: Z=12 Selection and Plots (start)--------------------------
% Select Z threshold based on RMSE, R^2 and gaps analysis
Z_selected = 12;
k_sel = find([stored.Z] == Z_selected);

% Save all data when Z=12
X_train = stored(k_sel).X_train;
Y_train = stored(k_sel).Y_train;
X_test = stored(k_sel).X_test;
Y_test = stored(k_sel).Y_test;
FeatureCols = stored(k_sel).featureCols;
theta_hat = stored(k_sel).theta_hat;
rmse_train = stored(k_sel).rmse_train;
rmse_test = stored(k_sel).rmse_test;
r2_train = stored(k_sel).r2_train;
r2_test = stored(k_sel).r2_test;

corr_vals = zeros(length(FeatureCols), 1);
for i = 1:length(FeatureCols)
    r = corrcoef(X_train(:, i), Y_train);
    corr_vals(i) = r(1, 2);
end

[~, idx] = sort(abs(corr_vals), 'descend');
sorted_corr = corr_vals(idx);
sorted_names = FeatureCols(idx);

% ----- Display 2.3: Z Selection and Standardisation Check  ------------
% Verify standarisation again before continue
fprintf('%s\n', repmat('=', 1, 35));
fprintf('Standardisation check (if Z=12):\n'); 
fprintf('Mean of X_train (expect ≈ 0): %.3f\n', mean(mean(X_train))); % display X_train (mean) and should be 0
fprintf('Std  of X_train (expect ≈ 1): %.3f\n', mean(std(X_train))); % display X_train (std) and should be 1
fprintf('%s\n', repmat('=', 1, 35));


% ----- Display 2.4: Linear Regression Results ----------
fprintf('\n==========Normal Equation===========\n');
fprintf('RMSE Train: %.4f kW\n', rmse_train); 
fprintf('RMSE Test:  %.4f kW\n', rmse_test);
fprintf('R² Train:   %.4f\n',    r2_train);
fprintf('R² Test:    %.4f\n',    r2_test);
fprintf('%s\n', repmat('=', 1, 35));

% ----- Display 2.5: Normal Equation Weights----------
fprintf('\n\n===== Normal Equation Weights =====\n');
fprintf('%-35s  %s\n', 'Feature', 'Weight');
fprintf('%s\n', repmat('-', 1, 50));
fprintf('%-35s  %+.4f\n', 'Intercept (a0)', theta_hat(1));
for i = 2:length(theta_hat)
    fprintf('%-35s  %+.4f\n', FeatureCols{i-1}, theta_hat(i));
end
fprintf('%s\n', repmat('=', 1, 35));

% ------ Task 2.3: Plots (start)--------------------------
% Figure 2.1: Predicted vs Actual (Normal Equation)
figure(3);
subplot(1,2,1);
scatter(Y_test, Y_hat_test, 5, 'filled');
hold on;
refLine = [min(Y_test), max(Y_test)];
plot(refLine, refLine, 'r--', 'LineWidth', 2);
hold off;
xlabel('Actual Energy (kW)');
ylabel('Predicted Energy (kW)');
title(sprintf('Figure 2.1: Predicted vs Actual (Z=12)'));
legend('Predictions', 'Perfect fit (y = x)', 'Location', 'northwest');
grid on;

% Figure 2.2: Residual Plot (Normal Equation) (plotted with Figure 2.1)
residuals = Y_test - Y_hat_test;
subplot(1,2,2)
scatter(Y_hat_test, residuals, 5, 'filled', 'MarkerFaceAlpha', 0.6);
yline(0, 'r--', 'LineWidth', 2);
xlabel('Predicted Energy (kW)');
ylabel('Residual (Actual − Predicted) (kW)');
title('Figure 2.2: Normal Equation: Residual Plot (Test Set)');
grid on;

% Figure 2.3: Residual Histogram
figure(4);
histogram(residuals, 50);
xlabel('Residual (kW)');
ylabel('Frequency');
title('Figure 2.3: Distribution of Residuals');
xline(0, 'r--', 'LineWidth', 2);
grid on;

corrMatrix = corrcoef(X_train); % compute Pearson correlation coefficient between features
threshold  = 0.7; % first set threshold for r

% Figure 2.4; Collinearity Heatmap
figure(5);
imagesc(corrMatrix);
colorbar; colormap('jet'); clim([-1 1]);
xticks(1:length(FeatureCols)); yticks(1:length(FeatureCols));
xticklabels(FeatureCols); yticklabels(FeatureCols);
xtickangle(45);
title('Figure 2.4: Feature-Feature Correlation Matrix (Collinearity Check)');

% Figure 2.5: Feature Correlation with Target
figure(6);
bar(sorted_corr, 'FaceColor', [0.2 0.5 0.8]);
set(gca, 'XTick', 1:length(sorted_names), 'XTickLabel', sorted_names,'XTickLabelRotation', 45);
ylabel('Pearson Correlation with Energy Demand');
title('Figure 2.5: Feature Correlation with Target (Z = 12, Training Set)');
yline( 0.7, '--r', '|r| = 0.7', 'LabelHorizontalAlignment', 'left');
yline(-0.7, '--r');
ylim([-1 1]);
grid on;

% Figure 2.6: Z-Score Sensitivity — RMSE vs Z threshold
figure(7);
plot(Z, results(:,1), 'b-o', 'LineWidth', 1.3, 'DisplayName', 'RMSE Train','MarkerSize',9);
hold on;
plot(Z, results(:,2), 'r-o', 'LineWidth', 1.3, 'DisplayName', 'RMSE Test','MarkerSize',7);
hold off;
xline(12, 'k--', 'Z = 12', 'LineWidth', 1);
xlabel('Z Threshold');
ylabel('RMSE (kW)');
title('Figure 2.6: Z-Score Sensitivity Analysis — RMSE vs Threshold');
legend('Location', 'best');
grid on;
% ------ Task 2.3: Plots (end)--------------------------


% ------ Task 2.4: Collinearity Analysis-----------
fprintf('\n===== HIGHLY CORRELATED FEATURE PAIRS (|r| > %.1f) =====\n', threshold);
fprintf('%-30s  %-30s  %s\n', 'Feature 1', 'Feature 2', 'Correlation');
fprintf('%s\n', repmat('-', 1, 72));
foundAny = false; % set a flag to see whether it found any high correlated pairs
for i = 1:length(FeatureCols) 
    for j = i+1:length(FeatureCols) % each paris will only be checked once (avoid check A&B then B&A)
        r = corrMatrix(i, j); % retrieve i and j
        if abs(r) > threshold
            fprintf('%-30s  %-30s  %+.4f\n', FeatureCols{i}, FeatureCols{j}, r); % list all feature pairs if abs(r)>0.7
            foundAny = true;
        end
    end
end
if ~foundAny
    fprintf('No feature pairs found with |r| > %.1f\n', threshold); % return non if not found
end
fprintf('%s\n', repmat('=', 1, 72));

%% Task 3: Second Model — Gradient Boosting (GB) Method(Non-linear Model))

% =========================================================================
% MAIN IDEA: Using "fitrensemble" function to train ensemble and
% "LSBoost" method to fit all previous residuals into new trees.
% The final prediction = weighted sum of all trees * LearnRate; need to
% tune the number of cycles (trees) and learning rate.
%
% APPROACH: Iterative with selecting hyperparameters, then compare RMSE and
% R^2 for train and test result to find the suitable combinaiton (mitigate 
% overfittings). Finally train the model again using the selected 
% hyperparameters.
% =========================================================================

% ----- Task 3.1: Hyperparameter Grid Search (5x5 = 25 combinations) ----------
cycles_range = 100:100:500; % 5 values, interval of 100
learnRate_range = 0.1:0.1:0.5; % 5 values, interval of 0.1

GB_results = zeros(length(cycles_range), length(learnRate_range), 5); % create a 5x5x5 matrix (two RMSE, twoR^2, gap)

fprintf('Running GB grid search (25 combinations)...This process might take few minutes...\n');

for cycles_idx = 1:length(cycles_range)
    for learnRate_idx = 1:length(learnRate_range)
        mdl_tmp = fitrensemble(X_train, Y_train, 'Method', 'LSBoost','NumLearningCycles',...
                cycles_range(cycles_idx),'LearnRate',  learnRate_range(learnRate_idx)); % (see description above)
        % to complete a fair comparison, the step above use X_train and Y_train splitted in task 1

        Y_hat_train_GB = predict(mdl_tmp, X_train); % find ŷ_train_GB
        Y_hat_test_GB = predict(mdl_tmp, X_test); % find ŷ_test_GB

        GB_results(cycles_idx, learnRate_idx, 1) = sqrt(mean((Y_train - Y_hat_train_GB).^2)); % RMSE_train_GB
        GB_results(cycles_idx, learnRate_idx, 2) = sqrt(mean((Y_test - Y_hat_test_GB).^2)); % RMSE_test_GB
        GB_results(cycles_idx, learnRate_idx, 3) = 1 - sum((Y_train - Y_hat_train_GB).^2) / sum((Y_train - mean(Y_train)).^2); % R^2_train_GB
        GB_results(cycles_idx, learnRate_idx, 4) = 1 - sum((Y_test - Y_hat_test_GB).^2) / sum((Y_test - mean(Y_test )).^2); % R^2_test_GB
        GB_results(cycles_idx, learnRate_idx, 5) = GB_results(cycles_idx, learnRate_idx, 2) - GB_results(cycles_idx, learnRate_idx, 1); % RMSE gaps
    end
end

% ----- Display 3.1: Display Full Results Table --------------------------
fprintf('\n================ GB Hyperparameter Search Results ==================\n');
fprintf('%-8s  %-10s  %-12s  %-12s  %-10s  %-10s  %-10s\n', ...
    'Cycles', 'LearnRate', 'RMSE Train', 'RMSE Test', 'R2 Train', 'R2 Test', 'RMSE Gap');
fprintf('%s\n', repmat('-', 1, 80));
for cycles_idx = 1:length(cycles_range)
    for learnRate_idx = 1:length(learnRate_range)
        fprintf('%-8d  %-10.2f  %-12.4f  %-12.4f  %-10.4f  %-10.4f  %-10.4f\n', ...
            cycles_range(cycles_idx), learnRate_range(learnRate_idx), ...
            GB_results(cycles_idx,learnRate_idx,1), GB_results(cycles_idx,learnRate_idx,2), ...
            GB_results(cycles_idx,learnRate_idx,3), GB_results(cycles_idx,learnRate_idx,4), ...
            GB_results(cycles_idx,learnRate_idx,5));
    end
end
fprintf('%s\n', repmat('=', 1, 80)); 

% ----- Task 3.2: Select Best Outputs (mini test RMSE) -------------------
rmse_test_grid = GB_results(:, :, 2); % extracts just the RMSE test values
[~, best_idx]  = min(rmse_test_grid(:)); % find the index which has mini value
[best_ci, best_li] = ind2sub(size(rmse_test_grid), best_idx); % convert back the index to find best outputs

best_cycles    = cycles_range(best_ci); %apply the selected cycles
best_learnRate = learnRate_range(best_li); % apply the selected learning rate

fprintf('Selected: NumCycles=%d, LearnRate=%.2f (lowest test RMSE)\n\n', ...
    best_cycles, best_learnRate);


% ---- Task 3.3: Train Final GB Model with Best Hyperparameters -------------
% Same logic as task 3.1 but only apply the best cycles and learnRate values instead
mdl_GB = fitrensemble(X_train, Y_train, 'Method','LSBoost','NumLearningCycles', ...
        best_cycles, 'LearnRate', best_learnRate);

Y_hat_train_GB_final = predict(mdl_GB, X_train); 
Y_hat_test_GB_final = predict(mdl_GB, X_test);

rmse_train_GB = sqrt(mean((Y_train - Y_hat_train_GB_final).^2));
rmse_test_GB = sqrt(mean((Y_test - Y_hat_test_GB_final ).^2));
r2_train_GB = 1 - sum((Y_train - Y_hat_train_GB_final).^2) / sum((Y_train - mean(Y_train)).^2);
r2_test_GB = 1 - sum((Y_test - Y_hat_test_GB_final ).^2) / sum((Y_test - mean(Y_test )).^2);
gap_GB = rmse_test_GB - rmse_train_GB;

% Display 3.2 GB Results
fprintf('\n===== Gradient Boosting (LSBoost) =====\n');
fprintf('NumLearningCycles:     %d\n',  best_cycles);
fprintf('LearnRate:             %.2f\n', best_learnRate);
fprintf('RMSE Train:            %.4f kW\n', rmse_train_GB);
fprintf('RMSE Test:             %.4f kW\n', rmse_test_GB);
fprintf('R² Train:              %.4f\n', r2_train_GB);
fprintf('R² Test:               %.4f\n', r2_test_GB);
fprintf('Train/Test Gap (RMSE): %.4f kW\n', gap_GB);
fprintf('%s\n', repmat('=', 1, 40));

% Display 3.3: Model Comparison Table (GB & Linear Regression)
fprintf('\n========================= MODEL COMPARISON =========================\n');
fprintf('%-22s  %-10s  %-10s  %-8s  %-8s  %s\n', ...
    'Model', 'RMSE Train', 'RMSE Test', 'R2 Train', 'R2 Test', 'Gap');
fprintf('%s\n', repmat('-', 1, 72));
fprintf('%-22s  %-10.4f  %-10.4f  %-8.4f  %-8.4f  %.4f\n', ...
    'Linear Regression', rmse_train, rmse_test, r2_train, r2_test, ...
    rmse_test - rmse_train);
fprintf('%-22s  %-10.4f  %-10.4f  %-8.4f  %-8.4f  %.4f\n', ...
    'Gradient Boosting', rmse_train_GB, rmse_test_GB, r2_train_GB, r2_test_GB, gap_GB);
fprintf('%s\n', repmat('=', 1, 72));

% Figure 3.1; GB model Predicted vs Actual 
figure(8);
subplot(1,2,1);
scatter(Y_test, Y_hat_test_GB_final, 3, 'filled', 'MarkerFaceAlpha', 0.7);
hold on;
refLine = [min(Y_test), max(Y_test)];
plot(refLine, refLine, 'r--', 'LineWidth', 2);
hold off;
xlabel('Actual Energy (kW)');
ylabel('Predicted Energy (kW)');
title(sprintf('Figure 3.1: GB: Predicted vs Actual\nRMSE=%.4f  R²=%.4f', ...
    rmse_test_GB, r2_test_GB));
legend('Predictions', 'Perfect fit (y = x)', 'Location', 'northwest');
grid on;

% Figure 3.2: Residual Plot
residuals_GB = Y_test - Y_hat_test_GB_final;
subplot(1,2,2);
scatter(Y_hat_test_GB_final, residuals_GB, 3, 'filled', 'MarkerFaceAlpha', 0.7);
yline(0, 'r--', 'LineWidth', 2);
xlabel('Predicted Energy (kW)');
ylabel('Residual (Actual − Predicted) (kW)');
title('Figure 3.2 — GB: Residual Plot (Test Set)');
grid on;

% ------ Task 3.4:Learning curve --------------
% fix best LearnRate, vary cycles
lc_cycles = 50:50:500;
lc_rmse_train = zeros(length(lc_cycles), 1);
lc_rmse_test = zeros(length(lc_cycles), 1);

for i = 1:length(lc_cycles)
    mdl_lc = fitrensemble(X_train, Y_train, 'Method', 'LSBoost', ...
        'NumLearningCycles', lc_cycles(i), 'LearnRate', best_learnRate);
    lc_rmse_train(i) = sqrt(mean((Y_train - predict(mdl_lc, X_train)).^2));
    lc_rmse_test(i) = sqrt(mean((Y_test - predict(mdl_lc, X_test )).^2));
end

% Figure 3.3: Learning Curve Plot
figure(9);
plot(lc_cycles, lc_rmse_train, 'b-o', 'LineWidth',1.3, 'DisplayName', 'RMSE Train', 'MarkerSize',9);
hold on;
plot(lc_cycles, lc_rmse_test,  'r-o', 'LineWidth',1.3, 'DisplayName', 'RMSE Test','MarkerSize',9);
xline(200, 'k--', '200 Cycles','LineWidth',1);
xlabel('NumLearningCycles'); ylabel('RMSE (kW)');
title('Figure 3.3: Learning Curve and Overfitting Check');
legend; 
grid on;