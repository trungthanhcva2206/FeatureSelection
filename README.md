# FeatureSelection

Toolbox for comparing manual feature selection implementations against scikit-learn variants on small-sample, high-dimensional gene-expression datasets.

## Data Summary
All CSVs place the label in the first column and features afterward.

| Dataset | Samples | Features | Classes |
| --- | ---: | ---: | ---: |
| adenocarcinoma.csv | 76 | 9868 | 2 |
| Brain.csv | 42 | 5597 | 5 |
| Breast2classes.csv | 77 | 4869 | 2 |
| Breast3classes.csv | 95 | 4869 | 3 |
| CNS1.csv | 60 | 7129 | 2 |
| colon1.csv | 62 | 2000 | 2 |
| DLBCL.csv | 77 | 5469 | 2 |
| Leukemia_3c1.csv | 72 | 7129 | 3 |
| Leukemia_4c1.csv | 72 | 7129 | 4 |
| Lung_cancer.csv | 203 | 12600 | 5 |
| Lymphoma.csv | 62 | 4026 | 3 |
| NCI.csv | 61 | 5244 | 8 |
| Prostate.csv | 102 | 6033 | 2 |
| SRBCT_txt.csv | 63 | 2308 | 4 |
| Tumors9.csv | 60 | 5726 | 9 |

## Algorithms (manual implementations)
- `variance_threshold.py`: drop features with variance below threshold or keep top-k by variance.
- `chi_square.py`: Chi-square per feature. Two modes: binned (quantile bins) or raw non-negative shift (sklearn-style when `--use-raw`).
- `mutual_information.py`: Mutual information per feature. Modes: quantile binning; k-NN estimator via sklearn when `--use-knn`; or manual k-NN (no sklearn) when `--use-knn-manual`.
- `pearson_correlation.py`: Pearson correlation (optionally absolute) between feature and label-coded numeric.
- `relief_f.py`: ReliefF with k-NN hits/misses after min-max scaling.

## Algorithms (sklearn-backed)
- `sklearn_variance_threshold.py`: Uses sklearn VarianceThreshold.
- `sklearn_chi_square.py`: Uses sklearn `chi2` (requires non-negative, raw, no binning).
- `sklearn_mutual_information.py`: Uses sklearn `mutual_info_classif` (k-NN, no binning).
- `sklearn_pearson.py`: Correlation via numpy/scipy (aligned with manual scoring) plus sklearn CV for k search.
- `sklearn_relief_f.py`: ReliefF-style implementation mirroring manual code for comparison.

## Core formulas (per feature)
- Variance: $\sigma^2 = \frac{1}{n} \sum_i (x_i - \bar{x})^2$.
- Chi-square on contingency: $\chi^2 = \sum_{c,b} \frac{(O_{cb}-E_{cb})^2}{E_{cb}}$, with $E_{cb}=\frac{(\text{row}_c)(\text{col}_b)}{N}$.
- Mutual information (discrete): $I(X;Y)=\sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$.
- Pearson: $r = \frac{\sum_i (x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_i (x_i-\bar{x})^2}\sqrt{\sum_i (y_i-\bar{y})^2}}$ (often $|r|$ for ranking).
- ReliefF weight update: for each sample, subtract distance to nearest hit and add distance to nearest miss, averaged across neighbors and samples.

## Mathematical foundations (sketch)
- Variance threshold
	- Assumes informative features vary across samples. Low $\sigma^2$ implies near-constant, so drop.
- Chi-square
	- Tests independence between feature and label on a contingency table. Large $\chi^2$ means observed counts deviate from independence. Requires non-negative counts.
	- In raw/non-binned mode, `chi2` in sklearn builds $O$ by summing non-negative feature values per class; expected $E$ uses row/column marginals.
	- In binned mode, continuous values are discretized; $O$ counts samples per (class, bin).
- Mutual information
	- Measures reduction in uncertainty: $I(X;Y)=H(Y)-H(Y|X)$. Zero means independence; higher is more informative.
	- k-NN estimator (Kraskov-style in `mutual_info_classif`) computes local neighbor volumes in joint vs marginal spaces; avoids bins.
	- Binned estimator replaces densities with empirical probabilities over quantile bins (plug-in estimate).
- Pearson correlation
	- Measures linear association; for classification, labels are encoded numeric. Using $|r|$ ranks by linear predictive strength. Squared correlation relates to simple linear regression $R^2$.
- ReliefF
	- For each sample, find $k$ nearest hits (same class) and misses (other classes). Update weight: decrease for large distance to hits, increase for large distance to misses. Aggregated over samples approximates feature's ability to separate classes in local neighborhoods.

## Binning vs non-binning
- Chi-square
	- Binned (manual default): quantile bins create a contingency table that tolerates signed values and outliers; reduces noise when $n \ll d$ but may blur fine-grained signal if bins are too coarse.
	- Non-binned (`--use-raw`, manual) and sklearn: per-feature min-shift to non-negative then treat magnitudes as counts; preserves ordering and spacing but requires data to be non-negative after shifting.
 - Mutual information
	- Binned (manual default): plug-in estimate on quantile bins; low-variance when $n$ is tiny, but bias grows as bins increase and boundaries can move with each dataset split.
	- Non-binned k-NN (`--use-knn` / `--use-knn-manual`): continuous estimator; keeps local geometry, less sensitive to arbitrary bin edges, but higher variance when $n$ is very small; choose larger `--n-neighbors` to smooth.
 - ReliefF
	- Always non-binned; distances use min–max scaling so magnitude differences are preserved. Binning would break neighborhood structure, so it is intentionally avoided.

**Khi nên bin**
- Dữ liệu có giá trị âm hoặc nhiều ngoại lệ, cần tránh bước shift không âm (chi-square). 
- Số mẫu rất nhỏ, muốn giảm nhiễu bằng cách gom nhóm giá trị liên tục.

**Khi không nên bin**
- Cần giữ thứ tự và độ lớn gốc (MI k-NN, chi-square dạng raw, ReliefF). 
- Dữ liệu đủ sạch/đủ mẫu để ước lượng liên tục ổn định; tránh sai lệch do ranh giới bin tùy ý.

**Tóm tắt so sánh**
- Binned: giảm phương sai, thêm bias do ranh giới; an toàn cho dữ liệu âm và nhiễu; phù hợp khi $n$ rất nhỏ hoặc cần diễn giải bằng bảng đếm.
- Non-binned: giữ cấu trúc liên tục, tránh bias bin; cần chuẩn hóa/shift thích hợp và cẩn trọng với ngoại lệ; hiệu quả hơn khi số mẫu vừa đủ và muốn tận dụng khoảng cách/k-láng giềng.

## Kết quả gần đây (adenocarcinoma)
| Algorithm | Chế độ | Metric | k chọn | Điểm | Tham số chính | Log |
| --- | --- | --- | ---: | ---: | --- | --- |
| Mutual Information (manual k-NN) | Non-bin (`--use-knn-manual`, n=3) | Accuracy (CV 5-fold) | 500 | 0.9208 | `n_neighbors=3` | [mi_sklearn_log.txt](mi_sklearn_log.txt#L1-L9)
| Mutual Information (manual k-NN) | Non-bin (`--use-knn-manual`, n=3) | Macro-F1 (CV 5-fold) | 500 | 0.8336 | `n_neighbors=3` | [mi_log.txt](mi_log.txt#L1-L8)
| ReliefF (manual) | Non-bin | Accuracy (CV grid) | 10 | 0.9467 | `n_neighbors=10` | [rf_sklearn_log.txt](rf_sklearn_log.txt#L1-L10)
| ReliefF (manual) | Non-bin | Macro-F1 (CV grid) | 10 | 0.8519 | `n_neighbors=10` | [rf_log.txt](rf_log.txt#L1-L10)

## Usage examples
- Manual Chi-square, sklearn-style raw, CV over k: 
	`python chi_square.py Data/adenocarcinoma.csv --use-raw --cv-topk-grid 10,50,100,500,1000,2000,5000,7000 --cv-folds 5 --cv-metric accuracy --log-file chi_log.txt`
- Sklearn Chi-square: 
	`python sklearn_chi_square.py Data/adenocarcinoma.csv --cv-topk-grid 10,50,100,500,1000,2000,5000,7000 --cv-folds 5 --cv-metric accuracy --log-file chi_sklearn_log.txt`
- Manual MI with k-NN (no bins, manual estimator): 
	`python mutual_information.py Data/adenocarcinoma.csv --use-knn-manual --n-neighbors 3 --cv-topk-grid 10,50,100,500,1000,2000,5000,7000 --log-file mi_manual_knn_log.txt`
- Manual MI with k-NN (no bins, sklearn estimator): 
	`python mutual_information.py Data/adenocarcinoma.csv --use-knn --n-neighbors 3 --cv-topk-grid 10,50,100,500,1000,2000,5000,7000 --log-file mi_manual_log.txt`
- Sklearn MI: 
	`python sklearn_mutual_information.py Data/adenocarcinoma.csv --n-neighbors 3 --cv-topk-grid 10,50,100,500,1000,2000,5000,7000 --log-file mi_sklearn_log.txt`
- ReliefF manual: 
	`python relief_f.py Data/adenocarcinoma.csv --n-neighbors 10 --top-k 50 --log-file rf_manual_log.txt`
- ReliefF sklearn-style: 
	`python sklearn_relief_f.py Data/adenocarcinoma.csv --n-neighbors 10 --top-k 50 --log-file rf_sklearn_log.txt`

## Cross-validation search for k
All scripts accept `--cv-topk-grid` to search k via StratifiedKFold + LogisticRegression (+ StandardScaler). Metrics: `accuracy` or `macro-f1`. If no grid is provided, use `--top-k` or `--threshold`.

## Logging
`--log-file path.txt` appends a human-readable report (timestamp, dataset, selection mode, CV scores, top-5 feature scores, dropped features). Logs are not required for execution but help reproducibility.

## Setup
- Requires Python 3.9+; install deps: `pip install -r requirements.txt` (or `pip install numpy pandas scikit-learn` for full functionality).
- Run scripts from repo root so relative paths to `Data/` resolve.

## Notes and tips
- Chi-square requires non-negative inputs; manual binned mode sidesteps this, raw mode shifts features non-negative to match sklearn.
- MI: prefer k-NN mode for continuous data to avoid bin-sensitivity; adjust `--n-neighbors` for bias-variance trade-off.
- High-dimensional data benefits from CV grid that spans small and large k; narrow the grid if CV keeps selecting very large k.