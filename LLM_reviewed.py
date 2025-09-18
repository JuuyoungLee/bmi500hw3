'''
prompt used
You are a senior Python reviewer. Review the code below. 
Focus: correctness/performance/readability/typo/errors 
Return a table: Severity/Lines/Issue/Why/Specific fix 
Then show a minimal diff patch for the top 3 issues.
'''
'''
| Severity |                                  Lines | Issue                                                                                                                                                                     | Why it matters                                                                                            | Specific fix                                                                                                                       |
| -------- | -------------------------------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **High** |       24–34 (`FeatureSelection.lasso`) | **Wrong feature indices returned.** You sort `coefs[non_zero_ind]` but then return positions within that slice, not the original column indices.                          | Downstream selection uses wrong columns → silent correctness bug (models trained on unintended features). | Map back to original indices: `order = np.argsort(coefs[non_zero_ind])[::-1]; ranks = non_zero_ind[order]`. Handle all-zero coefs. |
| **High** |                          3–6 (imports) | `gradescope_utils` is imported but unused. In most environments it isn’t installed → `ModuleNotFoundError` before anything runs. `scipy` top-level import is also unused. | Prevents execution; hurts portability.                                                                    | Remove `gradescope_utils` and unused `scipy` import (keep `from scipy.stats import …`).                                            |
| **Med**  |             13–22 (`rank_correlation`) | Name says “rank correlation” but you compute **Pearson**, use **signed** correlation, and don’t handle constant cols → `nan` can break rankings.                          | Misranks strong *negative* relationships; constants yield `nan`; behavior is unstable.                    | Use **Spearman** (rank-based), take `abs`, and coerce `nan` to 0. Also drop the wrong preallocation (`y.shape[0]`).                |
| Med      |                     37–66 (`stepwise`) | Forward selection evaluates on **training** RMSE only.                                                                                                                    | Overfits; optimistic performance, unstable feature order.                                                 | Evaluate on validation (or CV). E.g., pass `(X_val, y_val)` and use that RMSE for the stopping rule.                               |
| Med      |              79–88 (`tree_regression`) | `DecisionTreeRegressor` has no `random_state`.                                                                                                                            | Non-reproducible metrics and plots.                                                                       | Set `random_state=0` (or param).                                                                                                   |
| Med      | 168–258 (grid search prints) & 289–293 | You tune depth/leaf options several times but **ignore** the result later; final tree is hard-coded to `(max_depth=3, min_items=1)`.                                      | Reported “best config” isn’t actually used → misleading results.                                          | Store best config for the intended scenario (e.g., all features) and reuse it in final fit/plot.                                   |
| Low      |                           70–74, 80–85 | Preallocate `test_prob` then overwrite.                                                                                                                                   | Noise; minor perf/readability.                                                                            | Remove the redundant preallocation lines.                                                                                          |
| Low      |                                154–162 | `result_df` isn’t printed/ saved in script mode.                                                                                                                          | Nothing visible in non-notebook runs.                                                                     | `print(result_df)` or `to_csv`.                                                                                                    |
| Low      |                                General | No type hints / docstrings.                                                                                                                                               | Harder to review & test.                                                                                  | Add signatures & short docstrings.                                                                                                 |
| Low      |                                Data IO | No NA handling (estimators choke on NaNs).                                                                                                                                | Potential runtime errors.                                                                                 | `SimpleImputer` or dropna prior to fit.                                                                                            |

'''
--- a/previous_project.py
+++ b/previous_project.py
@@
-import numpy as np
-import sklearn.linear_model as skl
-import gradescope_utils
-import pandas as pd
-import scipy
-
-from scipy.stats import pearsonr
+import numpy as np
+import sklearn.linear_model as skl
+import pandas as pd
+# removed unused/fragile imports: gradescope_utils, scipy
+from scipy.stats import spearmanr
 from sklearn.metrics import mean_squared_error, r2_score
 from sklearn.tree import DecisionTreeRegressor
@@
 class FeatureSelection:
     def rank_correlation(self, x, y):
-        ranks = np.zeros(y.shape[0])
-        #your code here
-        corr_all = []
-        for i in range(x.shape[1]):
-            corr, _ = pearsonr(x[:, i], y)
-            corr_all.append(corr)
-        ranks = np.argsort(corr_all)[::-1]    
-  
-        return ranks
+        # Rank features by absolute **Spearman** (rank-based) correlation with y.
+        scores = []
+        for i in range(x.shape[1]):
+            corr, _ = spearmanr(x[:, i], y)
+            # handle constants/NaNs robustly
+            if not np.isfinite(corr):
+                corr = 0.0
+            scores.append(abs(corr))
+        ranks = np.argsort(scores)[::-1]
+        return ranks
@@
     def lasso(self, x, y):
-        ranks = np.zeros(y.shape[0])
-        # your code here
-        lasso = skl.Lasso(alpha=0.01) 
+        # rank by absolute Lasso coefficients (largest first)
+        lasso = skl.Lasso(alpha=0.01, max_iter=10000)
         lasso.fit(x, y)
-        
-        coefs = np.abs(lasso.coef_)
-        non_zero_ind = np.where(coefs != 0)[0]
-        ranks = np.argsort(coefs[non_zero_ind])[::-1]
-
-        return ranks
+        coefs = np.abs(lasso.coef_)
+        if np.all(coefs == 0):
+            # fallback: rank all features by (zero) coefs → stable order
+            return np.argsort(coefs)[::-1]
+        non_zero_ind = np.flatnonzero(coefs)
+        order = np.argsort(coefs[non_zero_ind])[::-1]
+        ranks = non_zero_ind[order]
+        return ranks
@@
-        tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_items)
+        tree = DecisionTreeRegressor(
+            max_depth=max_depth,
+            min_samples_leaf=min_items,
+            random_state=0  # reproducible results
+        )
