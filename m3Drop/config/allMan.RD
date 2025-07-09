\name{bg__fit_gamma}
\alias{bg__fit_gamma}
\title{Fit Gamma Distribution}
\description{
	Fits the parameters of a gamma distribution to a set of observations
}
\usage{
	bg__fit_gamma(x)
}
\arguments{
  \item{x}{vector of observations to fit.}
}
\details{
	scale = var(x)/mean(x)
	shape = mean(x)/scale
}
\value{list with two entries: shape and size.}
\examples{
# dat <- rgamma(100, 0.1, 1)
# params <- bg__fit_gamma(dat)
}
\keyword{simulations}

\name{bg__get_stats}
\alias{bg__get_stats}
\title{Calculate Simulation Statistics}
\description{
	Given significant genes and a ground truth calculates the false discovery rate and false negative rate of the significance calls.
}
\usage{	bg__get_stats(sig, TP, ngenes)}
\arguments{
  \item{sig}{vector of gene IDs/names which were called as significant by some method.}
  \item{TP}{vector of gene IDs/names which known to be positives (ground truth).}
  \item{ngenes}{number of total genes under consideration.}
}
\details{False discovery rate (FDR) is the proportion of all genes considered significant which are not in the ground truth. False negative rate (FNR) is the proportion of all ground truth genes which are not considered significant.
}
\value{vector of two elements: (FDR, FNR).}
\examples{
#	Calls <- c(1,3,5,8,10)
#	Truth <- c(2,3,5,9)
#	total=10
#	stats <- bg__get_stats(Calls, Truth, total)
}
\keyword{simulations, statistics}

\name{bg__default_mean2disp}
\alias{bg__default_mean2disp}
\title{Mean to Dispersion}
\description{
	Default function to calculate the negative binomial dispersion parameter given the mean expression of the gene.
}
\usage{
	bg__default_mean2disp(mu, coeffs=c(3.967816,-1.855054))
}
\arguments{
  \item{mu}{mean expression of the gene(s)}
  \item{coeffs}{coefficients of linear regression between log(cv2) and log10(mean expression) for observed data.}
}
\details{
Given the coefficients of a linear regression between the natural logarithm of the squared coefficient of variation and log base 10 of mean expression, calculates the dispersion parameter (1/size) of the negative binomial distribution for a given expression level.

Default coefficients were fit to the Buettner at al.[1] single cell mESC data.
}
\value{the dispersion parameter (1/size) for the (mu,size) paramterization of the \code{rnbinom} function.
}
\examples{
# mu <- 100
# r <- bg__default_mean2disp(mu)
# n_cells <- 50
# sim <- rnbinom(n_cells, size=1/r, mu=mu)
}
\keyword{simulations}
\references{
[1] Buettner et al. (2015) Computational analysis of cell-to-cell heterogeneity in single-cell RNA-sequencing data reveals hidden subpopulations of cells. Nature Biotechnology 33 : 155-160.
}

\name{bg__shift_size}
\alias{bg__shift_size}
\title{Shift Size Parameter}
\description{
	Shifts a fitted size paramter according to the a power-law relationship.
}
\usage{
	bg__shift_size(mu_all, size_all, mu_group, coeffs) 
}
\arguments{
	\item{mu_all}{vector of original mus}
	\item{size_all}{vector of original sizes}
	\item{mu_group}{vector of new mus}
	\item{coeffs}{coefficients of the power-law fit}
}
\details{
	Wrapper of short function which converts values to their logs then shifts them using the parameters of a linear fit before un-logging them.
}
\value{vector of new size parameters}
\examples{
#require("M3DExampleData")
#counts <- NBumiConvertData(Mmus_example_list$data, is.counts=TRUE);
#fit <- NBumiFitModel(counts);
#ceoffs <- NBumiFitDispVsMean(fit, suppress.plot=TRUE)
#new_size <- bg__shift_size(rowMeans(counts), fit$size, rowMeans(counts)*2, coeffs)
}
\keyword{simulations}

\name{bg__var_vs_drop}
\alias{bg__var_vs_drop}
\title{Variance vs Dropout Rate}
\description{
	Fixes the mean expression level and simulates a single gene with different fold changes across two equally sized subpopulations. 
}
\usage{
	bg__var_vs_drop(pop_size, fixed_mean, K=10.3, dispersion_from_mean=bg__default_mean2disp, suppress.plot=TRUE)
}
\arguments{
  \item{pop_size}{size of each subpopulation, i.e. 1/2 the total number of cells.}
  \item{fixed_mean}{expected mean expression across all cells.}
  \item{K}{K of the Michaelis-Menten function.}
  \item{dispersion_from_mean}{function which calculates the dispersion parameter of the negative binomial distribution given the mean expression of the gene.}
  \item{suppress.plot}{Whether to plot variance vs fold change and dropout rate vs fold change plots.}
}
\details{
	Simulates a single gene that is differentially expressed across two equally sized cell populations (\code{pop_size}) with a constant expected mean expression across all cells. Simulations are performed for fold changes of every integer value from 1 (no DE) to 100. Simulations use a negative binomial distribution inflated with zeros according to a Michaelis-Menten function. 

	Default mean2disp relationship and K were fit to the Buettner et al. [1] mESC data.

	Calculates pearson correlations between observed sample variance and dropout rate and fold change of the differential expression. Sample variance is decomposed into between subpopulation and within subpopulation variance ANOVA. Optionally plots the relationship between observed sample variance and dropout rate and the fold change. 
}
\value{
Name list of values:
	var_r = pearson correlation between observed sample variance and fold change.
	drop_r = pearson correlation between observed dropout rate and fold change.
	vars = vector of observed sample variances
	drops = vector of observed dropout rates
	fc = vector of expected fold changes
	Vbtw = vector of between subpopulation variances
	Vwithin = vector of within subpopulation variances
}
\examples{
#	mu100_N200 <- bg__var_vs_drop(100,100, suppress.plot=FALSE)
#	mu100_N2000 <- bg__var_vs_drop(1000,100)
#	mu1000_N200 <- bg__var_vs_drop(1000,100)
#	c(mu100_N200$var_r,mu100_N2000$var_r,mu1000_N200$var_r)
}
\keyword{simulations, variance, dropouts, differential expression}
\references{
[1] Buettner et al. (2015) Computational analysis of cell-to-cell heterogeneity in single-cell RNA-sequencing data reveals hidden subpopulations of cells. Nature Biotechnology 33 : 155-160.
}

\name{BrenneckeGetVariableGenes}
\alias{BrenneckeGetVariableGenes}
\title{Identify Highly Variable Genes}
\description{
	Implements the method of Brennecke et al. (2013) to identify highly variable genes.
}
\usage{BrenneckeGetVariableGenes(expr_mat, spikes=NA, suppress.plot=FALSE, fdr=0.1, minBiolDisp=0.5, fitMeanQuantile=0.8)}
\arguments{
  \item{expr_mat}{a numeric matrix of normalized or raw (not log-transformed) expression values, columns = samples, rows = genes.}
  \item{spikes}{a vector of gene names of row numbers of spike-in genes which are subject to only technical variance.}
  \item{suppress.plot}{Whether to make the plot or just calculate the requisite values.}
  \item{fdr}{Use FDR to identify significantly highly variable genes.}
  \item{minBiolDisp}{Minimum percentage of variance due to biological factors.}
  \item{fitMeanQuantile}{Threshold for genes to be used in fitting. May need to be decreased in low-depth/umi-tagged datasets to achieve good fit.}
}
\details{Identifies significantly highly variable genes as detailed in Brennecked et al [1]. If spike-ins are provided they are used fit a function to the relationship between gene expression and variance due to technical factors. If spike-ins are not provided then all genes are used in the fitting. }
\value{Vector of names of highly variable genes.}
\references{Brennecke et al. (2013) Accounting for technical noise in single-cell RNA-seq experiments. Nature Methods 10, 1093-1095. doi:10.1038/nmeth.2645}
\examples{
  library(M3DExampleData)
  HVG <- BrenneckeGetVariableGenes(Mmus_example_list$data)
  HVG_spike <- BrenneckeGetVariableGenes(Mmus_example_list$data, spikes=5550:5600)
}
\keyword{highly variable genes}

\name{Consensus_FS}
\title{Consensus Feature Selection}
\alias{Consensus_FS}
\description{
	Performs seven different feature selection methods then calculates the consensus ranking of features from that.
}
\usage{
	Consensus_FS(counts, norm=NA, is.spike=rep(FALSE, times=nrow(counts)), pcs=c(2,3), include_cors=TRUE)
}
\arguments{
  \item{counts}{raw count matrix, rows=genes, cols=cells}
  \item{norm}{normalized but not log-transformed gene expression matrix, rows=genes, cols=cells}
  \item{is.spike}{logical, vector of whether each gene is/isn't a spike-in}
  \item{pcs}{which principle components to use to score genes}
  \item{include_cors}{logical, whether to perform gene-gene correlation feature selection which is much slower than all other methods.}
}
\details{

	Performs:
	\code{NBumiFeatureSelectionCombinedDrop} (aka: DANB_drop)
	\code{NBumiFeatureSelectionHighVar} (aka: DANB_var)
	\code{BrenneckeGetVariableGenes} (aka: HVG)
	\code{M3DropFeatureSelection} (aka: M3Drop)
	\code{giniFS}
	\code{irlbaPcaFS} (with provided PCs)
	\code{corFS} ("both" direction)

	Genes are ranked by each method and the consensus (Cons) is calculated as the average of those ranks.

	Automatically removes invariant genes. If only raw counts are provided then will apply counts per million normalization (scaled to the median library size) for those methods which require normalized data.
	
}
\value{
	Table of ranking of each gene by each method including the consensus (Cons). Columns are feature selection methods named using the shorter aliases (see: Details). 
}

\examples{
library(M3DExampleData)
norm <- as.matrix(Mmus_example_list$data[1:500,]);
norm <- norm[rowSums(norm) > 0,];
counts <- NBumiConvertToInteger(norm);
spikes <- sample(1:nrow(counts), 50);
spikes <- rownames(norm)[spikes];
spikes <- rownames(norm) %in% spikes;
Features_consensus <- Consensus_FS(counts, norm, is.spike=spikes);
}
\keyword{single cell}
\keyword{feature selection}

\name{Fitting_Dropout_Models}
\alias{bg__fit_MM}
\alias{bg__fit_ZIFA}
\alias{bg__fit_logistic}
\title{Fit functions to the dropouts vs expression distribution.}
\description{
	Fits the modified Michaelis-Menten equation (MM), a logistic regession (logistic), or a double exponential (ZIFA) function to the relationship between mean expression and dropout-rate (proportion of zero values).
}
\usage{
  bg__fit_MM(p, s)
  bg__fit_logistic(p, s)
  bg__fit_ZIFA(p, s)
}
\arguments{
  \item{p}{a vector of dropout rates for each gene.}
  \item{s}{a vector of mean expression values for each gene. Must be the same order & length as p.}
}
\details{Fits one of different models to the relationship between dropout rate and mean expression. The three models are: 
		\code{bg__fit_MM} : the Michaelis-Menten function \deqn{P=1-\frac{S}{S+K}}{P = 1 - S/(K+S)} (see: [1]). Fit using \code{mle2} using normally distributed error.
		\code{bg__fit_logistic} : a logistic regression between P and log base 10 of S (used by [2]). Fit using \code{glm} (excludes genes where S == 0).
		\code{bg__fit_ZIFA} : a double exponential \deqn{P = e^{\lambda S^2}}{P = e^(-lambda*S^2)} (used by [3]). Fit using \code{lm} after log-transformation (genes were P == 0 are assigned a value of one tenth of the smallest P which is not 0).
}
\value{Named list including:
	K,fitted_err/B0,B1/lambda,fitted_err : the fitted parameters
	predictions : predicted values of p for each gene
	SSr/SAr :  sum of squared/absolute residuals
	model : vector of string descriptors of the fit
}
\examples{
#  library(M3DExampleData)
#  gene_info = bg__calc_variables(Mmus_example_list$data)
#  MM_fit = bg__fit_MM(gene_info$p, gene_info$s)
#  logistic_fit = bg__fit_logistic(gene_info$p, gene_info$s)
#  ZIFA_fit = bg__fit_ZIFA(gene_info$p, gene_info$s) 
}
\references{
  [1] Keener, J.; Sneyd, J. (2008). Mathematical Physiology: I: Cellular Physiology (2 ed.). Springer. ISBN 978-0-387-75846-6 
  [2] Kharchenko, PV; Silberstein, L; Scadden, DT. (2014) Bayesian approach to single-cell differential expression analysis. Nature Methods. 11:740-742 
  [3] Pierson, E; Yau, C. (2015) ZIFA: Dimensionality reduction for zero-inflated single-cell gene expression analysis. Genome Biology. 16:241 doi:10.1186/s13059-015-0805-z}
\keyword{Michaelis Menten, model fitting}

\name{Imputation}
\title{Normalized Data using the DANB model}
\alias{NBumiImputeNorm}
\description{
	Normalizes data to a common library size, imputing zeros as needed.
}
\usage{
	NBumiImputeNorm(counts, fit, total_counts_per_cell=median(fit$vals$tis))
}
\arguments{
  \item{counts}{raw count matrix, rows=genes, cols=cells}
  \item{fit}{Output from \code{NBumiFitModel}}
  \item{total_counts_per_cell}{reference library size to normalize all cells to.}
}
\details{
	Converts raw counts into positions in the CDF for the depth-adjusted negative binomial model fit to each observation. Adjusts the DANB parameters (mean and size) for the new library size. Then calculate the normalized counts for the equivalent position in the CDF for the NB using the new parameters. 
}
\value{
	Normalized count matrix.
}

\examples{
library(M3DExampleData)
counts <- as.matrix(Mmus_example_list$data);
counts <- counts[rowSums(counts) > 0,];
fit <- NBumiFitModel(counts);
normed_counts <- NBumiImputeNorm(counts, fit, 1000000)
}
\keyword{single cell}
\keyword{normalization}

\name{bg__calc_variables}
\alias{bg__calc_variables}
\title{Calculate Gene Variables}
\description{
	Calculates a suite of gene-specific variables including: mean, dropout rate, and their standard errors.
}
\usage{
	bg__calc_variables(expr_mat)
}
\arguments{
  \item{expr_mat}{a numeric matrix of normalized (not log-transformed) expression values, columns = samples, rows = genes.}
}
\details{ Calculates mean expression and its standard error ( sd/sqrt(n) ), and the dropout rate, proportion of observations that are zero, and its standard error ( sqrt(p*(1-p)/n) ) for each gene in the dataset. Performs various checks to ensure expression matrix is of suitable format for M3Drop. Removes undetected genes if they are present.
}
\value{
	Named list of calculated values:
	s = vector of mean expression for each gene
	s_stderr = vector of mean expression standard error for each gene
	p = vector of dropout rate for each gene
	p_stderr = vector of dropout rate standard error for each gene
}
\keyword{background}


\name{M3DropCleanData}
\alias{M3DropCleanData}
\alias{bg__filter_cells}
\title{Filter Expression Data}
\description{
	Filters and normalizes a given expression matrix. Removes low quality cells and undetected genes, and normalizes counts to counts per million. Functions tagged with "bg__" are not meant for direct usage and are not available in the Bioconductor release.
}
\usage{
  M3DropCleanData(expr_mat, labels = NA, is.counts = TRUE, suppress.plot = FALSE, pseudo_genes = NA, min_detected_genes = NA)
  bg__filter_cells(expr_mat, labels = NA, suppress.plot = FALSE, min_detected_genes = NA)
}
\arguments{
  \item{expr_mat}{a numeric matrix of raw or normalized (not log-transformed) expression values, columns = samples/cells, rows = genes.}
  \item{labels}{a vector of length equal to the number of columns of expr_mat with names or group IDs for each cell.}
  \item{is.counts}{logical, whether the provided data is unnormalized read/fragment counts.}
  \item{suppress.plot}{logical, whether to plot the distribution of number of detected genes per cell.}
  \item{pseudo_genes}{a vector of gene names of known pseudogenes which will be removed from the cleaned data.}
  \item{min_detected_genes}{minimum number of genes/cell for a cell to be included in the cleaned data.}
}
\details{Retains genes detected (expression>0) in more than 3 cells and with mean normalized expression >= 10^-5. If min_detected_genes is defined all cells not reaching the threshold are removed. Otherwise, fits a normal distribution to the distribution of detected genes/cell and removes those cells with significantly few detected genes (FDR 5\%). This fit is plotted for visual inspection. If is.counts==TRUE then each column is converted to counts per million (ignoring ERCC spike-ins if present).}
\value{A list with elements: data, the normalized filtered expression matrix; and labels, labels of the remaining cells.}
\examples{
  library(M3DExampleData)
  # Remove all cells with < 2000 detected genes and convert to cpm 
  cpm <- M3DropCleanData(Mmus_example_list$data, Mmus_example_list$labels, 
			is.counts=TRUE, min_detected_genes=2000)
  # Removes cells with significantly few detected genes (FDR=5%)
  filtered_only <- M3DropCleanData(Mmus_example_list$data, Mmus_example_list$labels, 
			is.counts=FALSE)
#  QCed <- bg__filter_cells(Mmus_example_list$data[,1:10], Mmus_example_list$labels[1:10], suppress.plot=TRUE)
}
\keyword{normalization, quality control}

\name{M3DropConvertData}
\alias{M3DropConvertData}
\title{Convert Data to be suitable for M3Drop}
\description{
	Recognizes a variety of R objects/classes and extracts expression 
	matrices from them then converts that to a normalized but 
	not-log transformed matrix appropriate for input into M3Drop functions.
}
\usage{
  M3DropConvertData(input, is.log=FALSE, is.counts=FALSE, pseudocount=1)
}
\arguments{
  \item{input}{ a matrix, data.frame or object }
  \item{is.log}{ has the data been log-transformed? (assumes log-base 2 with pseudocount of 1) }
  \item{is.counts}{ is the data raw unnormalized counts? (raw counts will be CPM normalized) }
  \item{pseudocount}{ pseudocount added before log-transformation }
}
\details{
	You must have loaded the respective packages (in parentheses) into your namespace before running this function on the respective objects. Note that to maintain scalability sparse matrices will remain as such.

	Supported classes/objects:
		\describe{

		\item{SCESet (scater <= 1.4.0)}{uses "exprs" or if unavailable then "counts"}

		\item{SingleCellExperiment (scater >= 1.6.0)}{uses "normcounts" if available, then "logcounts", which is assumed to be log-normalized, then "counts"}

		\item{CellDataSet (monocle)}{uses "exprs", specify log/counts using arguments}

		\item{ExpressionSet (Biobase)}{uses "exprs", specify log/counts using arguments}

		\item{seurat (Seurat)}{uses "raw.data" as counts.}
		}

		Matrix/Dataframe classes :

		\describe{
		\item{dgCMatrix (Matrix)}{specify log/counts using arguments}

		\item{data.table (data.table)}{specify log/counts using arguments}

		\item{DataTable (S4Vectors)}{specify log/counts using arguments}

		\item{DataFrame (S4Vectors)}{specify log/counts using arguments}

		\item{AnnotatedDataFrame (Biobase)}{specify log/counts using arguments}

		\item{matrix (base-r)}{specify log/counts using arguments}

		\item{data.frame (base-r)}{specify log/counts using arguments}
		}

}
\value{A normalized but not log-transformed matrix appropriate for input into M3Drop functions.}
\examples{
	# Simulated raw count matrix:
	set.seed(42)
	counts <- matrix(rpois(200, lambda=3), ncol=10)
	expr_mat <- M3DropConvertData(counts, is.counts=TRUE)

	# log normalized data frame
	lognorm <-log2( t(t(counts)/colSums(counts)*100)+1 )
	lognorm <- as.data.frame(lognorm)
	expr_mat <- M3DropConvertData(lognorm)

	# Sparse matrix
	require("Matrix")
	counts <- Matrix(counts, sparse=TRUE)
	expr_mat <- M3DropConvertData(counts, is.counts=TRUE)

	# SingleCellExperiment Object
	require("SingleCellExperiment")
	SCE <- SingleCellExperiment(assays=list(counts=counts))
	expr_mat <- M3DropConvertData(SCE)

	# monocle Object
	require("monocle")
	obj <- suppressWarnings(newCellDataSet(as.matrix(lognorm)))
	expr_mat <- M3DropConvertData(obj, is.log=TRUE)
}
\keyword{normalization}

\name{M3DropFeatureSelection}
\alias{M3DropFeatureSelection}
\alias{bg__test_DE_K_equiv}
\title{Differentially Expressed Genes.}
\description{
	Use Michaelis-Menten curve to find differentially expressed (DE) genes. Functions tagged with "bg__" are not meant for direct usage and are not available in the Bioconductor release.
}
\usage{
  M3DropFeatureSelection(expr_mat, mt_method="bon", mt_threshold=0.05, suppress.plot=FALSE, xlim=NA)
  bg__test_DE_K_equiv(gene_info, fit=NA)
}
\arguments{
  \item{expr_mat}{a numeric matrix of normalized (not log-transformed) expression values, columns = samples, rows = genes.}
  \item{mt_method}{the multiple testing method used in p.adjust}
  \item{mt_threshold}{the threshold for identifying significantly DE genes.}
  \item{suppress.plot}{logical, whether to plot the fitted curve and highlight selected features.}
  \item{xlim}{specify the limits of the x-axis of the plot.}
  \item{fit}{Output from fitting the Michaelis-Menten equation (see: \code{bg__fit_MM}}
  \item{gene_info}{List of calculated gene-specific values output by \code{bg__calc_variables}}
}
\details{Fits a Michaelis-Menten function to the dropout-rate (if not provided) of the provided expression matrix. Identifies genes where the gene-specific K calculated as (S = mean expression, P = dropout rate): \deqn{K=\frac{S*P}{1-P}}{K = (S * P)/(1 - P)} is significantly larger than the K fitted to the entire dataset. Combines standard errors of the fitted K, the gene-specific dropout rate and the gene-specific average expression using error propagation rules. Determines the signficance of the gene-specific K using a Z-test of the log-transformed Ks with the propagated error then applies the specified multiple testing correction to identify DE genes. Plots the dropout rate vs gene expression with the fitted MM curve and highlights in purple the significantly DE genes.}
\value{ 
  \code{M3DropFeatureSelection} : a data.frame of significantly differentially expressed genes with columns: Gene, p.value, q.value
  \code{bg__test_DE_K_equiv} : a named list of containing: 
	pval : the significance of differential expression for each gene
	fold_change : ratio of gene-specific K to globally fit K
}
\examples{
	library(M3DExampleData)
	norm <- M3DropConvertData(Mmus_example_list$data, is.counts=TRUE)
	DE_genes <- M3DropFeatureSelection(norm, 
		mt_method="fdr", mt_threshold=0.01)
#  gene_info <- bg__calc_variables(Normalized_data$data[1:1000,])
#  DE_output <- bg__test_DE_K_equiv(gene_info)
}
\keyword{differential expression}

\name{M3DropDropoutModels}
\alias{M3DropDropoutModels}
\title{Fit functions to the dropouts vs expression distribution.}
\description{
	Fits the modified Michaelis-Menten equation (MM), a logistic regession (logistic), or a double exponential (ZIFA) function to the relationship between mean expression and dropout-rate (proportion of zero values).
}
\usage{M3DropDropoutModels(expr_mat, xlim=NA, suppress.plot=FALSE)}
\arguments{
  \item{expr_mat}{a numeric matrix of normalized (not log-transformed) expression values, columns = samples, rows = genes.}
  \item{xlim}{limits for x-axis of plot.}
  \item{suppress.plot}{logical, whether to plot fit curves or not.}
}
\details{Plots the dropout-rate (P) vs average gene expression (S) for all genes. Fits three different models and adds the fitted curves to the plot. The three models are: 
	MMfit : the Michaelis-Menten function \deqn{P=1-\frac{S}{S+K}}{P = 1 - S/(K+S)} (see: [1]).
	LogiFit : a logistic regression between P and log base 10 of S (used by [2]).
	ExpoFit : a double exponential \deqn{P = e^{\lambda S^2}}{P = e^(-lambda*S^2)} (used by [3]).
}
\value{Invisibly, a list of output from each fit (MMfit, LogiFit, ExpoFit).}
\examples{
  library(M3DExampleData)
  norm <- M3DropConvertData(Mmus_example_list$data, is.counts=TRUE)
  M3DropDropoutModels(norm)
} 

\references{
  [1] Keener, J.; Sneyd, J. (2008). Mathematical Physiology: I: Cellular Physiology (2 ed.). Springer. ISBN 978-0-387-75846-6 
  [2] Kharchenko, PV; Silberstein, L; Scadden, DT. (2014) Bayesian approach to single-cell differential expression analysis. Nature Methods. 11:740-742 
  [3] Pierson, E; Yau, C. (2015) ZIFA: Dimensionality reduction for zero-inflated single-cell gene expression analysis. Genome Biology. 16:241 doi:10.1186/s13059-015-0805-z}
\keyword{Michaelis Menten, model fitting}

\name{M3DropExpressionHeatmap}
\alias{M3DropExpressionHeatmap}
\alias{bg__expression_heatmap}
\title{Plot Heatmap of Gene Expression}
\description{
	Plots a customized heatmap of scaled log expression values. Functions tagged with "bg__" are not meant for direct usage and are not available in the Bioconductor release.
}
\usage{
  M3DropExpressionHeatmap(genes, expr_mat, cell_labels=NA, interesting_genes=NA, key_genes=genes, key_cells=NA)
  bg__expression_heatmap(genes, expr_mat, cell_labels=NA, gene_labels=NA, key_genes=genes, key_cells=NA)
}
\arguments{
  \item{genes}{a character vector of gene names to be plot.}
  \item{expr_mat}{a numeric matrix of normalized (not log-transformed) expression values, columns = samples, rows = genes.}
  \item{cell_labels}{factor of labels for each cell in the expression matrix that will be used to colour the bar on the top of the heatmap.}
  \item{interesting_genes}{list of vectors of gene names that will be used to colour the bar to the left of the heatmap.}
  \item{key_genes}{a character vector of gene names to be labelled on the heatmap.}
  \item{key_cells}{a character vector of cells to be labelled on the heatmap. Unlabelled cells will be assigned a numerical index}
  \item{gene_labels}{factor of labels for each gene that will be used to colour the bar on the left of the heatmap.}
}
\details{Modifies the gplots function heatmap.2 to replace the row dendrogram with a legend of the colours used in the columns colour bar (cell_labels) and use a custom colour scalling. Expression is displayed as Z-scores of log transformed expression (adding a pseudocount of 1) coloured blue-white-red centered at 0 and binned in the range [-2,2].

\code{M3DropExpressionHeatmap} is a wrapper for \code{bg__expression_heatmap} that checks and reformats provided arguments. 
}
\value{Invisibly, output from heatmap.2 call.}
\examples{
  library(M3DExampleData)
  M3DropExpressionHeatmap(head(rownames(Mmus_example_list$data),20),Mmus_example_list$data, cell_labels = Mmus_example_list$labels)
#  bg__expression_heatmap(head(rownames(Mmus_example_list$data),20),Mmus_example_list$data, cell_labels = Mmus_example_list$labels)
}
\keyword{heatmap}

\name{M3DropGetExtremes}
\alias{M3DropGetExtremes}
\alias{bg__get_extreme_residuals}
\title{Get outliers from MM curve.}
\description{
	Identifies outliers left and right of a fitted Michaelis-Menten curve. Functions tagged with "bg__" are not meant for direct usage and are not available in the Bioconductor release.
}
\usage{
  M3DropGetExtremes(expr_mat, fdr_threshold=0.1, percent=NA, v_threshold=c(0.05,0.95), suppress.plot=FALSE)
  bg__get_extreme_residuals(expr_mat, fit=NA, fdr_threshold=0.1, percent=NA, v_threshold=c(0.05, 0.95), direction="right", suppress.plot=FALSE)
}
\arguments{
  \item{expr_mat}{a numeric matrix of normalized (not log-transformed) expression values, columns = samples, rows = genes.}
  \item{fit}{output from fitting the Michaelis-Menten equation, see \code{bg__fit_MM}}
  \item{fdr_threshold}{the threshold for identifying significant outliers after multiple testing correction.}
  \item{percent}{identify this percentage of data that is most extreme in each direction.}
  \item{v_threshold}{restrict to this range of dropout rates to avoid poorly fit regions of the data.}
  \item{direction}{"left" or "right", which tail of outliers to examine.}
  \item{suppress.plot}{logical, whether to plot the fitted Michaelis-Menten curve and highlight to identified most extreme outliers.}
}
\details{Fits a Michaelis-Menten function to the dropout-rate of the provided data, then identifies the most extreme left and/or right outliers from the curve. Horizontal residuals are calculated as : \deqn{\log_{10} S - \log_{10} \frac{K*(1-P)}{P}}{log_10(S) - log_10( (K * (1 - P)) / P )}. Extreme left[right] outliers are identified either as the \code{percent} smallest[largest] horizontal residuals. If \code{percent} is undefined (default) a normal distribution is fitted to the horizontal residuals and a Z-test is used to identify significant outliers after FDR multiple testing correction. 

Only genes with dropout rates within v_threshold will be considered to avoid the skewing of residuals due to the exponential parts of the MM curve near P = 0 & P = 1.

\code{M3DropGetExtremes} identifies both left and right residuals using the provided thresholds in each direction. Eg. will return the \code{percent} smallest and \code{percent} largest residuals. It also plots the fitted MM curve and highlights the left and right extreme outliers unless suppress.plot=TRUE .
\code{bg__get_extreme_residuals} identifies one-sided extreme outliers.
}
\value{
  \code{M3DropGetExtremes} List containing elements left and right, vectors of the names of the extreme genes to the left and right of the curve respectively.
  \code{bg__get_extreme_residuals} A vector of names of the extreme genes in the specified direction.
}
\examples{
  library(M3DExampleData)
  norm <- M3DropConvertData(Mmus_example_list$data, is.counts=TRUE)
  extreme_gene_lists <- M3DropGetExtremes(norm, fdr_threshold=0.1)
  extreme_gene_lists <- M3DropGetExtremes(norm, percent=0.01)
#  Lextremes <- bg__get_extreme_residuals(norm, fdr_threshold=0.1, direction="left")
}
\keyword{extremes, outliers, residuals}

\name{M3DropGetHeatmapClusters}
\alias{M3DropGetHeatmapClusters}
\alias{M3DropGetHeatmapNames}
\alias{M3DropGetHeatmapClusters}
\title{Extracts clusters/ordered names from heatmap output}
\description{Extracts the groupings correponding to the given number of clusters from heatmap output.}
\usage{M3DropGetHeatmapClusters(heatout, k, type="cell")
	M3DropGetHeatmapNames(heatout, type="cell")}
\arguments{
  \item{heatout}{Output from a gene-expression heatmap.}
  \item{k}{Number of clusters.}
  \item{type}{One of "cell" or "gene" indicating whether to get clusters or names of cells(columns) or genes(rows).}
}
\details{\code{M3DropGetHeatmapClusters} : Traverses down the row or column dendrogram and cuts at the first point where there are at least k clusters.
	 \code{M3DropGetHeatmapNames}: gets the names of the cells/genes in the order that they appear on the dendrogram.
}
\value{A vector of cluster labels for each cell.}
\examples{
  library(M3DExampleData)
  genes <- rownames(Mmus_example_list$data)[1:20]
  heatmap_out <- M3DropExpressionHeatmap(genes, Mmus_example_list$data)
  clusters <- M3DropGetHeatmapClusters(heatmap_out, k=5)
  heatmap_gene_labels <- M3DropGetHeatmapNames(heatmap_out, type="gene")
}
\keyword{heatmap, cluster}

\name{M3DropGetMarkers}
\alias{M3DropGetMarkers}
\title{Identify marker genes}
\description{
	Calculates area under the ROC curve for each gene to predict the best group of cells from all other cells.
}
\usage{
  M3DropGetMarkers(expr_mat, labels)
}
\arguments{
  \item{expr_mat}{a numeric matrix of normalized expression values, columns = samples, rows = genes.}
  \item{labels}{a vector of group ids for each cell/sample.}
}
\details{Uses the ROCR package to calculate the AUC for each gene for the group with the highest average rank. Significant is calculated using a Wilcox rank-sum test. 
}
\value{
  A dataframe with a row for each gene and columns: AUC, Group (which label this gene had the highest average rank for), and pval (uncorrected p-value of prediction).
}
\examples{
  library(M3DExampleData)
  norm <- M3DropConvertData(Mmus_example_list$data, is.counts=TRUE)
  marker_gene_table <- M3DropGetMarkers(norm, Mmus_example_list$labels)
}
\keyword{markers}

\name{bg__horizontal_residuals_MM_log10}
\alias{bg__horizontal_residuals_MM_log10}
\title{Calculate Horizontal Residuals}
\description{
	Calculates horizontal residuals from the Michaelis-Menten Function. Functions tagged with "bg__" are not meant for direct usage and are not available in the Bioconductor release.
}
\usage{
	bg__horizontal_residuals_MM_log10(K, p, s)
}
\arguments{
  \item{K}{fitted Michaelis-Menten constant.}
  \item{p}{Observed dropout rate.}
  \item{s}{Observed mean expression.}
}
\details{ Calculates the log-transformed horizontal residuals from the Michaelis-Menten function. Input values may be single values or vectors.

\deqn{r = log_{10}(S) - \frac{(1-P)K}{P}}{log10(S) - K*(1-P)/P}
	
}
\value{ Value of horizontal residual.}
\examples{
#  s <- c(10,100,100)
#  p <- c(0.9, 0.5, 0.1)
#  res <- bg__horizontal_residuals_MM_log10(10, p, s)
}
\keyword{background}

\name{M3DropPlottingFunctions}
\alias{bg__dropout_plot_base}
\alias{bg__add_model_to_plot}
\alias{bg__highlight_genes}
\title{Make M3Drop Plots}
\description{
	Background functions used for making M3Drop plots. Functions tagged with "bg__" are not meant for direct usage and are not available in the Bioconductor release.
}
\usage{
	bg__dropout_plot_base(expr_mat, xlim=NA, suppress.plot=FALSE)
	bg__add_model_to_plot(fitted_model, base_plot, lty=1, lwd=1, col="dodgerblue", legend_loc="topright")
	bg__highlight_genes(base_plot, expr_mat, genes, col="darkorange", pch=16)
}
\arguments{
  \item{expr_mat}{a numeric matrix of normalized (not log-transformed) expression values, columns = samples, rows = genes.}
  \item{xlim}{limits of x-axis of plot, for comparing distributions across many datasets.}
  \item{suppress.plot}{Whether to make the plot or just return values.}
  \item{fitted_model}{output from fitting a model to dropout rate vs mean expression, see \code{bg__fit_MM},\code{bg__fit_ZIFA},\code{bg__fit_logistic}}
  \item{base_plot}{output from \code{bg__dropout_plot_base}}
  \item{legend_loc}{coordinates of top-right corner of the legend.}
  \item{genes}{list of genes to be highlighted.}
  \item{col,lty,lwd,pch}{Graphical parameters passed to plotted points & lines respectively.}
}
\details{ 
\code{bg__dropout_plot_base} plots grey-scale distribution of dropout-rate vs log10(mean expression) for each each. Colour indicates density of points light-grey = low, black = high.

\code{bg__add_model_to_plot} adds a line based on the predicted dropout rate for each gene from a particular model.

\code{bg__highlight_genes} highlights specified genes in the given colour.

}
\value{
\code{bg__dropout_plot_base} A named list of output:
	p = dropout rate for each gene
	s = mean expression of each gene
	xes = log10 transformed mean expression of each gene
	data = original expression matrix
	order = ordering of xes from smallest to largest
\code{bg__add_model_to_plot} Invisibly location of the plotted legend.
\code{bg__highlight_genes} Invisibly the coordinates of the highlighted genes.
}
\keyword{plotting, background}

\name{bg__MakeSimData}
\alias{bg__MakeSimData}
\alias{bg__MakeSimDE}
\alias{bg__MakeSimDVar}
\alias{bg__MakeSimHVar}
\title{Make Simulated Data}
\description{
	Makes simulated data based on a negative binomial distribution inflated with zeros based on the Michaelis-Menten equation.
}
\usage{
	bg__MakeSimData(dispersion_fun=bg__default_mean2disp, n_cells=300, dispersion_factor=1, base_means=10^rnorm(25000, 1, 1), K=10.3)
	bg__MakeSimDE(dispersion_fun=bg__default_mean2disp, fold_change=10, frac_change=0.1, n_cells=300, sub_pop=0.5, dispersion_factor=1, base_means=10^rnorm(25000,1,1), K=10.3)
	bg__MakeSimDVar(dispersion_fun=bg__default_mean2disp, fold_change=10, frac_change=0.1, n_cells=300, sub_pop=0.5, dispersion_factor=1, base_means=10^rnorm(25000,1,1), K=10.3)
	bg__MakeSimHVar(dispersion_fun=bg__default_mean2disp, fold_change=10, frac_change=0.1, n_cells=300, dispersion_factor=1, base_means=10^rnorm(25000,1,1), K=10.3)
}
\arguments{
  \item{dispersion_fun}{a function which takes mean experssion and returns the dispersion parameter of the negative binomial distribution.}
  \item{n_cells}{total number of cells (columns) in the simulated dataset.}
  \item{sub_pop}{proportion of cells with changed expression.}
  \item{frac_change}{proportion of genes with changed expression.}
  \item{fold_change}{fold change in dispersion or mean expression.}
  \item{dispersion_factor}{a factor that multiplies the calculated mean-specific dispersion for all genes.}
  \item{base_means}{a vector of background mean expression values.}
  \item{K}{K of the Michaelis-Menten function}
}
\details{Generates simulated single-cell gene expression data using a zero-inflated negative binomial distribution. A user-supplied function relates the dispersion parameter (1/size of the R parameterization of the negative binomial distribution). Zeros are added based on a Michaelis-Menten function. 

Default values of base_means, K, and dispersion_fun were fit to the Buettner et al. 2015 data [1].


\code{bg__MakeSimData} generates simulated single-cell data for a single homogeneous population.

\code{bg__MakeSimDE} generates simulated single-cell data for two different populations where a proportion of genes have a \code{fold_change} difference in the mean for population "2".

\code{bg__MakeSimDVar} generates simulated single-cell data for two different populations where a proportion of genes have a \code{fold_change} difference in the dispersion for population "2".

\code{bg__MakeSimHVar} generates simulated single-cell data for a single homogeneous population where a proportion of genes have a \code{fold_change} increase in dispersion over the expectation given the mean expression of the gene.
}
\value{
	\code{bg__MakeSimData} : a gene expression matrix where rows are genes, columns are cells.
	\code{bg__MakeSimDE}, \code{bg__MakeSimDVar}, \code{bg__MakeSimHVar} : a list of three named items:
		data : the gene expression matrix where rows are genes, columns are cells
		cell_labels : a vector of 1 or 2 indicating which cells are the unchanged ("1") or changed ("2") population.
		TP : a vector of row IDs of those genes that change (true positives).
}
\examples{
#  means = c(1,2,5,10,20,50,100,200,500,1000,2000,5000)
#  population1 <- bg__MakeSimData(n_cells=10, base_means=means)
#  population2 <- bg__MakeSimData(n_cells=10, base_means=means*2, dispersion_factor=0.5)
#  sim_DE <- bg__MakeSimDE(n_cells=100, base_means=means)
#  sim_DVar <- bg__MakeSimDVar(n_cells=100, sub_pop=0.25, base_means=means)
#  sim_HVar <- bg__MakeSimHVar(base_means=means, fold_change=3)
}
\keyword{simulations, differential expression, differential variance}
\references{
[1] Buettner et al. (2015) Computational analysis of cell-to-cell heterogeneity in single-cell RNA-sequencing data reveals hidden subpopulations of cells. Nature Biotechnology 33 : 155-160.
}

\name{M3DropTestShift}
\alias{M3DropTestShift}
\title{Test for horizontal shift.}
\description{
	Tests whether a given set of genes are significantly shifted to the left or right of the Michaelis-Menten curve.
}
\usage{
  M3DropTestShift(expr_mat, genes_to_test, name="", background=rownames(expr_mat), suppress.plot=FALSE)
}
\arguments{
  \item{expr_mat}{a numeric matrix of normalized (not log-transformed) expression values, columns = samples, rows = genes.}
  \item{genes_to_test}{vector of gene names to test.}
  \item{name}{string used to title the plot.}
  \item{background}{vector of gene names to test against. (default = all genes)}
  \item{suppress.plot}{logical, whether to the fitted Michaelis-Menten curve and highlight the given set of genes to test.}
}
\details{Fits a Michaelis-Menten function to the dropout-rate of the provided data, then tests whether a given set of genes (eg. pseudogenes) is significantly shifted left or right of the curve. Horizontal residuals are calculated as : \deqn{\log_{10} S - \log_{10} \frac{K*(1-P)}{P}}{log_10(S) - log_10( (K * (1 - P)) / P )}. Uses a Wilcox rank-sum test/Mann-Whitney U test to compare the residuals for the given genes to the residuals for all genes.
}
\value{
A one row dataframe with columns: 
    sample (median horizontal residual of genes in the test set), 
    pop (median horizontal residual of genes in the background set), 
    p.value
}
\examples{
  library(M3DExampleData)
  gene_set <- c("Dppa2","Tdgf1","Rnf130","Tet1","Uhrf1","Pttg1","Zfp600","Stat1")
  shift_output <- M3DropTestShift(Mmus_example_list$data, gene_set)
}
\keyword{residuals}

\name{M3DropThreeSetVenn}
\alias{M3DropThreeSetVenn}
\title{Three-way Venn Diagram}
\description{
	Plot an area-proportional three-set Venn Diagram with labels.
}
\usage{
  M3DropThreeSetVenn(set1, set2, set3, names)
}
\arguments{
  \item{set1}{a vector of items in the first set.}
  \item{set2}{a vector of items in the second set.}
  \item{set3}{a vector of items in the third set.}
  \item{names}{a vector of names of each set}
}
\details{Approximates area-proportional three-set Venn Diagram with code by David J. States (available at: http://tolstoy.newcastle.edu.au/R/help/03a/1115.html). Then places labels within each circle and overlap-region using code by Tallulah Andrews. 
}
\value{None}
\examples{
  SetA <- c(1:20)
  SetB <- c(15:30)
  SetC <- c(5,10,15,20,25,30,35,40,45,50,55,60)
  M3DropThreeSetVenn(SetA, SetB, SetC, names=c("A","B","C"))
}
\keyword{Venn Diagram}

\name{M3DropTraditionalDE}
\alias{bg__m3dropTraditionalDE}
\alias{bg__m3dropTraditionalDEShiftDisp}
\alias{bg__fitdispersion}
\alias{bg__get_mean2disp}
\title{Perform Traditional Differential Expression}
\description{
	Performs traditional (i.e. compare defined groups) differential expression using a negative binomial model with MM zero-inflation. Functions tagged with "bg__" are not meant for direct usage and are not available in the Bioconductor release.
}
\usage{
	bg__get_mean2disp(expr_mat)
	bg__fitdispersion(expr_mat)
}
\arguments{
  \item{expr_mat}{a numeric matrix of library-size normalized expression values, columns = samples, rows = genes.}
}
\details{
THESE FUNCTIONS SHOULD NOT BE USED.

\code{unfinished__m3dropTraditionalDE} : Uses a log-likelihood ratio test to perform model selection between a model of constant mean expression vs a model of different mean expression across the biological groups. 
Probabilities of observing the data given the model are calculated using a zero-inflated negative binomial distribution. Global relationships between mean and dispersion (power-law) as well as mean and 
dropouts (Michaelis-Menten) for genes are fit independently for each batch. Dispersions are fixed for each batch and calculated using the fitted power-law using the global mean expression of each gene.
Significance is evaluated using the chi-square distribution.

\code{unfinished__m3dropTraditionalDEShiftDisp}: Uses a log-likelihood ratio test to perform model selection between a model of constant mean expression vs a model of different mean expression across the biological groups. 
Probabilities of observing the data given the model are calculated using a zero-inflated negative binomial distribution. Global relationships between mean and dispersion (power-law) as well as mean and 
dropouts (Michaelis-Menten) for genes are fit independently for each batch. Dispersions are shifted from the global variance according to the mean expression for each biological group, using batch-specific power-laws.
Significance is evaluated using the chi-square distribution.


\code{broken__m3dropCTraditionalDE} : Uses a log-likelihood ratio test to perform model selection between a model of constant mean expression vs a model of different mean expression across the biological groups. 
Probabilities of observing the data given the model are calculated using a zero-inflated negative binomial distribution. Global relationships between mean and dispersion (power-law) as well as mean and 
dropouts (Michaelis-Menten) for genes are fit to the full dataset. Significance is evaluated using the chi-square distribution.

\code{bg__get_mean2disp} fits a power-law relationship between the squared coefficient of variation and mean expression of each gene, which is used to calculate the expected dispersion parameter for the negative binomial distribution from a given mean expression value.

\code{bg__fitdispersion} estimates gene-specific dispersions from the mean and variance of gene expression values \deqn{r = \frac{\mu^2}{\sigma^2-\mu}}{r = mu^2/(var-mu)}. Then fits a power-law relationship between the estimated dispersion and mean exprssion.
}

\value{
	\code{bg__m3dropTraditionalDE} : a table of observed mean expression levels for each biological group and each batch with raw p-values and FDR corrected p-values for each gene.
	\code{bg__m3dropTraditionalDEShiftDisp} : a table of observed mean expression levels for each biological group and each batch with raw p-values and FDR corrected p-values for each gene.
	\code{broken__m3dropCTraditionalDE} : a table of observed mean expression levels for each biological group with raw p-values and FDR corrected p-values for each gene.
	\code{bg__get_mean2disp} : a function which calculates the expected dispersion given the mean expression.
	\code{bg__fitdispersion} : exponent of the power-law relationship between dispersion and mean expression.
}
\examples{
library(M3DExampleData)
#Normalized_data <- M3DropCleanData(Mmus_example_list$data,
#                  labels = Mmus_example_list$labels,
#                  is.counts=TRUE, min_detected_genes=2000)
#DE_output <- bg__m3dropTraditionalDE(Normalized_data$data[1:100,], Normalized_data$labels)
#DE_shifted_output <- bg__m3dropTraditionalDEShiftDisp(Normalized_data$data[1:100,], Normalized_data$labels)
#DE_output_batches <- bg__m3dropTraditionalDE(Normalized_data$data[1:100,], Normalized_data$labels, batches=Normalized_data$labels) # each biological condition was performed separately
}
\keyword{differential expression}

\name{NBumiCheckFit}
\alias{NBumiCheckFit}
\alias{NBumiCheckFitFS}
\title{Check Fit Quality}
\description{
	Checks the quality of the fit of the depth-adjusted negative binomial model.
}
\usage{
	NBumiCheckFit(counts, fit, suppress.plot=FALSE)
	NBumiCheckFitFS(counts, fit, suppress.plot=FALSE)
}
\arguments{
  \item{counts}{a numeric matrix of raw UMI counts, columns = samples, rows = genes.}
  \item{fit}{output from NBumiFitModel or NBumiFitBasicModel.}
  \item{suppress.plot}{Whether to plot the observed vs expected frequency of dropouts for genes & cells.}
}
\details{
	Calculates expected dropouts for genes and cells and compares to observed values. Optionally plots observed vs expected dropouts for both genes and cells.
	\code{NBumiCheckFit} uses depth-adjusted negative binomial with gene-specific dispersions.
	\code{NBumiCheckFitFS} uses depth-adjusted negative binomial with dispersions calculated from the power-law between gene-specific dispersion and mean expression (as is used for feature selection).
}
\value{
	Invisibly, named list of output: 
		gene_error = sum of squared error between observed and expected gene-specific total dropouts
		cell_error = sum of squared error between observed and expected cell-specific total dropouts
		exp_ps = gene by sample matrix of probability of a dropout for the negative binomial model fitted to each observation.
}
\examples{
	library(M3DExampleData)
	counts <- NBumiConvertData(Mmus_example_list$data)
	fit <- NBumiFitModel(counts);
	fit_quality <- NBumiCheckFitFS(counts, fit);
}
