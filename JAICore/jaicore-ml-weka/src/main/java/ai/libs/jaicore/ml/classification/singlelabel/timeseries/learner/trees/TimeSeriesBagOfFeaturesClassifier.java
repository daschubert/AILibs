package ai.libs.jaicore.ml.classification.singlelabel.timeseries.learner.trees;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.aeonbits.owner.ConfigCache;
import org.api4.java.ai.ml.core.exception.PredictionException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.jaicore.basic.IOwnerBasedRandomizedAlgorithmConfig;
import ai.libs.jaicore.basic.sets.Pair;
import ai.libs.jaicore.ml.classification.singlelabel.timeseries.dataset.TimeSeriesDataset2;
import ai.libs.jaicore.ml.classification.singlelabel.timeseries.dataset.TimeSeriesFeature;
import ai.libs.jaicore.ml.classification.singlelabel.timeseries.learner.ASimplifiedTSClassifier;
import ai.libs.jaicore.ml.classification.singlelabel.timeseries.learner.trees.TimeSeriesBagOfFeaturesLearningAlgorithm.ITimeSeriesBagOfFeaturesConfig;
import ai.libs.jaicore.ml.classification.singlelabel.timeseries.util.TimeSeriesUtil;
import ai.libs.jaicore.ml.classification.singlelabel.timeseries.util.WekaTimeseriesUtil;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

/**
 * Implementation of the Time Series Bag-of-Features (TSBF) classifier as
 * described in Baydogan, Mustafa & Runger, George & Tuv, Eugene. (2013). A
 * Bag-of-Features Framework to Classify Time Series. IEEE Transactions on
 * Pattern Analysis and Machine Intelligence. 35. 2796-802.
 * 10.1109/TPAMI.2013.72.
 *
 * This classifier only supports univariate time series prediction.
 *
 * @author Julian Lienen
 *
 */
public class TimeSeriesBagOfFeaturesClassifier extends ASimplifiedTSClassifier<Integer> {

	/**
	 * Log4j logger.
	 */
	private static final Logger LOGGER = LoggerFactory.getLogger(TimeSeriesBagOfFeaturesClassifier.class);

	/**
	 * Random Forest classifier used for the internal OOB probability estimation.
	 */
	private RandomForest subseriesClf;

	/**
	 * Random Forest classifier used for the final class prediction.
	 */
	private RandomForest finalClf;

	/**
	 * Number of total classes used within training.
	 */
	private int numClasses;

	/**
	 * Intervals of each subsequence storing the start and exclusive end index. It
	 * is used for feature generation.
	 */
	private int[][][] intervals;

	/**
	 * Subsequences storing the start and exclusive end index. Used for feature
	 * generation.
	 */
	private int[][] subsequences;

	private final ITimeSeriesBagOfFeaturesConfig config;

	/**
	 * Standard constructor using the default parameters (numBins = 10, numFolds =
	 * 10, zProp = 0.1, minIntervalLength = 5) for the TSBF classifier.
	 *
	 * @param seed
	 *            Seed used for randomized operations
	 */
	public TimeSeriesBagOfFeaturesClassifier(final int seed) {
		this(seed, 10, 10, 0.1d, 5, false);
	}

	/**
	 * Constructor specifying parameters (cf.
	 * {@link TimeSeriesBagOfFeaturesClassifier#TimeSeriesBagOfFeaturesClassifier(int)}).
	 *
	 * @param seed
	 *            Seed used for randomized operations
	 * @param numBins
	 *            See {@link TimeSeriesBagOfFeaturesClassifier#numBins}
	 * @param numFolds
	 *            Number of folds for the internal OOB probability CV estimation
	 * @param zProp
	 *            Proportion of the total time series length to be used for the
	 *            subseries generation
	 * @param minIntervalLength
	 *            The minimal interval length used for the interval generation
	 */
	public TimeSeriesBagOfFeaturesClassifier(final int seed, final int numBins, final int numFolds, final double zProp, final int minIntervalLength) {
		this(seed, numBins, numFolds, zProp, minIntervalLength, false);
	}

	/**
	 * Constructor specifying parameters (cf.
	 * {@link TimeSeriesBagOfFeaturesClassifier#TimeSeriesBagOfFeaturesClassifier(int)}).
	 *
	 * @param seed
	 *            Seed used for randomized operations
	 * @param numBins
	 *            See {@link TimeSeriesBagOfFeaturesClassifier#numBins}
	 * @param numFolds
	 *            Number of folds for the internal OOB probability CV estimation
	 * @param zProp
	 *            Proportion of the total time series length to be used for the
	 *            subseries generation
	 * @param minIntervalLength
	 *            The minimal interval length used for the interval generation
	 * @param useZNormalization
	 *            Indicator whether the Z normalization should be used
	 */
	public TimeSeriesBagOfFeaturesClassifier(final int seed, final int numBins, final int numFolds, final double zProp, final int minIntervalLength, final boolean useZNormalization) {
		this.config = ConfigCache.getOrCreate(ITimeSeriesBagOfFeaturesConfig.class);
		this.config.setProperty(IOwnerBasedRandomizedAlgorithmConfig.K_SEED, "" + seed);
		this.setNumBins(numBins);
		this.config.setProperty(ITimeSeriesBagOfFeaturesConfig.K_NUMFOLDS, "" + numFolds);
		this.config.setProperty(ITimeSeriesBagOfFeaturesConfig.K_ZPROP, "" + zProp);
		this.config.setProperty(ITimeSeriesBagOfFeaturesConfig.K_MIN_INTERVAL_LENGTH, "" + minIntervalLength);
		this.config.setProperty(ITimeSeriesBagOfFeaturesConfig.K_USE_ZNORMALIZATION, "" + useZNormalization);
	}

	/**
	 * Method predicting the class of the given <code>univInstance</code>. At first,
	 * an internal feature representation using a bag of features is generated by
	 * the previously trained {@link TimeSeriesBagOfFeaturesClassifier#subsequences}
	 * and {@link TimeSeriesBagOfFeaturesClassifier#intervals}. These internal
	 * instances are used to get an internal class probability estimation for each
	 * subsequence and interval for each instance using a Random Forest classifier.
	 * These probabilities are aggregated to a histogram which is then fed to a
	 * final Random Forest classifier predicting the instance's target class.
	 */
	@Override
	public Integer predict(double[] univInstance) throws PredictionException {
		if (!this.isTrained()) {
			throw new PredictionException("Model has not been built before!");
		}

		// Z-Normalize if enabled
		if (this.config.zNormalization()) {
			univInstance = TimeSeriesUtil.zNormalize(univInstance, true);
		}

		// Generate features and interval instances
		double[][] intervalFeatures = new double[this.intervals.length][(this.intervals[0].length + 1) * 3 + 2];

		for (int i = 0; i < this.intervals.length; i++) {
			// Feature generation for each interval
			for (int j = 0; j < this.intervals[i].length; j++) {
				double[] tmpFeatures = TimeSeriesFeature.getFeatures(univInstance, this.intervals[i][j][0], this.intervals[i][j][1] - 1, TimeSeriesBagOfFeaturesLearningAlgorithm.USE_BIAS_CORRECTION);
				intervalFeatures[i][j * 3] = tmpFeatures[0];
				intervalFeatures[i][j * 3 + 1] = tmpFeatures[1] * tmpFeatures[1];
				intervalFeatures[i][j * 3 + 2] = tmpFeatures[2];
			}

			// Feature generation for each subseries itself
			double[] subseriesFeatures = TimeSeriesFeature.getFeatures(univInstance, this.subsequences[i][0], this.subsequences[i][1] - 1, TimeSeriesBagOfFeaturesLearningAlgorithm.USE_BIAS_CORRECTION);
			intervalFeatures[i][this.intervals[i].length * 3] = subseriesFeatures[0];
			intervalFeatures[i][this.intervals[i].length * 3 + 1] = subseriesFeatures[1] * subseriesFeatures[1];
			intervalFeatures[i][this.intervals[i].length * 3 + 2] = subseriesFeatures[2];

			// Add start and end indices of subseries to features
			intervalFeatures[i][intervalFeatures[i].length - 2] = this.subsequences[i][0];
			intervalFeatures[i][intervalFeatures[i].length - 1] = this.subsequences[i][1];
		}

		// Prepare Weka instances for generated features
		Instances subseriesInstances = WekaTimeseriesUtil.simplifiedTimeSeriesDatasetToWekaInstances(TimeSeriesUtil.createDatasetForMatrix(intervalFeatures),
				IntStream.rangeClosed(0, this.numClasses - 1).boxed().map(String::valueOf).collect(Collectors.toList()));

		// Predict probabilities using the subseries Random Forest classifier
		double[][] probs = null;
		int[] predictedTargets = new int[subseriesInstances.numInstances()];
		try {
			probs = this.subseriesClf.distributionsForInstances(subseriesInstances);
			for (int i = 0; i < subseriesInstances.numInstances(); i++) {
				predictedTargets[i] = (int) this.subseriesClf.classifyInstance(subseriesInstances.get(i));
			}
		} catch (Exception e) {
			throw new PredictionException("Cannot derive the probabilities using the subseries classifier due to an internal Weka exception.", e);
		}

		// Discretize probabilities and create histograms for final Weka instance
		int[][] discretizedProbs = TimeSeriesBagOfFeaturesLearningAlgorithm.discretizeProbs(this.getNumBins(), probs);
		Pair<int[][][], int[][]> histFreqPair = TimeSeriesBagOfFeaturesLearningAlgorithm.formHistogramsAndRelativeFreqs(discretizedProbs, 1, this.numClasses, this.getNumBins());
		int[][][] histograms = histFreqPair.getX();
		int[][] relativeFrequencies = histFreqPair.getY();

		// Prepare final Weka instance
		double[][] finalHistogramInstances = TimeSeriesBagOfFeaturesLearningAlgorithm.generateHistogramInstances(histograms, relativeFrequencies);
		Instances finalInstances = WekaTimeseriesUtil.simplifiedTimeSeriesDatasetToWekaInstances(TimeSeriesUtil.createDatasetForMatrix(finalHistogramInstances),
				IntStream.rangeClosed(0, this.numClasses - 1).boxed().map(String::valueOf).collect(Collectors.toList()));

		// Ensure that only on instance has been generated out of the given
		// probabilities
		if (finalInstances.size() != 1) {
			final String errorMessage = "There should be only one instance given to the final Random Forest classifier.";
			throw new PredictionException(errorMessage, new IllegalStateException(errorMessage));
		}

		// Predict using the generated Weka instance
		try {
			return (int) this.finalClf.classifyInstance(finalInstances.firstInstance());
		} catch (Exception e) {
			throw new PredictionException("Could not predict instance due to an internal Weka exception.", e);
		}
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public Integer predict(final List<double[]> multivInstance) throws PredictionException {
		LOGGER.warn("Dataset to be predicted is multivariate but only first time series (univariate) will be considered.");

		return this.predict(multivInstance.get(0));
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public List<Integer> predict(final TimeSeriesDataset2 dataset) throws PredictionException {
		if (!this.isTrained()) {
			throw new PredictionException("Model has not been built before!");
		}

		// Uses the prediction of single instances
		final List<Integer> result = new ArrayList<>();
		for (int i = 0; i < dataset.getValues(0).length; i++) {
			result.add(this.predict(dataset.getValues(0)[i]));
		}
		return result;
	}

	/**
	 * @return the subseriesClf
	 */
	public RandomForest getSubseriesClf() {
		return this.subseriesClf;
	}

	/**
	 * @param subseriesClf
	 *            the subseriesClf to set
	 */
	public void setSubseriesClf(final RandomForest subseriesClf) {
		this.subseriesClf = subseriesClf;
	}

	/**
	 * @return the finalClf
	 */
	public RandomForest getFinalClf() {
		return this.finalClf;
	}

	/**
	 * @param finalClf
	 *            the finalClf to set
	 */
	public void setFinalClf(final RandomForest finalClf) {
		this.finalClf = finalClf;
	}

	/**
	 * @return the numBins
	 */
	public int getNumBins() {
		return this.config.numBins();
	}

	/**
	 * @param numBins
	 *            the numBins to set
	 */
	public void setNumBins(final int numBins) {
		this.config.setProperty(ITimeSeriesBagOfFeaturesConfig.K_NUMBINS, "" + numBins);
	}

	/**
	 * @return the numClasses
	 */
	public int getNumClasses() {
		return this.numClasses;
	}

	/**
	 * @param numClasses
	 *            the numClasses to set
	 */
	public void setNumClasses(final int numClasses) {
		this.numClasses = numClasses;
	}

	/**
	 * @return the intervals
	 */
	public int[][][] getIntervals() {
		return this.intervals;
	}

	/**
	 * @param intervals
	 *            the intervals to set
	 */
	public void setIntervals(final int[][][] intervals) {
		this.intervals = intervals;
	}

	/**
	 * @return the subsequences
	 */
	public int[][] getSubsequences() {
		return this.subsequences;
	}

	/**
	 * @param subsequences
	 *            the subsequences to set
	 */
	public void setSubsequences(final int[][] subsequences) {
		this.subsequences = subsequences;
	}

	@Override
	public TimeSeriesBagOfFeaturesLearningAlgorithm getLearningAlgorithm(final TimeSeriesDataset2 dataset) {
		return new TimeSeriesBagOfFeaturesLearningAlgorithm(this.config, this, dataset);
	}

	public ITimeSeriesBagOfFeaturesConfig getConfig() {
		return this.config;
	}
}
