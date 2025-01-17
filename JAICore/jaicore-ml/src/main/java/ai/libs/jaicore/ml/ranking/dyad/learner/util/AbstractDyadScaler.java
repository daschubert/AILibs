package ai.libs.jaicore.ml.ranking.dyad.learner.util;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.api4.java.ai.ml.ranking.dyad.dataset.IDyad;
import org.api4.java.ai.ml.ranking.dyad.dataset.IDyadRankingDataset;
import org.api4.java.ai.ml.ranking.dyad.dataset.IDyadRankingInstance;
import org.api4.java.common.math.IVector;

import ai.libs.jaicore.ml.ranking.dyad.dataset.DenseDyadRankingInstance;
import ai.libs.jaicore.ml.ranking.dyad.dataset.DyadRankingDataset;
import ai.libs.jaicore.ml.ranking.dyad.dataset.SparseDyadRankingInstance;

/**
 * A scaler that can be fit to a certain dataset and then be used to standardize
 * datasets, i.e. transform the data to have a mean of 0 and a standard
 * deviation of 1 according to the data it was fit to.
 *
 * @author Michael Braun, Jonas Hanselle, Mirko Jürgens, Helena Graf
 *
 */
public abstract class AbstractDyadScaler implements Serializable {

	/**
	 *
	 */
	private static final long serialVersionUID = -825893010030419116L;

	protected SummaryStatistics[] statsX;
	protected SummaryStatistics[] statsY;

	public SummaryStatistics[] getStatsX() {
		return this.statsX;
	}

	public SummaryStatistics[] getStatsY() {
		return this.statsY;
	}

	/**
	 * Fits the standard scaler to the dataset.
	 *
	 * @param dataset The dataset the scaler should be fit to.
	 */
	public void fit(final IDyadRankingDataset dataset) {
		int lengthX = dataset.get(0).getLabel().get(0).getContext().length();
		int lengthY = dataset.get(0).getLabel().get(0).getAlternative().length();
		this.statsX = new SummaryStatistics[lengthX];
		this.statsY = new SummaryStatistics[lengthY];
		for (int i = 0; i < lengthX; i++) {
			this.statsX[i] = new SummaryStatistics();
		}
		for (int i = 0; i < lengthY; i++) {
			this.statsY[i] = new SummaryStatistics();
		}
		for (IDyadRankingInstance instance : dataset) {
			for (IDyad dyad : instance) {
				for (int i = 0; i < lengthX; i++) {
					this.statsX[i].addValue(dyad.getContext().getValue(i));
				}
				for (int i = 0; i < lengthY; i++) {
					this.statsY[i].addValue(dyad.getAlternative().getValue(i));
				}
			}
		}
	}

	/**
	 * Transforms the entire dataset according to the mean and standard deviation of
	 * the data the scaler has been fit to.
	 *
	 * @param dataset The dataset to be standardized.
	 */
	public void transform(final IDyadRankingDataset dataset) {
		int lengthX = dataset.get(0).getLabel().get(0).getContext().length();
		int lengthY = dataset.get(0).getLabel().get(0).getAlternative().length();

		if (lengthX != this.statsX.length || lengthY != this.statsY.length) {
			throw new IllegalArgumentException("The scaler was fit to dyads with instances of length " + this.statsX.length + " and alternatives of length " + this.statsY.length + "\n but received instances of length " + lengthX
					+ " and alternatives of length " + lengthY);
		}

		this.transformInstances(dataset);
		this.transformAlternatives(dataset);
	}

	/**
	 * Transforms only the instances of each dyad according to the mean and standard
	 * of the data the scaler has been fit to.
	 *
	 * @param dataset The dataset of which the instances are to be standardized.
	 */
	public void transformInstances(final IDyadRankingDataset dataset) {
		this.transformInstances(dataset, new ArrayList<>());
	}

	/**
	 * Transforms only the alternatives of each dyad according to the mean and
	 * standard deviation of the data the scaler has been fit to.
	 *
	 * @param dataset The dataset of which the alternatives are to be standardized.
	 */
	public void transformAlternatives(final IDyadRankingDataset dataset) {
		this.transformAlternatives(dataset, new ArrayList<>());
	}

	/**
	 * Transforms only the instances of each dyad according to the mean and standard
	 * deviation of the data the scaler has been fit to. The attributes with indices
	 * contained in ignoredIndices are not transformed. {
	 *
	 * @param dataset The dataset of which the alternatives are to be
	 *            standardized.
	 * @param ignoredIndices The {@link List} of indices that are been ignored by
	 *            the scaler.
	 */
	public abstract void transformInstances(IDyad dyad, List<Integer> ignoredIndices);

	/**
	 * Transforms only the alternatives of each dyad according to the mean and
	 * standard deviation of the data the scaler has been fit to.
	 *
	 * @param dataset The dataset of which the alternatives are to be
	 *            standardized.
	 * @param ignoredIndices The {@link List} of indices that are been ignored by
	 *            the scaler.
	 */
	public abstract void transformAlternatives(IDyad dyad, List<Integer> ignoredIndices);

	/**
	 * Transforms an instance feature vector.
	 *
	 * @param Instance vector to be transformed
	 * @param ignoredIndices
	 */
	public abstract void transformInstaceVector(IVector vector, List<Integer> ignoredIndices);

	/**
	 * Transforms only the instances of each dyad in a
	 * {@link SparseDyadRankingInstance} according to the mean and standard
	 * deviation of the data the scaler has been fit to. The attributes with indices
	 * contained in ignoredIndices are not transformed. {
	 *
	 * @param dataset The dataset of which the alternatives are to be
	 *            standardized.
	 * @param ignoredIndices The {@link List} of indices that are been ignored by
	 *            the scaler.
	 */
	public void transformInstances(final SparseDyadRankingInstance drInstance, final List<Integer> ignoredIndices) {
		this.transformInstaceVector(drInstance.getContext(), ignoredIndices);
	}

	/**
	 * Transforms only the instances of each dyad in a
	 * {@link DenseDyadRankingInstance} according to the mean and standard
	 * deviation of the data the scaler has been fit to. The attributes with indices
	 * contained in ignoredIndices are not transformed. {
	 *
	 * @param dataset The dataset of which the alternatives are to be
	 *            standardized.
	 * @param ignoredIndices The {@link List} of indices that are been ignored by
	 *            the scaler.
	 */
	public void transformInstances(final IDyadRankingInstance drInstance, final List<Integer> ignoredIndices) {
		for (IDyad dyad : drInstance) {
			this.transformInstances(dyad, ignoredIndices);
		}
	}

	/**
	 * Transforms only the alternatives of each dyad in an
	 * {@link IDyadRankingInstance} according to the mean and standard
	 * deviation of the data the scaler has been fit to. The attributes with indices
	 * contained in ignoredIndices are not transformed. {
	 *
	 * @param dataset The dataset of which the alternatives are to be
	 *            standardized.
	 * @param ignoredIndices The {@link List} of indices that are been ignored by
	 *            the scaler.
	 */
	public void transformAlternatives(final IDyadRankingInstance drInstance, final List<Integer> ignoredIndices) {
		for (IDyad dyad : drInstance) {
			this.transformAlternatives(dyad, ignoredIndices);
		}
	}

	/**
	 * Transforms only the instances of each dyad in a
	 * {@link DyadRankingDataset} according to the mean and standard
	 * deviation of the data the scaler has been fit to. The attributes with indices
	 * contained in ignoredIndices are not transformed. {
	 *
	 * @param dataset The dataset of which the alternatives are to be
	 *            standardized.
	 * @param ignoredIndices The {@link List} of indices that are been ignored by
	 *            the scaler.
	 */
	public void transformInstances(final IDyadRankingDataset dataset, final List<Integer> ignoredIndices) {
		for (IDyadRankingInstance instance : dataset) {
			if (instance instanceof SparseDyadRankingInstance) {
				SparseDyadRankingInstance drSparseInstance = (SparseDyadRankingInstance) instance;
				this.transformInstances(drSparseInstance, ignoredIndices);
			} else if (instance instanceof DenseDyadRankingInstance) {
				DenseDyadRankingInstance drDenseInstance = (DenseDyadRankingInstance) instance;
				this.transformInstances(drDenseInstance, ignoredIndices);
			} else {
				throw new IllegalArgumentException("The scalers only support SparseDyadRankingInstance and DyadRankingInstance!");
			}
		}
	}

	/**
	 * Transforms only the alternatives of each dyad in a
	 * {@link DyadRankingDataset} according to the mean and standard
	 * deviation of the data the scaler has been fit to. The attributes with indices
	 * contained in ignoredIndices are not transformed. {
	 *
	 * @param dataset The dataset of which the alternatives are to be
	 *            standardized.
	 * @param ignoredIndices The {@link List} of indices that are been ignored by
	 *            the scaler.
	 */
	public void transformAlternatives(final IDyadRankingDataset dataset, final List<Integer> ignoredIndices) {
		for (IDyadRankingInstance instance : dataset) {
			this.transformAlternatives(instance, ignoredIndices);
		}
	}

	/**
	 * Fits the standard scaler to the dataset and transforms the entire dataset
	 * according to the mean and standard deviation of the dataset.
	 *
	 * @param dataset The dataset to be standardized.
	 */

	public void fitTransform(final IDyadRankingDataset dataset) {
		this.fit(dataset);
		this.transform(dataset);
	}

	/**
	 * Prints the standard devations of all features this scaler has been fit to.
	 */
	public String getPrettySTDString() {
		if (this.statsX == null || this.statsY == null) {
			throw new IllegalStateException("The scaler must be fit before calling this method!");
		}

		StringBuilder builder = new StringBuilder();

		builder.append("Standard deviations for instances: ");
		for (SummaryStatistics stats : this.statsX) {
			builder.append(stats.getStandardDeviation());
			builder.append(", ");
		}
		builder.append(System.lineSeparator());

		builder.append("Standard deviations for alternatives: ");
		for (SummaryStatistics stats : this.statsY) {
			builder.append(stats.getStandardDeviation());
			builder.append(", ");
		}
		builder.append(System.lineSeparator());

		return builder.toString();
	}

	/**
	 * Returns a String for the means of all features this scaler has been fit to.
	 */
	public String getPrettyMeansString() {
		if (this.statsX == null || this.statsY == null) {
			throw new IllegalStateException("The scaler must be fit before calling this method!");
		}

		StringBuilder builder = new StringBuilder();

		builder.append("Means for instances: ");
		for (SummaryStatistics stats : this.statsX) {
			builder.append(stats.getMean());
			builder.append(", ");
		}
		builder.append(System.lineSeparator());

		builder.append("Means for alternatives: ");
		for (SummaryStatistics stats : this.statsY) {
			builder.append(stats.getMean());
			builder.append(", ");
		}
		builder.append(System.lineSeparator());

		return builder.toString();
	}
}
