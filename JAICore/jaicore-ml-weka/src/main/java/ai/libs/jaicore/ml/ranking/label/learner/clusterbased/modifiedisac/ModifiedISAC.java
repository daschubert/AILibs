package ai.libs.jaicore.ml.ranking.label.learner.clusterbased.modifiedisac;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.api4.java.ai.ml.core.exception.PredictionException;
import org.api4.java.ai.ml.core.exception.TrainingException;
import org.api4.java.ai.ml.ranking.IRanking;
import org.api4.java.ai.ml.ranking.IRankingPredictionBatch;
import org.api4.java.ai.ml.ranking.label.dataset.ILabelRankingDataset;
import org.api4.java.ai.ml.ranking.label.dataset.ILabelRankingInstance;

import ai.libs.jaicore.basic.sets.Pair;
import ai.libs.jaicore.ml.core.learner.ASupervisedLearner;
import ai.libs.jaicore.ml.ranking.RankingPredictionBatch;
import ai.libs.jaicore.ml.ranking.label.learner.clusterbased.IGroupBasedRanker;
import ai.libs.jaicore.ml.ranking.label.learner.clusterbased.customdatatypes.Group;
import ai.libs.jaicore.ml.ranking.label.learner.clusterbased.customdatatypes.ProblemInstance;
import ai.libs.jaicore.ml.ranking.label.learner.clusterbased.customdatatypes.RankingForGroup;
import weka.core.Instance;

/**
 * @author Helen
 *         ModifiedISAC handles the preparation of the data and the clustering of it as well as the
 *         the search for a cluster for a new instance.
 */
public class ModifiedISAC extends ASupervisedLearner<ILabelRankingInstance, ILabelRankingDataset, IRanking<String>, IRankingPredictionBatch> implements IGroupBasedRanker<String, ILabelRankingInstance, ILabelRankingDataset, double[]> {
	// Saves the position of the points in the original list to save their relation to the corresponding
	// instance.
	private Map<double[], Integer> positionOfInstance = new HashMap<>();
	// Saves the rankings for the found cluster in form of the cluster center and the ranking of Classifier by their name.
	private ArrayList<ClassifierRankingForGroup> rankings = new ArrayList<>();
	// Saves the found cluster
	private List<Group<double[], Instance>> foundCluster;
	// Saves the used normalizer
	private Normalizer norm;

	/* (non-Javadoc)
	 * @see jaicore.Ranker.Ranker#bulidRanker()
	 */
	@Override
	public void fit(final ILabelRankingDataset dTrain) throws TrainingException {
		ModifiedISACInstanceCollector collector;
		try {
			collector = new ModifiedISACInstanceCollector();
		} catch (Exception e) {
			throw new TrainingException("Could not build the ranker.", e);
		}
		List<ProblemInstance<Instance>> collectedInstances = collector.getProblemInstances();
		List<double[]> toClusterpoints = new ArrayList<>();

		this.norm = new Normalizer(collectedInstances);
		this.norm.setupnormalize();

		for (ProblemInstance<Instance> tmp : collectedInstances) {
			toClusterpoints.add(this.norm.normalize(tmp.getInstance().toDoubleArray()));
		}

		ModifiedISACGroupBuilder builder = new ModifiedISACGroupBuilder();
		builder.setPoints(toClusterpoints);

		int tmp = 0;
		for (ProblemInstance<Instance> i : collectedInstances) {
			this.positionOfInstance.put(i.getInstance().toDoubleArray(), tmp);
			tmp++;
		}

		this.foundCluster = builder.buildGroup(collectedInstances);
		this.constructRanking(collector);
	}

	/**
	 * given the collector and the used Classifier it construct a ranking for the found classifer
	 *
	 * @param collector
	 */
	private void constructRanking(final ModifiedISACInstanceCollector collector) {
		for (Group<double[], Instance> c : this.foundCluster) {
			ArrayList<String> ranking = new ArrayList<>();
			int[] tmp = new int[collector.getNumberOfClassifier()];
			double[] clusterMean = new double[collector.getNumberOfClassifier()];
			for (ProblemInstance<Instance> prob : c.getInstances()) {
				int myIndex = 0;
				for (Entry<double[], Integer> instancePositionWithNumber : this.positionOfInstance.entrySet()) {
					if (Arrays.equals(instancePositionWithNumber.getKey(), prob.getInstance().toDoubleArray())) {
						myIndex = instancePositionWithNumber.getValue();
						break;
					}
				}
				ArrayList<Pair<String, Double>> solutionsOfPoint = collector.getCollectedClassifierandPerformance().get(myIndex);
				for (int i = 0; i < solutionsOfPoint.size(); i++) {

					double perfo = solutionsOfPoint.get(i).getY();
					if (!Double.isNaN(perfo)) {

						clusterMean[i] += perfo;
						tmp[i]++;
					}
				}
			}
			for (int i = 0; i < clusterMean.length; i++) {
				clusterMean[i] = clusterMean[i] / tmp[i];
			}

			List<String> allClassifier = collector.getAllClassifier();
			Map<String, Double> remainingCandidiates = new HashMap<>();
			for (int i = 0; i < clusterMean.length; i++) {
				remainingCandidiates.put(allClassifier.get(i), clusterMean[i]);
			}

			while (!remainingCandidiates.isEmpty()) {
				double min = Double.MIN_VALUE;
				String classi = null;
				for (Entry<String, Double> nameWithCandidate : remainingCandidiates.entrySet()) {
					double candidate = nameWithCandidate.getValue();
					if (candidate > min) {
						classi = nameWithCandidate.getKey();
						min = candidate;
					}
				}
				if (classi == null) {
					for (String str : remainingCandidiates.keySet()) {
						ranking.add(str);
					}
					remainingCandidiates.clear();
				} else {
					ranking.add(classi);
					remainingCandidiates.remove(classi);
				}

			}

			this.rankings.add(new ClassifierRankingForGroup(c.getId(), ranking));
		}
	}

	/* (non-Javadoc)
	 * @see jaicore.GroupBasedRanker.GroupBasedRanker#getRanking(java.lang.Object)
	 */
	@Override
	public RankingForGroup<double[], String> getRanking(final ILabelRankingInstance prob) {
		RankingForGroup<double[], String> myRanking = null;
		double[] point = this.norm.normalize(prob.getPoint());
		L1DistanceMetric dist = new L1DistanceMetric();
		double minDist = Double.MAX_VALUE;
		for (RankingForGroup<double[], String> rank : this.rankings) {
			double computedDist = dist.computeDistance(rank.getIdentifierForGroup().getIdentifier(), point);

			if (computedDist <= minDist) {
				myRanking = rank;
				minDist = computedDist;
			}
		}
		return myRanking;
	}

	/**
	 * @return
	 */
	public List<ClassifierRankingForGroup> getRankings() {
		return this.rankings;
	}

	@Override
	public IRanking<String> predict(final ILabelRankingInstance xTest) throws PredictionException, InterruptedException {
		return this.getRanking(xTest);
	}

	@Override
	public IRankingPredictionBatch predict(ILabelRankingInstance[] dTest) throws PredictionException, InterruptedException {
		List<IRanking<?>> rankingPredictions = new ArrayList<>();
		for (ILabelRankingInstance instance : dTest) {
			rankingPredictions.add(predict(instance));
		}
		return new RankingPredictionBatch(rankingPredictions);
	}
}
