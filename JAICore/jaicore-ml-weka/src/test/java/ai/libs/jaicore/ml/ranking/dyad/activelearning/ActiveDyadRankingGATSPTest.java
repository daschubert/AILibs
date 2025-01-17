package ai.libs.jaicore.ml.ranking.dyad.activelearning;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import ai.libs.jaicore.ml.ranking.dyad.DyadRankingLossUtil;
import ai.libs.jaicore.ml.ranking.dyad.dataset.DyadRankingDataset;
import ai.libs.jaicore.ml.ranking.dyad.learner.activelearning.ActiveDyadRanker;
import ai.libs.jaicore.ml.ranking.dyad.learner.activelearning.ConfidenceIntervalClusteringBasedActiveDyadRanker;
import ai.libs.jaicore.ml.ranking.dyad.learner.activelearning.DyadDatasetPoolProvider;
import ai.libs.jaicore.ml.ranking.dyad.learner.activelearning.PrototypicalPoolBasedActiveDyadRanker;
import ai.libs.jaicore.ml.ranking.dyad.learner.activelearning.RandomPoolBasedActiveDyadRanker;
import ai.libs.jaicore.ml.ranking.dyad.learner.activelearning.UCBPoolBasedActiveDyadRanker;
import ai.libs.jaicore.ml.ranking.dyad.learner.algorithm.IPLDyadRanker;
import ai.libs.jaicore.ml.ranking.dyad.learner.algorithm.PLNetDyadRanker;
import ai.libs.jaicore.ml.ranking.dyad.learner.util.DyadStandardScaler;
import ai.libs.jaicore.ml.ranking.loss.KendallsTauDyadRankingLoss;
import weka.clusterers.SimpleKMeans;

/**
 * This is a test based on Dirk Schäfers dyad ranking dataset based on
 * performance data of genetic algorithms on traveling salesman problem
 * instances https://github.com/disc5/ga-tsp-dataset which was used in [1] for
 * evaluation.
 *
 * [1] Schäfer, D., & Hüllermeier, E. (2015). Dyad Ranking using a Bilinear
 * {P}lackett-{L}uce Model. In Proceedings ECML/PKDD--2015, European Conference
 * on Machine Learning and Knowledge Discovery in Databases (pp. 227–242).
 * Porto, Portugal: Springer.
 *
 * @author Jonas Hanselle
 *
 */
@RunWith(Parameterized.class)
public class ActiveDyadRankingGATSPTest {

	private static final String GATSP_DATASET_FILE = "testrsc/ml/dyadranking/ga-tsp/GATSP-Data.txt";

	// N = number of training instances
	private static final int N = 120;
	// seed for shuffling the dataset

	PLNetDyadRanker ranker;
	DyadRankingDataset dataset;

	public ActiveDyadRankingGATSPTest(final PLNetDyadRanker ranker) {
		this.ranker = ranker;
	}

	@Before
	public void init() throws FileNotFoundException {
		// load dataset
		this.dataset = new DyadRankingDataset();
		this.dataset.deserialize(new FileInputStream(new File(GATSP_DATASET_FILE)));
	}

	@Test
	public void test() throws Exception {
		int seed = 0;

		Collections.shuffle(this.dataset, new Random(seed));

		// split data
		DyadRankingDataset trainData = new DyadRankingDataset(this.dataset.subList(0, N));
		DyadRankingDataset testData = new DyadRankingDataset(this.dataset.subList(N, this.dataset.size()));

		// standardize data
		DyadStandardScaler scaler = new DyadStandardScaler();
		scaler.fit(trainData);
		scaler.transformInstances(trainData);
		scaler.transformInstances(testData);

		DyadDatasetPoolProvider poolProvider = new DyadDatasetPoolProvider(trainData);
		poolProvider.setRemoveDyadsWhenQueried(false);

		SimpleKMeans clusterer = new SimpleKMeans();
		clusterer.setNumClusters(5);

		ConfidenceIntervalClusteringBasedActiveDyadRanker activeRanker = new ConfidenceIntervalClusteringBasedActiveDyadRanker(this.ranker, poolProvider, seed, 5, 5, clusterer);

		List<ActiveDyadRanker> activeRankers = new ArrayList<>();
		activeRankers.add(activeRanker);
		activeRankers.add(new UCBPoolBasedActiveDyadRanker(new PLNetDyadRanker(), new DyadDatasetPoolProvider(trainData), seed, 5, 5));
		activeRankers.add(new PrototypicalPoolBasedActiveDyadRanker(new PLNetDyadRanker(), new DyadDatasetPoolProvider(trainData), 5, 5, 0.0d, 5, seed));
		activeRankers.add(new RandomPoolBasedActiveDyadRanker(new PLNetDyadRanker(), new DyadDatasetPoolProvider(trainData), seed, 5));

		for (ActiveDyadRanker curActiveRanker : activeRankers) {
			// train the ranker
			for (int i = 0; i < 10; i++) {
				curActiveRanker.activelyTrain(1);
				double avgKendallTau = DyadRankingLossUtil.computeAverageLoss(new KendallsTauDyadRankingLoss(), testData, curActiveRanker.getRanker());
				double avgKendallTauIS = DyadRankingLossUtil.computeAverageLoss(new KendallsTauDyadRankingLoss(), new DyadRankingDataset(poolProvider.getQueriedRankings()), curActiveRanker.getRanker());
				System.out.println("Current Kendalls Tau: " + avgKendallTau);
				System.out.println("Current Kendalls Tau IS: " + avgKendallTauIS);
			}
			double avgKendallTau = DyadRankingLossUtil.computeAverageLoss(new KendallsTauDyadRankingLoss(), testData, curActiveRanker.getRanker());
			Assert.assertTrue(avgKendallTau > 0.0d);

			System.out.println("final results: ");
		}
	}

	@Parameters
	public static List<IPLDyadRanker> supplyDyadRankers() {
		PLNetDyadRanker plNetRanker = new PLNetDyadRanker();
		// Use a simple config such that the test finishes quickly
		return Arrays.asList(plNetRanker);
	}
}
