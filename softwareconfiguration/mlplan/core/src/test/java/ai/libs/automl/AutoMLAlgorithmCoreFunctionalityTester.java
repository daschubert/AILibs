package ai.libs.automl;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.learner.ISupervisedLearner;
import org.api4.java.algorithm.IAlgorithm;
import org.junit.runners.Parameterized.Parameters;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.jaicore.basic.algorithm.AlgorithmCreationException;
import ai.libs.jaicore.basic.algorithm.GeneralAlgorithmTester;
import ai.libs.jaicore.ml.experiments.OpenMLProblemSet;

public abstract class AutoMLAlgorithmCoreFunctionalityTester extends GeneralAlgorithmTester {

	private static final Logger logger = LoggerFactory.getLogger(AutoMLAlgorithmCoreFunctionalityTester.class);

	// creates the test data
	@Parameters(name = "{0}")
	public static Collection<Object[]> data() throws IOException, Exception {
		List<Object> problemSets = new ArrayList<>();
		problemSets.add(new OpenMLProblemSet(3)); // kr-vs-kp
		//		problemSets.add(new OpenMLProblemSet(1150)); // AP_Breast_Lung
		//		problemSets.add(new OpenMLProblemSet(1156)); // AP_Omentum_Ovary
		//				problemSets.add(new OpenMLProblemSet(1152)); // AP_Prostate_Ovary
		//				problemSets.add(new OpenMLProblemSet(1240)); // AirlinesCodrnaAdult
		//				problemSets.add(new OpenMLProblemSet(1457)); // amazon
		//				problemSets.add(new OpenMLProblemSet(149)); // CovPokElec
		//				problemSets.add(new OpenMLProblemSet(41103)); // cifar-10
		//				problemSets.add(new OpenMLProblemSet(40668)); // connect-4
		Object[][] data = new Object[problemSets.size()][1];
		for (int i = 0; i < data.length; i++) {
			data[i][0] = problemSets.get(i);
		}
		return Arrays.asList(data);
	}

	@Override
	public IAlgorithm<?, ?> getAlgorithm(final Object problem) throws AlgorithmCreationException {
		try {
			ILabeledDataset<?> dataset = (ILabeledDataset<?>) problem;
			return this.getAutoMLAlgorithm(dataset);
		} catch (Exception e) {
			throw new AlgorithmCreationException(e);
		}
	}

	public abstract IAlgorithm<? extends ILabeledDataset<?>, ? extends ISupervisedLearner<?, ?>> getAutoMLAlgorithm(ILabeledDataset<?> data) throws AlgorithmCreationException, IOException;
}
