package ai.libs.automl.mlplan.test;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.Collection;
import java.util.stream.Collectors;

import org.junit.Test;

import ai.libs.jaicore.ml.weka.WekaUtil;
import ai.libs.mlplan.multiclass.wekamlplan.weka.WekaMLPlanWekaClassifier;

public class WekaMLPlanWekaClassifierCoverageTest {

	@Test
	public void testClassifierCoverage() throws IOException {
		WekaMLPlanWekaClassifier mlplan = new WekaMLPlanWekaClassifier();
		Collection<String> componentNames = mlplan.getComponents().stream().map(c -> c.getName()).collect(Collectors.toList());
		System.out.println("Components:");
		componentNames.forEach(n -> System.out.println("\t" + n));
		
		/* check that every standard classifier is in */
		for (String classifier : WekaUtil.getNativeMultiClassClassifiers()) {
			assertTrue("Classifier " + classifier + " not covered in component set.", componentNames.contains(classifier));
		}
		for (String classifier : WekaUtil.getBinaryClassifiers()) {
			assertTrue("Classifier " + classifier + " not covered in component set.", componentNames.contains(classifier));
		}
		for (String searcher : WekaUtil.getSearchers()) {
			assertTrue("Searcher " + searcher + " not covered in component set.", componentNames.contains(searcher));
		}
		for (String evaluators : WekaUtil.getFeatureEvaluators()) {
			assertTrue("Evaluator " + evaluators + " not covered in component set.", componentNames.contains(evaluators));
		}
	}

}
