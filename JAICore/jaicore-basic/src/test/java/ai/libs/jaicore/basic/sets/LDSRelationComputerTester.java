package ai.libs.jaicore.basic.sets;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

import org.api4.java.algorithm.IAlgorithm;
import org.api4.java.algorithm.exceptions.AlgorithmExecutionCanceledException;
import org.api4.java.algorithm.exceptions.AlgorithmTimeoutedException;
import org.junit.Test;

import ai.libs.jaicore.basic.algorithm.AlgorithmTestProblemSetCreationException;
import ai.libs.jaicore.basic.sets.algorithms.RelationComputerTester;

public class LDSRelationComputerTester extends RelationComputerTester {

	@Test
	public void testOutputSizeForCartesianProducts() throws AlgorithmTimeoutedException, InterruptedException, AlgorithmExecutionCanceledException, AlgorithmTestProblemSetCreationException {
		RelationComputationProblem<Object> problem = this.getCartesianProductProblem();
		int expected = problem.getSets().get(0).size() * problem.getSets().get(1).size() * problem.getSets().get(2).size();
		this.testRelation(problem, expected);
	}

	@Test
	public void testOutputSizeForNonEmptyRelation() throws AlgorithmTimeoutedException, InterruptedException, AlgorithmExecutionCanceledException, AlgorithmTestProblemSetCreationException {
		RelationComputationProblem<Object> problem = this.getProblemSet().getSimpleProblemInputForGeneralTestPurposes();
		LDSRelationComputer<Object> ldsComputer = new LDSRelationComputer<>(this.getCartesianProductProblem());
		ldsComputer.setLoggerName(TESTEDALGORITHM_LOGGERNAME);
		List<List<Object>> cartesianProduct = ldsComputer.call();
		List<List<?>> groundTruth = cartesianProduct.stream().filter(problem.getPrefixFilter()).collect(Collectors.toList());
		this.testRelation(problem, groundTruth.size());
	}

	@Test
	public void testOutputSizeForEmptyRelation() throws AlgorithmTimeoutedException, InterruptedException, AlgorithmExecutionCanceledException, AlgorithmTestProblemSetCreationException {
		this.testRelation(this.getInfeasibleRelationProblem(), 0);
	}

	@Test
	public void testOutputSizeForPrunedRelation() throws AlgorithmTimeoutedException, InterruptedException, AlgorithmExecutionCanceledException, AlgorithmTestProblemSetCreationException {
		this.testRelation(this.getInfeasibleCompletelyPrunedRelationProblem(), 0);
	}

	private void testRelation(final RelationComputationProblem<Object> problem, final int expected) throws InterruptedException, AlgorithmExecutionCanceledException, AlgorithmTimeoutedException {
		LDSRelationComputer<Object> cpc = new LDSRelationComputer<>(problem);
		List<List<Object>> relation = cpc.call();
		assertEquals(expected, relation.size()); // the size of the output must be correct
		for (int i = 0; i < expected - 1; i++) {
			List<?> tuple1 = relation.get(i);
			assertEquals(problem.getSets().size(), tuple1.size());
			List<?> tuple2 = relation.get(i + 1);
			assertEquals("The tuple does not have the right length. Expected tuple length was " + problem.getSets().size() + ", but the given tuple has " + tuple2.size() + " entries", problem.getSets().size(), tuple2.size());
			int d1 = this.computeDefficiency(problem.getSets(), tuple1);
			int d2 = this.computeDefficiency(problem.getSets(), tuple2);
			assertTrue(d1 <= d2);
		}
	}

	private int computeDefficiency(final List<? extends Collection<?>> collections, final List<?> tuple) {
		int defficiency = 0;
		for (int i = 0; i < tuple.size(); i++) {
			List<?> ithSet = (List<?>) collections.get(i);
			defficiency += ithSet.indexOf(tuple.get(i));
		}
		return defficiency;
	}

	public RelationComputationProblem<Object> getCartesianProductProblem() throws AlgorithmTestProblemSetCreationException, InterruptedException {
		return new RelationComputationProblem<>(this.getProblemSet().getSimpleProblemInputForGeneralTestPurposes().getSets()); // remove the filter condition
	}

	public RelationComputationProblem<Object> getInfeasibleRelationProblem() throws AlgorithmTestProblemSetCreationException, InterruptedException {
		return new RelationComputationProblem<>(this.getProblemSet().getSimpleProblemInputForGeneralTestPurposes().getSets(), t -> t.size() < 3); // all full tuples are forbidden
	}

	public RelationComputationProblem<Object> getInfeasibleCompletelyPrunedRelationProblem() throws AlgorithmTestProblemSetCreationException, InterruptedException {
		return new RelationComputationProblem<>(this.getProblemSet().getSimpleProblemInputForGeneralTestPurposes().getSets(), t -> false); // all tuples are forbidden
	}

	@Override
	public IAlgorithm<?, ?> getAlgorithm(final Object problem) {
		return new LDSRelationComputer<>((RelationComputationProblem<?>) problem);
	}
}
