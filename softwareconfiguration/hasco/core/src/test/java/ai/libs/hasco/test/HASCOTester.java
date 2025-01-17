package ai.libs.hasco.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeoutException;

import org.api4.java.algorithm.events.IAlgorithmEvent;
import org.api4.java.algorithm.exceptions.AlgorithmException;
import org.api4.java.algorithm.exceptions.AlgorithmExecutionCanceledException;
import org.api4.java.algorithm.exceptions.AlgorithmTimeoutedException;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.eventbus.Subscribe;

import ai.libs.hasco.core.HASCO;
import ai.libs.hasco.core.events.HASCOSolutionEvent;
import ai.libs.jaicore.basic.algorithm.AlgorithmTestProblemSetCreationException;
import ai.libs.jaicore.basic.sets.Pair;
import ai.libs.jaicore.components.model.ComponentInstance;
import ai.libs.jaicore.components.model.RefinementConfiguredSoftwareConfigurationProblem;
import ai.libs.jaicore.components.serialization.CompositionSerializer;
import ai.libs.jaicore.search.probleminputs.GraphSearchWithPathEvaluationsInput;
import ai.libs.jaicore.search.util.CycleDetectedResult;
import ai.libs.jaicore.search.util.DeadEndDetectedResult;
import ai.libs.jaicore.search.util.GraphSanityChecker;
import ai.libs.jaicore.search.util.SanityCheckResult;

public abstract class HASCOTester<S extends GraphSearchWithPathEvaluationsInput<N, A, Double>, N, A> extends SoftwareConfigurationAlgorithmTester {

	private Logger logger = LoggerFactory.getLogger(HASCOTester.class);

	@Override
	public abstract HASCO<N, A, Double> getAlgorithmForSoftwareConfigurationProblem(RefinementConfiguredSoftwareConfigurationProblem<Double> problem);

	@Override
	public SoftwareConfigurationProblemSet getProblemSet() {
		return (SoftwareConfigurationProblemSet) super.getProblemSet();
	}

	private HASCO<N, A, Double> getHASCOForSimpleProblem() throws AlgorithmTestProblemSetCreationException {
		HASCO<N, A, Double> hasco = this.getAlgorithmForSoftwareConfigurationProblem(this.getProblemSet().getSimpleProblemInputForGeneralTestPurposes());
		hasco.setLoggerName(TESTEDALGORITHM_LOGGERNAME);
		return hasco;
	}

	private HASCO<N, A, Double> getHASCOForDifficultProblem() throws AlgorithmTestProblemSetCreationException {
		HASCO<N, A, Double> hasco = this.getAlgorithmForSoftwareConfigurationProblem(this.getProblemSet().getDifficultProblemInputForGeneralTestPurposes());
		hasco.setLoggerName(TESTEDALGORITHM_LOGGERNAME);
		return hasco;
	}

	private HASCO<N, A, Double> getHASCOForProblemWithDependencies() throws AlgorithmTestProblemSetCreationException {
		HASCO<N, A, Double> hasco = this.getAlgorithmForSoftwareConfigurationProblem(this.getProblemSet().getDependencyProblemInput());
		hasco.setLoggerName(TESTEDALGORITHM_LOGGERNAME);
		return hasco;
	}

	private Collection<Pair<HASCO<N, A, Double>, Integer>> getAllHASCOObjectsWithExpectedNumberOfSolutionsForTheKnownProblems() throws AlgorithmTestProblemSetCreationException {
		Collection<Pair<HASCO<N, A, Double>, Integer>> hascoObjects = new ArrayList<>();
		hascoObjects.add(new Pair<>(this.getHASCOForSimpleProblem(), 6));
		hascoObjects.add(new Pair<>(this.getHASCOForDifficultProblem(), -1));
		hascoObjects.add(new Pair<>(this.getHASCOForProblemWithDependencies(), 17));
		return hascoObjects;
	}

	@Test
	public void sanityCheckOfSearchGraph() throws InterruptedException, AlgorithmExecutionCanceledException, AlgorithmTimeoutedException, AlgorithmException, AlgorithmTestProblemSetCreationException {
		for (Pair<HASCO<N, A, Double>, Integer> pairOfHASCOAndNumOfSolutions : this.getAllHASCOObjectsWithExpectedNumberOfSolutionsForTheKnownProblems()) {
			HASCO<N, A, Double> hasco = pairOfHASCOAndNumOfSolutions.getX();

			/* check on dead end */
			GraphSanityChecker<N, A> deadEndDetector = new GraphSanityChecker<>(hasco.getSearch().getInput(), 2000);
			deadEndDetector.setLoggerName("testedalgorithm");
			deadEndDetector.call();
			SanityCheckResult sanity = deadEndDetector.getSanityCheck();
			assertTrue("HASCO graph has a dead end: " + sanity, !(sanity instanceof DeadEndDetectedResult));
			assertTrue("HASCO graph has a cycle: " + sanity, !(sanity instanceof CycleDetectedResult));
		}
	}

	@Test
	public void testThatAnEventForEachPossibleSolutionIsEmittedInSimpleCall() throws InterruptedException, AlgorithmExecutionCanceledException, TimeoutException, AlgorithmException, AlgorithmTestProblemSetCreationException {
		for (Pair<HASCO<N, A, Double>, Integer> pairOfHASCOAndExpectedNumberOfSolutions : this.getAllHASCOObjectsWithExpectedNumberOfSolutionsForTheKnownProblems()) {
			HASCO<N, A, Double> hasco = pairOfHASCOAndExpectedNumberOfSolutions.getX();
			this.checkNumberOfSolutionOnHASCO(hasco, pairOfHASCOAndExpectedNumberOfSolutions.getY());
		}
	}

	@Test
	public void testThatAnEventForEachPossibleSolutionIsEmittedInParallelizedCall() throws AlgorithmTestProblemSetCreationException, InterruptedException, AlgorithmExecutionCanceledException, TimeoutException, AlgorithmException {
		for (Pair<HASCO<N, A, Double>, Integer> pairOfHASCOAndExpectedNumberOfSolutions : this.getAllHASCOObjectsWithExpectedNumberOfSolutionsForTheKnownProblems()) {
			HASCO<N, A, Double> hasco = pairOfHASCOAndExpectedNumberOfSolutions.getX();
			hasco.setNumCPUs(Runtime.getRuntime().availableProcessors());
			this.checkNumberOfSolutionOnHASCO(hasco, pairOfHASCOAndExpectedNumberOfSolutions.getY());
		}
	}

	private void checkNumberOfSolutionOnHASCO(final HASCO<N, A, Double> hasco, final int numberOfExpectedSolutions) throws InterruptedException, AlgorithmExecutionCanceledException, TimeoutException, AlgorithmException {
		if (numberOfExpectedSolutions < 0) {
			return;
		}
		List<ComponentInstance> solutions = new ArrayList<>();
		hasco.registerListener(new Object() {

			@Subscribe
			public void registerSolution(final HASCOSolutionEvent<Double> e) {
				solutions.add(e.getSolutionCandidate().getComponentInstance());
				HASCOTester.this.logger.info("Found solution {}", CompositionSerializer.serializeComponentInstance(e.getSolutionCandidate().getComponentInstance()));
			}
		});
		hasco.call();
		Set<Object> uniqueSolutions = new HashSet<>(solutions);
		assertEquals("Only found " + uniqueSolutions.size() + "/" + numberOfExpectedSolutions + " solutions", numberOfExpectedSolutions, uniqueSolutions.size());
		assertEquals("All " + numberOfExpectedSolutions + " solutions were found, but " + solutions.size() + " solutions were returned in total, i.e. there are solutions returned twice", numberOfExpectedSolutions, solutions.size());
	}

	@SuppressWarnings("unchecked")
	@Test
	public void testThatIteratorReturnsEachPossibleSolution() throws AlgorithmTestProblemSetCreationException {
		for (Pair<HASCO<N, A, Double>, Integer> pairOfHASCOAndExpectedNumberOfSolutions : this.getAllHASCOObjectsWithExpectedNumberOfSolutionsForTheKnownProblems()) {
			HASCO<N, A, Double> hasco = pairOfHASCOAndExpectedNumberOfSolutions.getX();
			int numberOfExpectedSolutions = pairOfHASCOAndExpectedNumberOfSolutions.getY();
			this.logger.info("Starting HASCO on problem {} with {} solutions.", hasco, numberOfExpectedSolutions);
			if (numberOfExpectedSolutions < 0) {
				continue;
			}
			List<ComponentInstance> solutions = new ArrayList<>();
			for (IAlgorithmEvent e : hasco) {
				if (e instanceof HASCOSolutionEvent) {
					solutions.add(((HASCOSolutionEvent<Double>) e).getSolutionCandidate().getComponentInstance());
					this.logger.info("Found solution {}", CompositionSerializer.serializeComponentInstance(((HASCOSolutionEvent<Double>) e).getSolutionCandidate().getComponentInstance()));
				}
			}
			this.logger.info("Finished HASCO, now evaluating numbers of found solutions.");
			Set<Object> uniqueSolutions = new HashSet<>(solutions);
			assertEquals("Only found " + uniqueSolutions.size() + "/" + numberOfExpectedSolutions + " solutions", numberOfExpectedSolutions, uniqueSolutions.size());
			assertEquals("All " + numberOfExpectedSolutions + " solutions were found, but " + solutions.size() + " solutions were returned in total, i.e. there are solutions returned twice", numberOfExpectedSolutions, solutions.size());
		}

	}
}
