package ai.libs.jaicore.components.optimizingfactory;

import java.util.HashMap;
import java.util.Map;

import org.api4.java.algorithm.Timeout;
import org.api4.java.algorithm.events.IAlgorithmEvent;
import org.api4.java.algorithm.exceptions.AlgorithmException;
import org.api4.java.algorithm.exceptions.AlgorithmExecutionCanceledException;
import org.api4.java.algorithm.exceptions.AlgorithmTimeoutedException;
import org.api4.java.common.control.ILoggingCustomizable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.eventbus.Subscribe;

import ai.libs.jaicore.basic.algorithm.AAlgorithm;
import ai.libs.jaicore.basic.algorithm.AlgorithmFinishedEvent;
import ai.libs.jaicore.basic.algorithm.AlgorithmInitializedEvent;
import ai.libs.jaicore.components.exceptions.ComponentInstantiationFailedException;
import ai.libs.jaicore.components.model.ComponentInstance;
import ai.libs.jaicore.components.model.EvaluatedSoftwareConfigurationSolution;
import ai.libs.jaicore.components.model.SoftwareConfigurationProblem;
import ai.libs.jaicore.logging.ToJSONStringUtil;

public class OptimizingFactory<P extends SoftwareConfigurationProblem<V>, T, C extends EvaluatedSoftwareConfigurationSolution<V>, V extends Comparable<V>> extends AAlgorithm<OptimizingFactoryProblem<P, T, V>, T> {

	/* logging */
	private Logger localLogger = LoggerFactory.getLogger(OptimizingFactory.class);
	private String loggerName;

	private final SoftwareConfigurationAlgorithmFactory<P, C, V, ?> factoryForOptimizationAlgorithm;
	private T constructedObject;
	private V performanceOfObject;
	private ComponentInstance componentInstanceOfObject;
	private final SoftwareConfigurationAlgorithm<P, C, V> optimizer;

	public OptimizingFactory(final OptimizingFactoryProblem<P, T, V> problem, final SoftwareConfigurationAlgorithmFactory<P, C, V, ?> factoryForOptimizationAlgorithm) {
		super(problem);
		this.factoryForOptimizationAlgorithm = factoryForOptimizationAlgorithm;
		this.optimizer = this.factoryForOptimizationAlgorithm.getAlgorithm(this.getInput().getConfigurationProblem());
		this.optimizer.registerListener(new Object() {
			@Subscribe
			public void receiveAlgorithmEvent(final IAlgorithmEvent event) {
				if (!(event instanceof AlgorithmInitializedEvent || event instanceof AlgorithmFinishedEvent)) {
					OptimizingFactory.this.post(event);
				}
			}
		});
	}

	@Override
	public IAlgorithmEvent nextWithException() throws AlgorithmException, InterruptedException, AlgorithmExecutionCanceledException, AlgorithmTimeoutedException {
		switch (this.getState()) {
		case CREATED:

			/* initialize optimizer */
			IAlgorithmEvent initEvent = this.optimizer.next();
			assert initEvent instanceof AlgorithmInitializedEvent : "The first event emitted by the optimizer has not been its AlgorithmInitializationEvent";
			return this.activate();

		case ACTIVE:
			C solutionModel = this.optimizer.call();
			try {
				this.constructedObject = this.getInput().getBaseFactory().getComponentInstantiation(solutionModel.getComponentInstance());
				this.performanceOfObject = solutionModel.getScore();
				this.componentInstanceOfObject = solutionModel.getComponentInstance();
				return this.terminate();
			} catch (ComponentInstantiationFailedException e) {
				throw new AlgorithmException("Could not conduct next step in OptimizingFactory due to an exception in the component instantiation.", e);
			}
		default:
			throw new IllegalStateException("Cannot do anything in state " + this.getState());
		}
	}

	@Override
	public T call() throws AlgorithmException, InterruptedException, AlgorithmExecutionCanceledException, AlgorithmTimeoutedException {
		while (this.hasNext()) {
			this.nextWithException();
		}
		return this.constructedObject;
	}

	/**
	 * @return the optimizer that is used for building the object
	 */
	public SoftwareConfigurationAlgorithm<P, C, V> getOptimizer() {
		return this.optimizer;
	}

	public AlgorithmInitializedEvent init() {
		IAlgorithmEvent e = null;
		while (this.hasNext()) {
			e = this.next();
			if (e instanceof AlgorithmInitializedEvent) {
				return (AlgorithmInitializedEvent) e;
			}
		}
		throw new IllegalStateException("Could not complete initialization");
	}

	public V getPerformanceOfObject() {
		return this.performanceOfObject;
	}

	public ComponentInstance getComponentInstanceOfObject() {
		return this.componentInstanceOfObject;
	}

	@Override
	public String getLoggerName() {
		return this.loggerName;
	}

	@Override
	public void setLoggerName(final String name) {
		this.localLogger.info("Switching logger from {} to {}", this.localLogger.getName(), name);
		this.loggerName = name;
		this.localLogger = LoggerFactory.getLogger(name);
		this.localLogger.info("Activated logger {} with name {}", name, this.localLogger.getName());
		if (this.optimizer instanceof ILoggingCustomizable) {
			this.optimizer.setLoggerName(name + ".optimizer");
		}
		super.setLoggerName(this.loggerName + "._algorithm");
	}

	@Override
	public String toString() {
		Map<String, Object> fields = new HashMap<>();
		fields.put("factoryForOptimizationAlgorithm", this.factoryForOptimizationAlgorithm);
		fields.put("constructedObject", this.constructedObject);
		fields.put("performanceOfObject", this.performanceOfObject);
		fields.put("optimizer", this.optimizer);
		return ToJSONStringUtil.toJSONString(fields);
	}

	@Override
	public void cancel() {
		this.localLogger.info("Received cancel. First canceling the optimizer {}, then my own routine!", this.optimizer.getId());
		this.optimizer.cancel();
		this.localLogger.debug("Now canceling the OptimizingFactory itself.");
		super.cancel();
		assert this.isCanceled() : "Cancel-flag must be true at end of cancel routine!";
	}

	@Override
	public void setTimeout(final Timeout to) {
		super.setTimeout(to);
		this.localLogger.info("Forwarding timeout {} to optimizer.", to);
		this.optimizer.setTimeout(to);
	}
}
