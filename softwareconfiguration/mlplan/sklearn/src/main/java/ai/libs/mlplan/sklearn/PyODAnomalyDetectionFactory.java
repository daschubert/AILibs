package ai.libs.mlplan.sklearn;

import ai.libs.jaicore.ml.core.EScikitLearnProblemType;

public class PyODAnomalyDetectionFactory extends ATwoStepPipelineScikitLearnFactory {

    public PyODAnomalyDetectionFactory() {
		super(EScikitLearnProblemType.ANOMALY_DETECTION, "anomaly_detector");
	}
}
