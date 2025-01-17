package ai.libs.hasco.gui.statsplugin;

import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import ai.libs.jaicore.basic.sets.Pair;
import ai.libs.jaicore.graphvisualizer.events.gui.Histogram;
import ai.libs.jaicore.graphvisualizer.plugin.ASimpleMVCPluginView;
import ai.libs.jaicore.graphvisualizer.plugin.solutionperformanceplotter.ScoredSolutionCandidateInfo;
import javafx.application.Platform;
import javafx.scene.control.TreeView;
import javafx.scene.layout.VBox;

/**
 *
 * @author fmohr
 *
 */
public class HASCOModelStatisticsPluginView extends ASimpleMVCPluginView<HASCOModelStatisticsPluginModel, HASCOModelStatisticsPluginController, VBox> {

	private final HASCOModelStatisticsComponentSelector rootNode; // the root of the TreeView shown at the top
	private final Histogram histogram; // the histogram shown on the bottom

	public HASCOModelStatisticsPluginView(final HASCOModelStatisticsPluginModel model) {
		this(model, 100);
	}

	public HASCOModelStatisticsPluginView(final HASCOModelStatisticsPluginModel model, final int n) {
		super(model, new VBox());
		this.rootNode = new HASCOModelStatisticsComponentSelector(this, model);
		TreeView<HASCOModelStatisticsComponentSelector> treeView = new TreeView<>();
		treeView.setCellFactory(HASCOModelStatisticsComponentCell::new);
		treeView.setRoot(this.rootNode);
		this.getNode().getChildren().add(treeView);
		this.histogram = new Histogram(n);
		this.histogram.setTitle("Performances observed on the filtered solutions");
		this.getNode().getChildren().add(this.histogram);
	}

	@Override
	public void update() {
		this.rootNode.update();
		this.updateHistogram();
	}

	/**
	 * Updates the histogram at the bottom. This is called in both the update method of the general view as well as in the change listener of the combo boxes.
	 */
	public void updateHistogram() {
		Collection<List<Pair<String, String>>> activeFilters = this.rootNode.getAllSelectionsOnPathToAnyLeaf();
		List<ScoredSolutionCandidateInfo> activeSolutions = this.getModel().getAllSeenSolutionCandidateFoundInfosUnordered().stream()
				.filter(i -> this.getModel().deserializeComponentInstance(i.getSolutionCandidateRepresentation()).matchesPathRestrictions(activeFilters)).collect(Collectors.toList());
		DescriptiveStatistics stats = new DescriptiveStatistics();
		activeSolutions.forEach(s -> stats.addValue(this.getModel().parseScoreToDouble(s.getScore())));
		Platform.runLater(() -> this.histogram.update(stats));
	}

	@Override
	public void clear() {
		this.rootNode.clear();
		this.histogram.clear();
	}
}
