package jaicore.search.algorithms.standard.uncertainty.paretosearch;

import java.util.Comparator;

public class FirstInFirstOutComparator<T> implements Comparator<ParetoNode<T, Double>> {

    /**
     * Compares two Pareto nodes on time of insertion (n). FIFO behaviour.
     * @param first
     * @param second
     * @return negative iff first.n < second.n, 0 iff fist.n == second.n, positive iff first.n > second.n
     */
    public int compare(ParetoNode<T, Double> first, ParetoNode<T, Double> second) {
        return first.n - second.n;
    }

}
