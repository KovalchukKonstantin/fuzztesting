import { useState, useEffect } from 'react';
import {
    LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
    ComposedChart, Area
} from 'recharts';

interface Metric {
    iteration: number;
    num_principles: number;
    new_principles_count: number;
    merged_principles_count: number;
    avg_score: number;
    score_variance: number;
    score_alignment: number | null;
}

const RubricMetricsView = () => {
    const [metrics, setMetrics] = useState<Metric[]>([]);
    const [loading, setLoading] = useState(true);

    const fetchMetrics = async () => {
        try {
            const res = await fetch('http://localhost:8000/api/rubrics/metrics');
            const data = await res.json();
            setMetrics(data);
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchMetrics();
    }, []);

    if (loading) return <div className="p-4 text-gray-400">Loading metrics...</div>;
    if (metrics.length === 0) return <div className="p-4 text-gray-400">No metrics data available yet. Run an iteration!</div>;

    return (
        <div className="h-full flex flex-col p-6 space-y-6 overflow-y-auto bg-gray-900 text-gray-200">
            <div className="flex justify-between items-center">
                <h2 className="text-xl font-bold text-white">Rubric Effectiveness Metrics</h2>
                <button
                    onClick={fetchMetrics}
                    className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm transition-colors"
                >
                    Refresh
                </button>
            </div>

            {/* Chart 1: Effectiveness (Alignment & Variance) */}
            <div className="bg-gray-800 p-4 rounded-lg shadow-lg border border-gray-700">
                <h3 className="text-lg font-semibold mb-4 text-secondary-400">Effectiveness: Alignment & Variance</h3>
                <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={metrics}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                            <XAxis dataKey="iteration" stroke="#9CA3AF" label={{ value: 'Iteration', position: 'insideBottom', offset: -5 }} />
                            <YAxis yAxisId="left" stroke="#10B981" label={{ value: 'Alignment', angle: -90, position: 'insideLeft' }} />
                            <YAxis yAxisId="right" orientation="right" stroke="#F59E0B" label={{ value: 'Variance', angle: 90, position: 'insideRight' }} />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#1F2937', borderColor: '#374151', color: '#F3F4F6' }}
                                itemStyle={{ color: '#F3F4F6' }}
                            />
                            <Legend />
                            <Line yAxisId="left" type="monotone" dataKey="score_alignment" stroke="#10B981" name="Alignment (Rel - Irrel)" strokeWidth={2} dot={{ r: 4 }} />
                            <Line yAxisId="right" type="monotone" dataKey="score_variance" stroke="#F59E0B" name="Score Variance" strokeWidth={2} dot={{ r: 4 }} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
                <p className="text-xs text-gray-500 mt-2">
                    <b>Alignment:</b> Higher is better (rubric distinguishes good/bad). <b>Variance:</b> Higher is better (rubric is opinionated).
                </p>
            </div>

            {/* Chart 2: Stability (Size, New, Merged) */}
            <div className="bg-gray-800 p-4 rounded-lg shadow-lg border border-gray-700">
                <h3 className="text-lg font-semibold mb-4 text-blue-400">Rubric Stability & Size</h3>
                <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={metrics}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                            <XAxis dataKey="iteration" stroke="#9CA3AF" label={{ value: 'Iteration', position: 'insideBottom', offset: -5 }} />
                            <YAxis yAxisId="left" stroke="#ffffff" label={{ value: 'Changes', angle: -90, position: 'insideLeft' }} />
                            <YAxis yAxisId="right" orientation="right" stroke="#60A5FA" label={{ value: 'Total Principles', angle: 90, position: 'insideRight' }} />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#1F2937', borderColor: '#374151', color: '#F3F4F6' }}
                                itemStyle={{ color: '#F3F4F6' }}
                            />
                            <Legend />

                            {/* Stacked Bars for Changes */}
                            <Bar yAxisId="left" dataKey="new_principles_count" stackId="a" fill="#10B981" name="New Principles" barSize={20} />
                            <Bar yAxisId="left" dataKey="merged_principles_count" stackId="a" fill="#F59E0B" name="Merged Principles" barSize={20} />

                            {/* Line for Total Size */}
                            <Line yAxisId="right" type="monotone" dataKey="num_principles" stroke="#60A5FA" name="Total Principles" strokeWidth={3} dot={{ r: 4 }} />
                        </ComposedChart>
                    </ResponsiveContainer>
                </div>
                <p className="text-xs text-gray-500 mt-2">
                    Bars show activity per iteration. Line shows total rubric size growth. Zero "New Principles" indicates convergence.
                </p>
            </div>
        </div>
    );
};

export default RubricMetricsView;
