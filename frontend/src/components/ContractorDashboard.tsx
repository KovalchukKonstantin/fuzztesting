import { useState, useEffect } from 'react';
import { api, type ContractorTask } from '../api';
import { CheckCircle, XCircle, RefreshCw, Briefcase, ChevronDown, ChevronUp } from 'lucide-react';

export function ContractorDashboard() {
    const [tasks, setTasks] = useState<ContractorTask[]>([]);
    const [loading, setLoading] = useState(false);
    const [message, setMessage] = useState<string | null>(null);
    const [contractorId] = useState(() => {
        const stored = localStorage.getItem("contractor_id");
        if (stored) return stored;
        const newId = "anon_" + Math.floor(Math.random() * 10000);
        localStorage.setItem("contractor_id", newId);
        return newId;
    });

    // Store local decisions: taskId -> isRelevant
    const [pendingDecisions, setPendingDecisions] = useState<Record<string, boolean>>({});
    const [completedBatch, setCompletedBatch] = useState(false);
    const [isDescriptionExpanded, setIsDescriptionExpanded] = useState(false);

    // Metadata from the batch
    const [productDescription, setProductDescription] = useState<string | null>(null);

    useEffect(() => {
        fetchNextBatch();
    }, []);

    const fetchNextBatch = async () => {
        setLoading(true);
        setMessage(null);
        setTasks([]);
        setPendingDecisions({});
        setCompletedBatch(false);

        try {
            const data = await api.contractor.getNextBatch(contractorId);
            if (data.message && data.tasks.length === 0) {
                setMessage(data.message);
                setProductDescription(data.product_description);
            } else {
                setTasks(data.tasks);
                setProductDescription(data.product_description);
            }
        } catch (e) {
            console.error("Failed to fetch tasks", e);
            setMessage("Error fetching tasks");
        } finally {
            setLoading(false);
        }
    };

    const handleLocalDecision = (taskId: string, isRelevant: boolean) => {
        setPendingDecisions(prev => ({
            ...prev,
            [taskId]: isRelevant
        }));
    };

    const handleSubmitBatch = async () => {
        setLoading(true);
        try {
            const submissions = Object.entries(pendingDecisions).map(([taskId, isRelevant]) => ({
                task_id: taskId,
                is_relevant: isRelevant
            }));

            const res = await api.contractor.submitBatchGrades(submissions, contractorId);
            if (res.loop_triggered) {
                console.log("Feedback loop triggered");
            }

            setCompletedBatch(true);

            // Auto-fetch next batch after short delay
            setTimeout(() => {
                fetchNextBatch();
            }, 1000);

        } catch (e) {
            alert("Error submitting batch: " + e);
            setLoading(false);
        }
    };

    if (loading && tasks.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center p-12 text-gray-500">
                <RefreshCw className="animate-spin mb-4" size={32} />
                <p>Loading batch...</p>
            </div>
        );
    }

    const decisionsCount = Object.keys(pendingDecisions).length;
    const allDecided = tasks.length > 0 && decisionsCount === tasks.length;

    return (
        <div className="max-w-4xl mx-auto p-6 space-y-8">
            <header className="flex items-center justify-between border-b pb-4">
                <div className="flex items-center gap-2 text-indigo-700">
                    <Briefcase size={24} />
                    <h1 className="text-2xl font-bold">Contractor Dashboard</h1>
                </div>
                <div className="flex items-center gap-4">
                    <div className="text-xs text-gray-400">ID: {contractorId}</div>
                    {tasks.length > 0 && (
                        <div className="text-sm font-medium text-gray-600">
                            {decisionsCount} / {tasks.length} Graded
                        </div>
                    )}
                </div>
            </header>

            {tasks.length === 0 ? (
                <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8 text-center">
                    <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4 text-gray-400">
                        <Briefcase size={32} />
                    </div>
                    <h2 className="text-lg font-semibold text-gray-900 mb-2">No tasks available</h2>
                    <p className="text-gray-500 mb-6">{message || "The queue is currently empty. Please wait for more tasks to be assigned."}</p>

                    <button
                        onClick={fetchNextBatch}
                        disabled={loading}
                        className="px-6 py-2 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 transition-colors disabled:opacity-50"
                    >
                        {loading ? 'Checking...' : 'Check Again'}
                    </button>
                </div>
            ) : (
                <div className="space-y-6">
                    {/* Product Context Card */}
                    {productDescription && (
                        <div className="bg-blue-50 border border-blue-100 rounded-xl p-4 text-sm text-blue-900 sticky top-4 z-10 shadow-sm transition-all">
                            <div className="flex justify-between items-start gap-4">
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center justify-between mb-1">
                                        <h3 className="font-semibold text-blue-700">Product Context</h3>
                                        <button
                                            onClick={() => setIsDescriptionExpanded(!isDescriptionExpanded)}
                                            className="text-xs flex items-center gap-1 text-blue-600 hover:text-blue-800 font-medium"
                                        >
                                            {isDescriptionExpanded ? (
                                                <>Show Less <ChevronUp size={12} /></>
                                            ) : (
                                                <>Show More <ChevronDown size={12} /></>
                                            )}
                                        </button>
                                    </div>
                                    <p className={`leading-relaxed transition-all ${isDescriptionExpanded ? '' : 'line-clamp-3'}`}>
                                        {productDescription}
                                    </p>
                                </div>
                                {allDecided && !completedBatch && (
                                    <div className="flex-shrink-0">
                                        <button
                                            onClick={handleSubmitBatch}
                                            disabled={loading}
                                            className="px-6 py-2 bg-indigo-600 text-white rounded-lg font-bold hover:bg-indigo-700 shadow-md animate-pulse whitespace-nowrap"
                                        >
                                            Submit Batch
                                        </button>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {completedBatch && (
                        <div className="bg-green-50 border border-green-100 rounded-xl p-8 text-center text-green-800">
                            <CheckCircle className="mx-auto mb-2" size={32} />
                            <h3 className="text-lg font-bold">Batch Submitted!</h3>
                            <p>Loading next tasks...</p>
                        </div>
                    )}

                    {!completedBatch && (
                        <div className="grid gap-4">
                            {tasks.map((task) => {
                                const decision = pendingDecisions[task.task_id];
                                const hasDecision = decision !== undefined;

                                return (
                                    <div key={task.task_id} className={`bg-white rounded-xl shadow border border-gray-200 overflow-hidden transition-all ${hasDecision ? 'ring-2 ring-indigo-50 border-indigo-200' : ''}`}>
                                        <div className="p-4 border-b flex justify-between items-start gap-4">
                                            <div className="text-lg font-medium text-gray-900 leading-relaxed flex-grow">
                                                {task.full_path_content || task.content}
                                            </div>
                                            <div className="text-xs text-gray-400 whitespace-nowrap mt-1">
                                                {task.node_id}
                                            </div>
                                        </div>

                                        <div className="p-3 bg-gray-50 flex gap-3 justify-end">
                                            <button
                                                onClick={() => handleLocalDecision(task.task_id, false)}
                                                className={`flex items-center gap-2 px-4 py-2 border rounded-lg transition-all text-sm font-medium shadow-sm ${hasDecision && !decision
                                                    ? 'bg-red-600 text-white border-red-600'
                                                    : 'bg-white text-gray-600 border-gray-200 hover:border-red-300 hover:bg-red-50 hover:text-red-700'
                                                    }`}
                                            >
                                                <XCircle size={18} />
                                                Irrelevant
                                            </button>

                                            <button
                                                onClick={() => handleLocalDecision(task.task_id, true)}
                                                className={`flex items-center gap-2 px-4 py-2 border rounded-lg transition-all text-sm font-medium shadow-sm ${hasDecision && decision
                                                    ? 'bg-emerald-600 text-white border-emerald-600'
                                                    : 'bg-white text-gray-600 border-gray-200 hover:border-emerald-300 hover:bg-emerald-50 hover:text-emerald-700'
                                                    }`}
                                            >
                                                <CheckCircle size={18} />
                                                Relevant
                                            </button>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
