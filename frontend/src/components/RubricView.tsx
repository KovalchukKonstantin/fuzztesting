import { useState, useEffect } from 'react';
import { api, type Rubric, type NodeScore } from '../api';
import { ChevronRight, ChevronDown, Award, Info, Search, RefreshCw } from 'lucide-react';
import { twMerge } from 'tailwind-merge';

export function RubricView() {
    const [rubric, setRubric] = useState<Rubric | null>(null);
    const [allRubrics, setAllRubrics] = useState<Rubric[]>([]);
    const [currentIndex, setCurrentIndex] = useState<number>(-1);
    const [scores, setScores] = useState<NodeScore[]>([]);
    const [loading, setLoading] = useState(false);
    const [searchTerm, setSearchTerm] = useState('');
    const [expandedNodes, setExpandedNodes] = useState<Record<string, boolean>>({});
    const [selectedPrincipleId, setSelectedPrincipleId] = useState<string | null>(null);

    useEffect(() => {
        loadLatest();
    }, []);

    const loadLatest = async () => {
        setLoading(true);
        try {
            const rubrics = await api.getRubrics();
            setAllRubrics(rubrics);

            if (rubrics.length > 0) {
                // Default to latest
                const idx = rubrics.length - 1;
                setCurrentIndex(idx);
                await loadRubricByIndex(idx, rubrics);
            } else {
                setRubric(null);
                setScores([]);
                setCurrentIndex(-1);
            }
        } catch (e) {
            console.error("Failed to load rubrics", e);
        } finally {
            setLoading(false);
        }
    };

    const loadRubricByIndex = async (index: number, rubricsList: Rubric[]) => {
        if (index < 0 || index >= rubricsList.length) return;

        const target = rubricsList[index];
        setRubric(target);
        setLoading(true);
        setSelectedPrincipleId(null);
        try {
            const scoreData = await api.getRubricScores(target.iteration);
            setScores(scoreData);
        } catch (e) {
            console.error("Failed to load scores", e);
            setScores([]);
        } finally {
            setLoading(false);
        }
    };

    const handlePrev = async () => {
        if (currentIndex > 0) {
            const newIndex = currentIndex - 1;
            setCurrentIndex(newIndex);
            await loadRubricByIndex(newIndex, allRubrics);
        }
    };

    const handleNext = async () => {
        if (currentIndex < allRubrics.length - 1) {
            const newIndex = currentIndex + 1;
            setCurrentIndex(newIndex);
            await loadRubricByIndex(newIndex, allRubrics);
        }
    };

    const toggleNode = (nodeId: string) => {
        setExpandedNodes(prev => ({ ...prev, [nodeId]: !prev[nodeId] }));
    };

    const togglePrinciple = (id: string) => {
        setSelectedPrincipleId(prev => prev === id ? null : id);
    };

    const filteredScores = scores
        .filter(s => s.content.toLowerCase().includes(searchTerm.toLowerCase()))
        .sort((a, b) => {
            if (selectedPrincipleId) {
                const scoreA = a.principle_scores.find(ps => ps.principle_id === selectedPrincipleId)?.score || 0;
                const scoreB = b.principle_scores.find(ps => ps.principle_id === selectedPrincipleId)?.score || 0;
                return scoreB - scoreA;
            }
            return b.aggregate_score - a.aggregate_score;
        });

    return (
        <div className="flex flex-col h-full gap-6">
            <div className="flex flex-col lg:flex-row gap-6">
                {/* Rubric Details Panel */}
                <aside className="w-full lg:w-80 flex-shrink-0 space-y-4">
                    <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-5">
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider">Rubric Iteration</h3>
                            <div className="flex items-center gap-2">
                                <button
                                    onClick={handlePrev}
                                    disabled={currentIndex <= 0 || loading}
                                    className="p-1 hover:bg-gray-100 rounded disabled:opacity-30"
                                >
                                    <ChevronDown className="rotate-90" size={14} />
                                </button>
                                {rubric && (
                                    <span className="text-[10px] font-bold bg-indigo-50 text-indigo-600 px-2 py-0.5 rounded-full border border-indigo-100">
                                        #{rubric.iteration}
                                    </span>
                                )}
                                <button
                                    onClick={handleNext}
                                    disabled={currentIndex >= allRubrics.length - 1 || loading}
                                    className="p-1 hover:bg-gray-100 rounded disabled:opacity-30"
                                >
                                    <ChevronRight size={14} />
                                </button>
                            </div>
                        </div>

                        {!rubric ? (
                            <div className="text-sm text-gray-400 py-4 italic text-center">No rubric points defined yet.</div>
                        ) : (
                            <ul className="space-y-4">
                                {rubric.principles.map((p) => (
                                    <li
                                        key={p.id}
                                        onClick={() => togglePrinciple(p.id)}
                                        className={twMerge(
                                            "relative cursor-pointer group transition-all p-2 -m-2 rounded-lg",
                                            selectedPrincipleId === p.id ? "bg-indigo-50 ring-1 ring-indigo-100 shadow-sm" : "hover:bg-gray-50"
                                        )}
                                    >
                                        <p className={twMerge(
                                            "text-xs leading-relaxed font-medium",
                                            selectedPrincipleId === p.id ? "text-indigo-900" : "text-gray-700"
                                        )}>
                                            {p.description}
                                        </p>
                                        <div className="mt-1 flex items-center gap-1.5 pl-1">
                                            <div className="h-1 flex-1 bg-gray-100 rounded-full overflow-hidden">
                                                <div
                                                    className={twMerge("h-full", selectedPrincipleId === p.id ? "bg-indigo-500" : "bg-gray-300")}
                                                    style={{ width: `${p.weight * 100}%` }}
                                                ></div>
                                            </div>
                                            <span className="text-[9px] text-gray-400 font-bold">W:{p.weight}</span>
                                        </div>
                                    </li>
                                ))}
                            </ul>
                        )}
                    </div>

                    <button
                        onClick={loadLatest}
                        disabled={loading}
                        className="w-full py-2.5 bg-gray-900 text-white rounded-xl text-sm font-bold flex items-center justify-center gap-2 hover:bg-gray-800 transition-colors shadow-lg active:scale-95 disabled:opacity-50"
                    >
                        <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
                        Refresh Scores
                    </button>
                </aside>

                {/* Scores Table/List */}
                <main className="flex-1 min-w-0">
                    <div className="bg-white rounded-xl shadow-sm border border-gray-100 flex flex-col h-full">
                        <div className="p-4 border-b flex items-center justify-between gap-4">
                            <div className="flex items-center gap-2">
                                <Award className="text-indigo-600" size={20} />
                                <h2 className="font-bold text-gray-900">
                                    Scores for Rubric #{rubric?.iteration}
                                </h2>
                            </div>
                            <div className="relative flex-1 max-w-sm">
                                <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={16} />
                                <input
                                    type="text"
                                    placeholder="Filter nodes..."
                                    value={searchTerm}
                                    onChange={(e) => setSearchTerm(e.target.value)}
                                    className="w-full pl-10 pr-4 py-2 bg-gray-50 border border-gray-200 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 outline-none transition-all"
                                />
                            </div>
                        </div>

                        <div className="flex-1 overflow-auto custom-scrollbar min-h-[400px]">
                            {loading ? (
                                <div className="flex items-center justify-center p-12 text-gray-400 animate-pulse font-medium">
                                    <RefreshCw size={20} className="animate-spin mr-2" />
                                    Loading scores...
                                </div>
                            ) : filteredScores.length === 0 ? (
                                <div className="p-12 text-center text-gray-400 italic">
                                    {searchTerm ? 'No nodes match search' : 'No scores found for this rubric'}
                                </div>
                            ) : (
                                <div className="divide-y">
                                    {filteredScores.map(score => (
                                        <div key={score.node_id} className="group">
                                            <div
                                                onClick={() => toggleNode(score.node_id)}
                                                className="p-4 flex items-center justify-between gap-4 cursor-pointer hover:bg-gray-50 transition-colors"
                                            >
                                                <div className="flex items-start gap-3 min-w-0 flex-1">
                                                    <div className="mt-1 flex-shrink-0">
                                                        {expandedNodes[score.node_id] ? <ChevronDown size={16} className="text-gray-400" /> : <ChevronRight size={16} className="text-gray-400" />}
                                                    </div>
                                                    <span className="text-sm font-medium text-gray-900 leading-snug break-words">
                                                        {score.full_path_content || score.content}
                                                    </span>
                                                </div>
                                                {/* Individual scores removed as per user request */}
                                                <div className="w-16 text-right border-l pl-3 ml-1 flex flex-col justify-center">
                                                    <span className="text-[10px] text-gray-400 uppercase font-bold leading-none mb-0.5">
                                                        {selectedPrincipleId ? 'Score' : 'Agg'}
                                                    </span>
                                                    <span className="text-sm font-black text-gray-900 leading-none">
                                                        {selectedPrincipleId
                                                            ? (score.principle_scores.find(ps => ps.principle_id === selectedPrincipleId)?.score || 0).toFixed(1)
                                                            : score.aggregate_score.toFixed(1)
                                                        }
                                                    </span>
                                                </div>
                                            </div>

                                            {expandedNodes[score.node_id] && (
                                                <div className="px-12 pb-4 space-y-3">
                                                    {score.principle_scores.map((ps, i) => {
                                                        const principle = rubric?.principles.find(p => p.id === ps.principle_id);
                                                        return (
                                                            <div key={ps.principle_id} className="bg-gray-50 rounded-lg p-3 text-xs border border-gray-100">
                                                                <div className="flex items-center justify-between mb-1.5">
                                                                    <span className="font-bold text-gray-700">P{i + 1}: {principle?.description || 'Unknown Principle'}</span>
                                                                    <span className={`px-2 py-0.5 rounded-full font-bold ${ps.score >= 8 ? 'bg-emerald-100 text-emerald-700' :
                                                                        ps.score >= 5 ? 'bg-amber-100 text-amber-700' :
                                                                            'bg-rose-100 text-rose-700'
                                                                        }`}>
                                                                        {ps.score}/10
                                                                    </span>
                                                                </div>
                                                                <div className="text-gray-600 italic leading-relaxed">
                                                                    <Info size={12} className="inline mr-1 text-gray-400" />
                                                                    {ps.reasoning || 'No reasoning provided.'}
                                                                </div>
                                                            </div>
                                                        );
                                                    })}
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                </main>
            </div>
        </div>
    );
}
