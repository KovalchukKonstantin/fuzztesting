import { useState, useMemo } from 'react';
import type { TreeNode } from '../api';
import { ChevronRight, ChevronDown, Circle, Clock, SkipBack, SkipForward } from 'lucide-react';

interface TreeEvolutionProps {
    tree: TreeNode | null;
    loading: boolean;
}

interface EvolutionNodeItemProps {
    node: TreeNode;
    currentIteration: number;
    defaultExpanded?: boolean;
}

function EvolutionNodeItem({ node, currentIteration, defaultExpanded = false }: EvolutionNodeItemProps) {
    const [expanded, setExpanded] = useState(defaultExpanded || node.depth < 2);

    // Filter children based on current iteration
    const visibleChildren = useMemo(() => {
        return node.children.filter(child => child.created_iteration <= currentIteration);
    }, [node.children, currentIteration]);

    const hasVisibleChildren = visibleChildren.length > 0;

    // Check if this node is "new" in this iteration
    const isNew = node.created_iteration === currentIteration;

    return (
        <div className="select-none">
            <div
                className={`flex items-start gap-2 py-1.5 px-2 rounded-lg transition-all cursor-pointer border
                    ${isNew
                        ? 'bg-indigo-50 border-indigo-200 shadow-sm ring-1 ring-indigo-300'
                        : 'hover:bg-gray-50 border-transparent'
                    }`}
                onClick={() => hasVisibleChildren && setExpanded(!expanded)}
                style={{ marginLeft: `${node.depth * 20}px` }}
            >
                {/* Expand/collapse icon */}
                <div className="w-5 h-5 flex items-center justify-center text-gray-400 flex-shrink-0">
                    {hasVisibleChildren ? (
                        expanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />
                    ) : (
                        <Circle size={6} className={isNew ? "text-indigo-400" : "text-gray-300"} />
                    )}
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0 flex items-center gap-2">
                    <span className={`text-sm ${isNew ? 'font-semibold text-indigo-900' : 'text-gray-700'}`}>
                        {node.content}
                    </span>

                    {isNew && (
                        <span className="text-[10px] bg-indigo-100 text-indigo-700 px-1.5 py-0.5 rounded-full font-bold uppercase tracking-wide">
                            New
                        </span>
                    )}

                    <span className="text-xs text-gray-400 ml-auto">
                        Iter {node.created_iteration}
                    </span>
                </div>
            </div>

            {/* Children */}
            {hasVisibleChildren && expanded && (
                <div className="border-l border-gray-200 ml-4 pl-1">
                    {visibleChildren.map(child => (
                        <EvolutionNodeItem
                            key={child.id}
                            node={child}
                            currentIteration={currentIteration}
                            defaultExpanded={true} // Auto-expand new nodes usually
                        />
                    ))}
                </div>
            )}
        </div>
    );
}

export function TreeEvolution({ tree, loading }: TreeEvolutionProps) {
    const [currentIteration, setCurrentIteration] = useState<number>(0);

    // Calculate max iteration from tree
    const maxIteration = useMemo(() => {
        if (!tree) return 0;
        let max = 0;
        const traverse = (node: TreeNode) => {
            if (node.created_iteration > max) max = node.created_iteration;
            node.children.forEach(traverse);
        };
        traverse(tree);
        return max;
    }, [tree]);

    // Count stats for current view
    const stats = useMemo(() => {
        if (!tree) return { total: 0, new: 0 };
        let total = 0;
        let newCount = 0;
        const traverse = (node: TreeNode) => {
            if (node.created_iteration <= currentIteration) {
                total++;
                if (node.created_iteration === currentIteration) newCount++;
                node.children.forEach(traverse);
            }
        };
        traverse(tree);
        return { total, new: newCount };
    }, [tree, currentIteration]);

    if (loading) return <div className="p-8 text-center text-gray-400">Loading evolution data...</div>;
    if (!tree) return <div className="p-8 text-center text-gray-400">No tree data available.</div>;

    return (
        <div className="flex flex-col h-[calc(100vh-100px)] gap-4">
            {/* Controls */}
            <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-200 flex flex-col gap-4">
                <div className="flex items-center justify-between">
                    <div>
                        <h2 className="text-lg font-bold text-gray-900 flex items-center gap-2">
                            <Clock className="text-indigo-600" size={20} />
                            Tree Evolution
                        </h2>
                        <p className="text-sm text-gray-500">
                            Viewing Iteration <span className="font-bold text-gray-900">{currentIteration}</span> of {maxIteration}
                        </p>
                    </div>

                    <div className="flex items-center gap-4 text-sm">
                        <div className="bg-gray-100 px-3 py-1.5 rounded-lg text-gray-600">
                            Total Nodes: <strong className="text-gray-900">{stats.total}</strong>
                        </div>
                        <div className="bg-indigo-50 px-3 py-1.5 rounded-lg text-indigo-700">
                            New in this step: <strong className="text-indigo-900">+{stats.new}</strong>
                        </div>
                    </div>
                </div>

                <div className="flex items-center gap-4">
                    <button
                        onClick={() => setCurrentIteration(0)}
                        disabled={currentIteration === 0}
                        className="p-2 hover:bg-gray-100 rounded-lg text-gray-500 disabled:opacity-30"
                    >
                        <SkipBack size={20} />
                    </button>

                    <input
                        type="range"
                        min="0"
                        max={maxIteration}
                        value={currentIteration}
                        onChange={(e) => setCurrentIteration(parseInt(e.target.value))}
                        className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                    />

                    <button
                        onClick={() => setCurrentIteration(maxIteration)}
                        disabled={currentIteration === maxIteration}
                        className="p-2 hover:bg-gray-100 rounded-lg text-gray-500 disabled:opacity-30"
                    >
                        <SkipForward size={20} />
                    </button>
                </div>
            </div>

            {/* Tree View */}
            <div className="flex-1 bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden flex flex-col">
                <div className="p-4 border-b bg-gray-50 text-xs text-gray-500 flex items-center justify-between">
                    <span>Taxonomy state at iteration #{currentIteration}</span>
                    {currentIteration === 0 && <span className="italic">Initial Seed State</span>}
                </div>

                <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
                    {tree.created_iteration <= currentIteration && (
                        <EvolutionNodeItem
                            node={tree}
                            currentIteration={currentIteration}
                            defaultExpanded={true}
                        />
                    )}
                </div>
            </div>
        </div>
    );
}
