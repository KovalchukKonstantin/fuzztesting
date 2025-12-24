import { useState } from 'react';
import type { TreeNode } from '../api';
import { ChevronRight, ChevronDown, Circle, CheckCircle, XCircle } from 'lucide-react';

interface TreeViewProps {
    tree: TreeNode | null;
    loading: boolean;
}

interface TreeNodeItemProps {
    node: TreeNode;
    defaultExpanded?: boolean;
}

function getScoreColor(score: number | null): string {
    if (score === null || score === undefined) return 'bg-gray-200';
    // Score is 0-10, map to color gradient
    if (score >= 7) return 'bg-emerald-500';
    if (score >= 5) return 'bg-yellow-500';
    if (score >= 3) return 'bg-orange-500';
    return 'bg-red-500';
}

function getStatusBadge(status: string): { color: string; label: string } {
    switch (status) {
        case 'alive':
            return { color: 'bg-blue-100 text-blue-700', label: 'Alive' };
        case 'killed':
            return { color: 'bg-red-100 text-red-700', label: 'Killed' };
        case 'human_verified_relevant':
            return { color: 'bg-green-100 text-green-700', label: 'Verified' };
        default:
            return { color: 'bg-gray-100 text-gray-700', label: status };
    }
}

function TreeNodeItem({ node, defaultExpanded = false }: TreeNodeItemProps) {
    const [expanded, setExpanded] = useState(defaultExpanded || node.depth < 2);
    const hasChildren = node.children && node.children.length > 0;
    const statusBadge = getStatusBadge(node.status);

    return (
        <div className="select-none">
            <div
                className={`flex items-start gap-2 py-2 px-2 rounded-lg hover:bg-gray-50 transition-colors cursor-pointer ${node.human_label !== null ? 'ring-2 ring-offset-1' : ''
                    } ${node.human_label === true ? 'ring-emerald-400 bg-emerald-50/50' :
                        node.human_label === false ? 'ring-red-400 bg-red-50/50' : ''
                    }`}
                onClick={() => hasChildren && setExpanded(!expanded)}
                style={{ marginLeft: `${node.depth * 16}px` }}
            >
                {/* Expand/collapse icon */}
                <div className="w-5 h-5 flex items-center justify-center text-gray-400 flex-shrink-0">
                    {hasChildren ? (
                        expanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />
                    ) : (
                        <Circle size={8} className="text-gray-300" />
                    )}
                </div>

                {/* Human label indicator */}
                <div className="w-5 h-5 flex items-center justify-center flex-shrink-0">
                    {node.human_label === true && <CheckCircle size={16} className="text-emerald-500" />}
                    {node.human_label === false && <XCircle size={16} className="text-red-500" />}
                    {node.human_label === null && <div className="w-4 h-4" />}
                </div>

                {/* Score indicator */}
                <div
                    className={`w-8 h-5 rounded text-white text-xs font-bold flex items-center justify-center flex-shrink-0 ${getScoreColor(node.rubric_score)}`}
                    title={`Rubric Score: ${node.rubric_score?.toFixed(1) ?? 'N/A'}`}
                >
                    {node.rubric_score !== null ? node.rubric_score.toFixed(1) : '-'}
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0">
                    <div className="text-sm text-gray-900 line-clamp-2">{node.content}</div>
                    <div className="flex items-center gap-2 mt-1">
                        <span className={`text-xs px-1.5 py-0.5 rounded ${statusBadge.color}`}>
                            {statusBadge.label}
                        </span>
                        <span className="text-xs text-gray-400">
                            ID: {node.id}
                        </span>
                        {node.ucb_score !== null && (
                            <span className="text-xs text-gray-400">
                                UCB: {node.ucb_score.toFixed(2)}
                            </span>
                        )}
                    </div>
                </div>
            </div>

            {/* Children */}
            {hasChildren && expanded && (
                <div className="border-l border-gray-200 ml-4">
                    {node.children.map(child => (
                        <TreeNodeItem key={child.id} node={child} />
                    ))}
                </div>
            )}
        </div>
    );
}

export function TreeView({ tree, loading }: TreeViewProps) {
    if (loading) {
        return (
            <div className="flex items-center justify-center h-64 text-gray-400">
                Loading tree...
            </div>
        );
    }

    if (!tree) {
        return (
            <div className="flex items-center justify-center h-64 text-gray-400">
                No tree data available. Initialize a project first.
            </div>
        );
    }

    // Count some stats
    let totalNodes = 0;
    let labeledRelevant = 0;
    let labeledIrrelevant = 0;
    let killedNodes = 0;

    const countNodes = (node: TreeNode) => {
        totalNodes++;
        if (node.human_label === true) labeledRelevant++;
        if (node.human_label === false) labeledIrrelevant++;
        if (node.status === 'killed') killedNodes++;
        node.children.forEach(countNodes);
    };
    countNodes(tree);

    return (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
            {/* Header with stats */}
            <div className="p-4 border-b bg-gray-50 flex items-center justify-between">
                <h3 className="font-semibold text-gray-900">Taxonomy Tree</h3>
                <div className="flex items-center gap-4 text-xs">
                    <span className="text-gray-500">{totalNodes} nodes</span>
                    <span className="text-emerald-600">{labeledRelevant} relevant</span>
                    <span className="text-red-600">{labeledIrrelevant} irrelevant</span>
                    <span className="text-gray-400">{killedNodes} killed</span>
                </div>
            </div>

            {/* Legend */}
            <div className="px-4 py-2 border-b bg-gray-50/50 flex items-center gap-4 text-xs text-gray-500">
                <span className="flex items-center gap-1">
                    <div className="w-4 h-3 rounded bg-emerald-500"></div> High Score (7+)
                </span>
                <span className="flex items-center gap-1">
                    <div className="w-4 h-3 rounded bg-yellow-500"></div> Medium (5-7)
                </span>
                <span className="flex items-center gap-1">
                    <div className="w-4 h-3 rounded bg-orange-500"></div> Low (3-5)
                </span>
                <span className="flex items-center gap-1">
                    <div className="w-4 h-3 rounded bg-red-500"></div> Poor (&lt;3)
                </span>
                <span className="flex items-center gap-1 ml-4">
                    <CheckCircle size={12} className="text-emerald-500" /> Labeled Relevant
                </span>
                <span className="flex items-center gap-1">
                    <XCircle size={12} className="text-red-500" /> Labeled Irrelevant
                </span>
            </div>

            {/* Tree content */}
            <div className="p-4 max-h-[600px] overflow-y-auto">
                <TreeNodeItem node={tree} defaultExpanded={true} />
            </div>
        </div>
    );
}
