import React from 'react';
import { type TaxonomyNode } from '../api';
import { Check, X } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

interface NodeCardProps {
    node: TaxonomyNode;
    isRelevant: boolean | null;
    onToggle: (relevant: boolean) => void;
    disabled?: boolean;
}

export const NodeCard: React.FC<NodeCardProps> = ({ node, isRelevant, onToggle, disabled }) => {
    return (
        <div className={twMerge(
            "p-4 rounded-lg border transition-all duration-200 shadow-sm",
            isRelevant === true ? "bg-green-50 border-green-200" :
                isRelevant === false ? "bg-red-50 border-red-200" :
                    "bg-white border-gray-200 hover:shadow-md"
        )}>
            <div className="flex justify-between items-start mb-2">
                <span className="text-xs font-mono text-gray-500">{node.id}</span>
                <div className="flex gap-2">
                    {node.rubric_score !== null && (
                        <span className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded-full font-medium">
                            Score: {node.rubric_score.toFixed(1)}
                        </span>
                    )}
                </div>
            </div>

            <p className="text-gray-800 font-medium mb-4">{node.full_path_content || node.content}</p>

            <div className="flex gap-2">
                <button
                    onClick={() => onToggle(true)}
                    disabled={disabled}
                    className={clsx(
                        "flex-1 flex items-center justify-center gap-2 py-2 rounded-md font-medium transition-colors",
                        isRelevant === true
                            ? "bg-green-600 text-white"
                            : "bg-gray-100 text-gray-600 hover:bg-green-100 hover:text-green-700"
                    )}
                >
                    <Check size={18} />
                    Relevant
                </button>
                <button
                    onClick={() => onToggle(false)}
                    disabled={disabled}
                    className={clsx(
                        "flex-1 flex items-center justify-center gap-2 py-2 rounded-md font-medium transition-colors",
                        isRelevant === false
                            ? "bg-red-600 text-white"
                            : "bg-gray-100 text-gray-600 hover:bg-red-100 hover:text-red-700"
                    )}
                >
                    <X size={18} />
                    Irrelevant
                </button>
            </div>
        </div>
    );
};
