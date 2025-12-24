import { useState, useRef, useEffect } from 'react';
import { ChevronDown, Folder, Check, FolderPlus, Trash2 } from 'lucide-react';
import { api } from '../api';

interface Project {
    id: string;
    product_description: string;
    created_at: string;
}

interface ProjectSwitcherProps {
    onSwitch: (projectId: string) => void;
    onNewProject: () => void;
}

export function ProjectSwitcher({ onSwitch, onNewProject }: ProjectSwitcherProps) {
    const [isOpen, setIsOpen] = useState(false);
    const [projects, setProjects] = useState<Project[]>([]);
    const [currentId, setCurrentId] = useState<string>("");
    const [loading, setLoading] = useState(false);
    const dropdownRef = useRef<HTMLDivElement>(null);

    const fetchProjects = async () => {
        try {
            setLoading(true);
            const data = await api.getProjects();
            setProjects(data.projects);
            setCurrentId(data.current_project_id);
        } catch (e) {
            console.error("Failed to fetch projects", e);
        } finally {
            setLoading(false);
        }
    };

    // Close dropdown when clicking outside
    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        }
        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, []);

    // Fetch projects when opening
    useEffect(() => {
        if (isOpen) {
            fetchProjects();
        }
    }, [isOpen]);

    // Initial fetch to get current project ID for the button label
    useEffect(() => {
        fetchProjects();
    }, [onSwitch]); // Refetch if parent signals a switch happened externally (like new project created)

    const handleDelete = async (projectId: string, e: React.MouseEvent) => {
        e.stopPropagation();
        if (!window.confirm(`Are you sure you want to delete project '${projectId}'?`)) {
            return;
        }

        try {
            await api.deleteProject(projectId);
            if (projectId === currentId) {
                window.location.reload();
            } else {
                fetchProjects();
            }
        } catch (e) {
            alert("Failed to delete project: " + e);
        }
    };

    const handleSwitch = async (projectId: string) => {
        if (projectId === currentId) {
            setIsOpen(false);
            return;
        }

        try {
            await api.switchProject(projectId);
            setIsOpen(false);
            setCurrentId(projectId);
            onSwitch(projectId); // Signal parent to reload data
        } catch (e) {
            alert("Failed to switch project: " + e);
        }
    };

    const currentProject = projects.find(p => p.id === currentId);

    return (
        <div className="relative" ref={dropdownRef}>
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg hover:bg-gray-100 transition-colors text-sm font-medium text-gray-700 border border-transparent hover:border-gray-200"
            >
                <span className="w-6 h-6 bg-indigo-100 text-indigo-700 rounded-md flex items-center justify-center">
                    <Folder size={14} />
                </span>
                <span className="max-w-[150px] truncate">
                    {currentProject ? currentProject.id : (loading ? "Loading..." : "Select Project")}
                </span>
                <ChevronDown size={14} className={`text-gray-400 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
            </button>

            {isOpen && (
                <div className="absolute top-full left-0 mt-2 w-96 bg-white rounded-xl shadow-xl border border-gray-100 py-2 z-50">
                    <div className="px-3 pb-2 border-b border-gray-50 flex justify-between items-center">
                        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Projects</span>
                        {loading && <div className="text-xs text-indigo-500 animate-pulse">Syncing...</div>}
                    </div>

                    <div className="max-h-[300px] overflow-y-auto py-1">
                        {projects.map(p => (
                            <div key={p.id} className="w-full flex items-center justify-between gap-2 px-2 py-1 group hover:bg-gray-50 rounded-lg transition-colors">
                                <button
                                    onClick={() => handleSwitch(p.id)}
                                    className="flex-1 flex items-center gap-3 p-2 text-left"
                                >
                                    <div className={`flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center ${p.id === currentId ? 'bg-indigo-50 text-indigo-600' : 'bg-gray-100 text-gray-400 group-hover:bg-gray-200 transition-colors'}`}>
                                        {p.id === currentId ? <Check size={16} /> : <Folder size={16} />}
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <div className={`text-sm font-medium truncate ${p.id === currentId ? 'text-indigo-700' : 'text-gray-700'}`}>
                                            {p.id}
                                        </div>
                                        <div className="text-xs text-gray-400 truncate">
                                            {p.product_description.slice(0, 40)}{p.product_description.length > 40 ? '...' : ''}
                                        </div>
                                    </div>
                                </button>

                                <button
                                    onClick={(e) => handleDelete(p.id, e)}
                                    className="p-2 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded-md transition-all opacity-0 group-hover:opacity-100"
                                    title="Delete Project"
                                >
                                    <Trash2 size={16} />
                                </button>
                            </div>
                        ))}
                    </div>

                    <div className="pt-2 mt-1 border-t border-gray-50 px-2">
                        <button
                            onClick={() => {
                                setIsOpen(false);
                                onNewProject();
                            }}
                            className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-gray-50 hover:bg-gray-100 text-gray-700 rounded-lg text-sm font-medium transition-colors border border-dashed border-gray-200"
                        >
                            <FolderPlus size={16} />
                            Create New Project
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
