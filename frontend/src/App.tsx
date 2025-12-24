import { useState, useEffect } from 'react';
import { api, type TaxonomyNode, type StatsResponse, type AutoResult, type TreeNode } from './api';
import { NodeCard } from './components/NodeCard';
import { TreeView } from './components/TreeView';
import { Play, RefreshCw, BarChart2, BookOpen, Layers, Zap, GitBranch, Award, TrendingUp, Users, Clock } from 'lucide-react';
import { RubricView } from './components/RubricView';
import { ContractorDashboard } from './components/ContractorDashboard';
import RubricMetricsView from './components/RubricMetricsView';
import { ProjectSwitcher } from './components/ProjectSwitcher';
import { TreeEvolution } from './components/TreeEvolution';

function App() {
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [nodes, setNodes] = useState<TaxonomyNode[]>([]);
  const [decisions, setDecisions] = useState<Record<string, boolean>>({});
  const [loading, setLoading] = useState(false);
  const [autoResults, setAutoResults] = useState<AutoResult[]>([]);
  const [mainTab, setMainTab] = useState<'taxonomy' | 'tree' | 'evolution' | 'rubrics' | 'metrics' | 'contractor'>('taxonomy');
  const [taxonomyMode, setTaxonomyMode] = useState<'manual' | 'auto'>('manual');
  const [numIterations, setNumIterations] = useState<number | string>(1);
  const [treeData, setTreeData] = useState<TreeNode | null>(null);
  const [treeLoading, setTreeLoading] = useState(false);
  const [projectId, setProjectId] = useState<string>("default_project");

  // ... (keeping existing hooks) ...

  // ... inside return ...
  // Load stats on mount
  useEffect(() => {
    loadStats();
    const interval = setInterval(loadStats, 5000);
    return () => clearInterval(interval);
  }, []);

  // Load tree when tab becomes active
  useEffect(() => {
    if (mainTab === 'tree' || mainTab === 'evolution') {
      loadTree();
    }
  }, [mainTab]);

  const loadStats = async () => {
    try {
      const data = await api.getState();
      setStats(data);
    } catch (e) {
      console.error("Failed to load stats", e);
    }
  };

  const loadTree = async () => {
    setTreeLoading(true);
    try {
      const data = await api.getTree();
      setTreeData(data);
    } catch (e) {
      console.error("Failed to load tree", e);
    } finally {
      setTreeLoading(false);
    }
  };

  // Init Logic
  const [showInitModal, setShowInitModal] = useState(false);
  const [initProjectName, setInitProjectName] = useState("demo_project");
  const [initDescription, setInitDescription] = useState("A smart home security camera with AI features.");


  const handleConfirmInit = async () => {
    setLoading(true);
    setShowInitModal(false);
    try {
      await api.init(initProjectName, initDescription);
      setProjectId(initProjectName);
      await loadStats();
      setNodes([]);
      setDecisions({});
      setTreeData(null);
      setAutoResults([]);
      alert("Project initialized!");
    } catch (e) {
      alert("Error initializing: " + e);
    } finally {
      setLoading(false);
    }
  };

  const handleNextIteration = async () => {
    setLoading(true);
    try {
      const data = await api.getNextIteration();
      setNodes(data.nodes_for_grading);
      setDecisions({});
    } catch (e) {
      alert("Error getting next iteration: " + e);
    } finally {
      setLoading(false);
    }
  };

  const handleToggleDecision = (nodeId: string, isRelevant: boolean) => {
    setDecisions(prev => ({ ...prev, [nodeId]: isRelevant }));
  };

  const handleSubmitFeedback = async () => {
    const items = Object.entries(decisions).map(([nodeId, isRelevant]) => ({
      node_id: nodeId,
      is_relevant: isRelevant
    }));

    if (items.length !== nodes.length) {
      if (!confirm(`You only graded ${items.length}/${nodes.length} nodes. Submit anyway?`)) return;
    }

    setLoading(true);
    try {
      await api.submitFeedback(items);
      setNodes([]);
      setDecisions({});
      await loadStats();
    } catch (e) {
      alert("Error submitting feedback: " + e);
    } finally {
      setLoading(false);
    }
  };

  const handleRunAuto = async (mode: 'random' | 'llm') => {
    setLoading(true);
    try {
      const iterations = typeof numIterations === 'string' ? (parseInt(numIterations) || 1) : numIterations;
      const res = await api.runAuto(iterations, mode);
      setAutoResults(res.results);
      await loadStats();
    } catch (e) {
      alert("Auto run failed: " + e);
    } finally {
      setLoading(false);
    }
  }

  const handleProjectSwitch = async (newProjectId: string) => {
    setProjectId(newProjectId);
    // Reload everything
    setLoading(true);
    try {
      await loadStats();
      setNodes([]);
      setDecisions({});
      setTreeData(null);
      setAutoResults([]);

      // If we are on tree tab, reload tree
      if (mainTab === 'tree' || mainTab === 'evolution') {
        await loadTree();
      }
    } finally {
      setLoading(false);
    }
  };

  const handleNewProject = () => {
    // For now, reusing init modal but clearing fields?
    // Or just opening init modal which serves as "Create/Reset Current"
    // The init endpoint is actually "initialize/create", so we can use it.
    // However, the Init Modal is currently "Reset Project".
    // We should probably adapt it or just reuse it for now.
    setInitProjectName("");
    setInitDescription("");
    setShowInitModal(true);
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col font-sans">
      {/* Header with top-level tabs */}
      <header className="bg-white border-b sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <Layers className="text-indigo-600" />
              <h1 className="text-xl font-bold text-gray-900">Edge Case Sampler</h1>
            </div>

            <ProjectSwitcher
              onSwitch={handleProjectSwitch}
              onNewProject={handleNewProject}
            />

            {/* Top-level tab navigation */}
            <div className="flex p-1 bg-gray-100 rounded-lg">
              <button
                onClick={() => setMainTab('taxonomy')}
                className={`px-4 py-1.5 text-sm font-medium rounded-md transition-all ${mainTab === 'taxonomy' ? 'bg-white shadow-sm text-gray-900' : 'text-gray-500 hover:text-gray-700'}`}
              >
                <Layers size={14} className="inline mr-1.5" />
                Taxonomy Generation
              </button>
              <button
                onClick={() => setMainTab('tree')}
                className={`px-4 py-1.5 text-sm font-medium rounded-md transition-all ${mainTab === 'tree' ? 'bg-white shadow-sm text-gray-900' : 'text-gray-500 hover:text-gray-700'}`}
              >
                <GitBranch size={14} className="inline mr-1.5" />
                Tree View
              </button>
              <button
                onClick={() => setMainTab('evolution')}
                className={`px-4 py-1.5 text-sm font-medium rounded-md transition-all ${mainTab === 'evolution' ? 'bg-white shadow-sm text-gray-900' : 'text-gray-500 hover:text-gray-700'}`}
              >
                <Clock size={14} className="inline mr-1.5" />
                Evolution
              </button>
              <button
                onClick={() => setMainTab('rubrics')}
                className={`px-4 py-1.5 text-sm font-medium rounded-md transition-all ${mainTab === 'rubrics' ? 'bg-white shadow-sm text-gray-900' : 'text-gray-500 hover:text-gray-700'}`}
              >
                <Award size={14} className="inline mr-1.5" />
                Rubrics
              </button>
              <button
                onClick={() => setMainTab('metrics')}
                className={`px-4 py-1.5 text-sm font-medium rounded-md transition-all ${mainTab === 'metrics' ? 'bg-white shadow-sm text-gray-900' : 'text-gray-500 hover:text-gray-700'}`}
              >
                <TrendingUp size={14} className="inline mr-1.5" />
                Metrics
              </button>
              <button
                onClick={() => setMainTab('contractor')}
                className={`px-4 py-1.5 text-sm font-medium rounded-md transition-all ${mainTab === 'contractor' ? 'bg-white shadow-sm text-gray-900' : 'text-gray-500 hover:text-gray-700'}`}
              >
                <Users size={14} className="inline mr-1.5" />
                Contractor
              </button>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {/* Stats */}
            {stats && (
              <div className="flex items-center gap-4 text-sm text-gray-600">
                <div className="flex items-center gap-1"><BookOpen size={14} /> {stats.total_rubrics} rubrics</div>
                <div className="flex items-center gap-1"><BarChart2 size={14} /> {stats.total_labeled} labeled</div>
                <div className="bg-indigo-50 text-indigo-700 px-3 py-1 rounded-full font-medium">{stats.total_nodes} nodes</div>
              </div>
            )}


          </div>
        </div>
      </header>

      {/* Tree View Tab */}
      {mainTab === 'tree' && (
        <main className="flex-1 max-w-7xl mx-auto w-full p-6">
          <TreeView key={projectId} tree={treeData} loading={treeLoading} />
        </main>
      )}

      {/* Evolution Tab */}
      {mainTab === 'evolution' && (
        <main className="flex-1 max-w-7xl mx-auto w-full p-6">
          <TreeEvolution key={projectId} tree={treeData} loading={treeLoading} />
        </main>
      )}

      {/* Rubrics Tab */}
      {mainTab === 'rubrics' && (
        <main className="flex-1 max-w-7xl mx-auto w-full p-6 overflow-hidden">
          <RubricView key={projectId} />
        </main>
      )}

      {/* Metrics Tab */}
      {mainTab === 'metrics' && (
        <main className="flex-1 max-w-7xl mx-auto w-full p-6 overflow-hidden">
          <RubricMetricsView key={projectId} />
        </main>
      )}

      {/* Contractor Tab */}
      {mainTab === 'contractor' && (
        <main className="flex-1 max-w-7xl mx-auto w-full bg-gray-50">
          <ContractorDashboard key={projectId} />
        </main>
      )}

      {/* Taxonomy Tab */}
      {mainTab === 'taxonomy' && (
        <main className="flex-1 max-w-7xl mx-auto w-full p-6 grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sidebar Controls */}
          <aside className="lg:col-span-1 space-y-6">
            {/* Mode switcher */}
            <div className="bg-white rounded-xl shadow-sm p-5 border border-gray-100">
              <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-4">Build Mode</h3>
              <div className="flex p-1 bg-gray-100 rounded-lg mb-4">
                <button
                  onClick={() => setTaxonomyMode('manual')}
                  className={`flex-1 py-1.5 text-sm font-medium rounded-md transition-all ${taxonomyMode === 'manual' ? 'bg-white shadow-sm text-gray-900' : 'text-gray-500'}`}
                >
                  Manual
                </button>
                <button
                  onClick={() => setTaxonomyMode('auto')}
                  className={`flex-1 py-1.5 text-sm font-medium rounded-md transition-all ${taxonomyMode === 'auto' ? 'bg-white shadow-sm text-gray-900' : 'text-gray-500'}`}
                >
                  Auto
                </button>
              </div>

              {taxonomyMode === 'manual' ? (
                <button
                  onClick={handleNextIteration}
                  disabled={loading || nodes.length > 0}
                  className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-indigo-600 text-white rounded-lg font-bold hover:bg-indigo-700 transition-shadow shadow-md disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Play size={18} fill="currentColor" />
                  Run Iteration
                </button>
              ) : (
                <div className="space-y-3">
                  <div>
                    <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1 block">Iterations</label>
                    <input
                      type="number"
                      min="1"
                      max="50"
                      value={numIterations}
                      onChange={(e) => setNumIterations(Math.max(1, parseInt(e.target.value) || 1))}
                      className="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 outline-none transition-all"
                    />
                  </div>
                  <button
                    onClick={() => handleRunAuto('random')}
                    disabled={loading}
                    className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-fuchsia-600 text-white rounded-lg font-bold hover:bg-fuchsia-700 transition-all shadow-md disabled:opacity-50"
                  >
                    <Zap size={18} fill="currentColor" />
                    Run {numIterations}x Random
                  </button>
                  <button
                    onClick={() => handleRunAuto('llm')}
                    disabled={loading}
                    className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-emerald-600 text-white rounded-lg font-bold hover:bg-emerald-700 transition-all shadow-md disabled:opacity-50"
                  >
                    <Zap size={18} fill="currentColor" />
                    Run {numIterations}x LLM
                  </button>
                </div>
              )}
            </div>

            {/* Enlarged Rubric Display */}
            {stats && stats.current_rubric_principles.length > 0 && (
              <div className="bg-white rounded-xl shadow-sm p-5 border border-gray-100">
                <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
                  Current Rubric ({stats.current_rubric_principles.length} principles)
                </h3>
                <ul className="space-y-3 max-h-[500px] overflow-y-auto pr-2 custom-scrollbar">
                  {stats.current_rubric_principles.map((p, i) => (
                    <li key={i} className="text-sm text-gray-700 p-3 bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg border border-indigo-100">
                      <span className="inline-block bg-indigo-600 text-white text-xs font-bold w-6 h-6 rounded-full text-center leading-6 mr-2">
                        {i + 1}
                      </span>
                      {p}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </aside>

          {/* Main Content Area */}
          <div className="lg:col-span-3 space-y-6">
            {taxonomyMode === 'manual' && (
              <>
                {nodes.length === 0 && !loading && (
                  <div className="h-full flex flex-col items-center justify-center text-center p-12 bg-white rounded-xl border border-dashed border-gray-300">
                    <div className="w-16 h-16 bg-indigo-50 text-indigo-200 rounded-full flex items-center justify-center mb-4">
                      <Play size={32} />
                    </div>
                    <h3 className="text-lg font-medium text-gray-900 mb-2">Ready to sample</h3>
                    <p className="text-gray-500 max-w-sm">Click "Run Iteration" to generate the next batch of edge cases for review.</p>
                  </div>
                )}

                {nodes.length > 0 && (
                  <>
                    <div className="flex items-center justify-between">
                      <h2 className="text-lg font-bold text-gray-900">Review Candidates ({nodes.length})</h2>
                      <button
                        onClick={handleSubmitFeedback}
                        disabled={loading}
                        className="px-6 py-2 bg-black text-white rounded-lg font-medium hover:bg-gray-800 transition-colors shadow-lg disabled:opacity-50"
                      >
                        Submit Feedback
                      </button>
                    </div>

                    <div className="grid grid-cols-1 gap-4">
                      {nodes.map(node => (
                        <NodeCard
                          key={node.id}
                          node={node}
                          isRelevant={decisions[node.id] ?? null}
                          onToggle={(val) => handleToggleDecision(node.id, val)}
                          disabled={loading}
                        />
                      ))}
                    </div>
                  </>
                )}
              </>
            )}

            {taxonomyMode === 'auto' && (
              <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
                <div className="p-4 border-b bg-gray-50 flex justify-between items-center">
                  <h3 className="font-semibold text-gray-900">Auto-Run Results</h3>
                  {autoResults.length > 0 && (
                    <button onClick={() => setAutoResults([])} className="text-xs text-gray-500 hover:text-red-500">Clear</button>
                  )}
                </div>
                <div className="p-0">
                  {autoResults.length === 0 ? (
                    <div className="p-8 text-center text-gray-400">No results yet. Run an auto-job.</div>
                  ) : (
                    <table className="w-full text-sm text-left">
                      <thead className="bg-gray-50 text-gray-500">
                        <tr>
                          <th className="px-4 py-3 font-medium">Iter</th>
                          <th className="px-4 py-3 font-medium">New Nodes</th>
                          <th className="px-4 py-3 font-medium">Rubric Updated?</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y">
                        {autoResults.map((res, i) => (
                          <tr key={i} className="hover:bg-gray-50">
                            <td className="px-4 py-3 text-gray-900">#{res.iteration}</td>
                            <td className="px-4 py-3 font-medium text-emerald-600">+{res.new_nodes}</td>
                            <td className="px-4 py-3 text-gray-600">{res.rubric_updated ? 'Yes' : 'No'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              </div>
            )}
          </div>
        </main>
      )}

      {loading && (
        <div className="fixed inset-0 bg-white/50 backdrop-blur-sm z-50 flex items-center justify-center">
          <div className="flex flex-col items-center bg-white p-8 rounded-xl shadow-2xl border">
            <div className="animate-spin text-indigo-600 mb-4"><RefreshCw size={32} /></div>
            <p className="text-gray-900 font-medium">Processing...</p>
          </div>
        </div>
      )}

      {/* Init Modal */}
      {showInitModal && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-xl shadow-2xl w-full max-w-lg overflow-hidden">
            <div className="px-6 py-4 border-b bg-gray-50 flex justify-between items-center">
              <h3 className="text-lg font-bold text-gray-900">Initialize Project</h3>
              <button
                onClick={() => setShowInitModal(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                ✕
              </button>
            </div>

            <div className="p-6 space-y-4">


              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Project Name</label>
                <input
                  type="text"
                  value={initProjectName}
                  onChange={(e) => setInitProjectName(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Product Description / Spec</label>
                <textarea
                  value={initDescription}
                  onChange={(e) => setInitDescription(e.target.value)}
                  className="w-full h-32 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none resize-none"
                  placeholder="Describe your product in detail..."
                />
                <p className="text-xs text-gray-500 mt-1">This context is used to generate relevant edge cases.</p>
              </div>
            </div>

            <div className="px-6 py-4 bg-gray-50 flex justify-end gap-3 border-t">
              <button
                onClick={() => setShowInitModal(false)}
                className="px-4 py-2 text-gray-700 font-medium hover:bg-gray-100 rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmInit}
                className="px-4 py-2 bg-indigo-600 text-white font-medium rounded-lg hover:bg-indigo-700 shadow-md transition-colors"
              >
                Initialize Project
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
